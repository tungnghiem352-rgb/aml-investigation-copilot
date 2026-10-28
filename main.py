import os
import re
import io
import tempfile
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st
from google import genai
from google.genai import types

# ==============================================================================
# CẤU HÌNH TRANG WEB STREAMLIT & SESSION STATE
# ==============================================================================
st.set_page_config(page_title="Hệ Thống AML Copilot", page_icon="🛡️", layout="wide")

# Khởi tạo các biến toàn cục cho ứng dụng web (thay thế cho biến global của Python)
if 'processed_df' not in st.session_state:
    st.session_state['processed_df'] = pd.DataFrame()
if 'customer_list' not in st.session_state:
    st.session_state['customer_list'] = []
if 'llm_report' not in st.session_state:
    st.session_state['llm_report'] = ""
if 'qa_result' not in st.session_state:
    st.session_state['qa_result'] = ""

# ==============================================================================
# PHẦN 1: LOGIC KIỂM ĐỊNH AI (QA EVALUATOR)
# ==============================================================================
class AMLReportEvaluator:
    def __init__(self, df_case: pd.DataFrame):
        self.df_case = df_case
        self.valid_numbers = self._extract_valid_numbers()

    def _extract_valid_numbers(self) -> set:
        num_set = set()
        if self.df_case.empty: return num_set
        for val in self.df_case['amount']: num_set.add(round(float(val), 2))
        num_set.add(round(float(self.df_case['amount'].sum()), 2))
        if 'Risk_Score' in self.df_case.columns: num_set.add(round(float(self.df_case['Risk_Score'].max()), 2))
        num_set.add(float(len(self.df_case)))
        return num_set

    @staticmethod
    def _parse_numbers_from_text(text: str) -> list:
        clean_text = re.sub(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', '', text)
        clean_text = re.sub(r'\b\d{2,4}[-/]\d{1,2}[-/]\d{2,4}\b', '', clean_text)
        clean_text = re.sub(r'\b[A-Za-z_]+\d+\b', '', clean_text)
        clean_text = re.sub(r'(VNĐ|VND|USD|\$|đ|triệu|tỷ|ngàn|k)', '', clean_text, flags=re.IGNORECASE)

        pattern = r'\b\d+(?:[.,]\d+)*\b'
        matches = re.findall(pattern, clean_text)
        results = []
        for num_str in matches:
            if ',' in num_str and '.' in num_str:
                if num_str.rfind(',') > num_str.rfind('.'): num_str = num_str.replace('.', '').replace(',', '.')
                else: num_str = num_str.replace(',', '')
            elif ',' in num_str:
                if len(num_str.split(',')[-1]) == 3: num_str = num_str.replace(',', '')
                else: num_str = num_str.replace(',', '.')
            try: results.append(float(num_str))
            except ValueError: continue
        return results

    def verify_traceability(self, llm_report: str, tolerance: float = 1.0) -> dict:
        reported_numbers = self._parse_numbers_from_text(llm_report)
        filtered_numbers = [n for n in reported_numbers if n > 9]
        if not filtered_numbers: return {"traceability_score": 100.0, "matched_figures": [], "unmatched_figures": []}

        matched, unmatched = [], []
        for num in filtered_numbers:
            if any(abs(round(num, 2) - ex_num) <= tolerance for ex_num in self.valid_numbers): matched.append(num)
            else: unmatched.append(num)

        return {"traceability_score": round(len(matched) / len(filtered_numbers) * 100, 2), "matched_figures": matched, "unmatched_figures": unmatched}

    def verify_usefulness(self, llm_report: str) -> dict:
        score = 100
        missing_kw = [kw for kw in ["Structuring", "Fan-out", "Shared Beneficiary", "giao dịch"] if kw.lower() not in llm_report.lower()]
        if missing_kw: score -= len(missing_kw) * 10
        if len(llm_report.strip().split('\n')) < 4: score -= 15
        return {"usefulness_score": max(0, round(score, 2))}

    def evaluate(self, llm_report: str) -> dict:
        t_res = self.verify_traceability(llm_report)
        u_res = self.verify_usefulness(llm_report)
        has_halluc = len(t_res["unmatched_figures"]) > 0
        penalty = 0.5 if has_halluc else 1.0
        conf = round((0.6 * t_res["traceability_score"] + 0.4 * u_res["usefulness_score"]) * penalty, 2)

        return {
            "confidence_score": conf, "ai_hallucinated": has_halluc,
            "verdict": "✅ ĐẠT TÍN NHIỆM (Dữ liệu chuẩn)" if conf >= 85 and not has_halluc else "⚠️ CẢNH BÁO BỊA ĐẶT SỐ LIỆU",
            "t_score": t_res["traceability_score"], "u_score": u_res["usefulness_score"],
            "unmatched_figures": t_res["unmatched_figures"]
        }

# ==============================================================================
# PHẦN 2: THUẬT TOÁN DETECTION (ML & RULE BASED)
# ==============================================================================
def create_behavior_features(df):
    df = df.sort_values("time").reset_index(drop=True)
    df["customer_tx_24h"] = 0
    df["customer_cp_24h"] = 0
    df["beneficiary_customers_24h"] = 0

    for i, row in df.iterrows():
        start = row["time"] - pd.Timedelta(hours=24)
        end = row["time"]
        customer_window = df[(df["customer"] == row["customer"]) & (df["time"] >= start) & (df["time"] <= end)]
        df.loc[i, "customer_tx_24h"] = len(customer_window)
        df.loc[i, "customer_cp_24h"] = customer_window["counterparty"].nunique()

        beneficiary_window = df[(df["counterparty"] == row["counterparty"]) & (df["time"] >= start) & (df["time"] <= end)]
        df.loc[i, "beneficiary_customers_24h"] = beneficiary_window["customer"].nunique()
    return df

def detect_fanout(df):
    flags = pd.Series(0, index=df.index)
    for _, group in df.groupby("customer"):
        group = group.sort_values("time")
        for _, row in group.iterrows():
            end = row["time"] + pd.Timedelta(hours=24)
            window = group[(group["time"] >= row["time"]) & (group["time"] <= end)]
            if window["counterparty"].nunique() >= 4:
                flags.loc[window.index] = 1
    return flags

def detect_shared_beneficiary(df):
    flags = pd.Series(0, index=df.index)
    for _, group in df.groupby("counterparty"):
        group = group.sort_values("time")
        for _, row in group.iterrows():
            end = row["time"] + pd.Timedelta(days=5)
            window = group[(group["time"] >= row["time"]) & (group["time"] <= end)]
            if window["customer"].nunique() >= 3:
                flags.loc[window.index] = 1
    return flags

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    return output.getvalue()

# ==============================================================================
# HEADER GIAO DIỆN WEB
# ==============================================================================
st.markdown("""
<div style="text-align: center; max-width: 1000px; margin: 0 auto; padding: 20px;">
    <h1 style="color: #1e3a8a; font-size: 2.5rem; margin-bottom: 10px;">🛡️ HỆ THỐNG ĐIỀU TRA CHỐNG RỬA TIỀN (AML COPILOT)</h1>
    <p style="font-size: 1.1rem; color: #475569;">Tích hợp Machine Learning, Graph Analysis và GenAI trong kiểm soát rủi ro tài chính</p>
</div>
""", unsafe_allow_html=True)

# TẠO 4 TABS CHÍNH
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 1. Quét & Lọc Dữ Liệu",
    "🕸️ 2. Phân Tích Mạng Lưới (Graph)",
    "🤖 3. Trợ Lý AI & QA Audit",
    "🔍 4. Đối Chiếu Lịch Sử (Vector Search)"
])

# ==============================================================================
# TAB 1: DATA PIPELINE
# ==============================================================================
with tab1:
    st.subheader("Phân tích Dị biệt Dữ liệu (Tự động tải Database)")
    col1_1, col1_2 = st.columns([1, 2])

    with col1_1:
        # BỎ ĐI CÁI HỘP FILE UPLOADER
        st.info("📂 Đã kết nối với CSDL Giao dịch (Dataset_full.xlsx)")
        btn_detect = st.button("🚀 Kích Hoạt Radar", type="primary", use_container_width=True)

    if btn_detect:
        with st.spinner("Đang chạy luồng phân tích (Rule Engine & Isolation Forest)..."):
            try:
                # ĐỌC TRỰC TIẾP FILE TỪ GITHUB
                df = pd.read_excel("Dataset_full.xlsx")

                # Xử lý dữ liệu khuyết
                df['customer'] = df['customer'].fillna("KHÔNG_XÁC_ĐỊNH")
                df['counterparty'] = df['counterparty'].fillna("KHÔNG_XÁC_ĐỊNH")
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
                df["time"] = pd.to_datetime(df["time"], errors='coerce').fillna(pd.Timestamp.now())

                # Rules & ML
                df["Rule_Structuring"] = (df["amount"].between(9000, 9999)).astype(int)
                df["Rule_FanOut"] = detect_fanout(df)
                df["Rule_SharedBen"] = detect_shared_beneficiary(df)

                df = create_behavior_features(df)
                df["log_amount"] = np.log1p(df["amount"])
                X = StandardScaler().fit_transform(df[["log_amount", "customer_tx_24h", "customer_cp_24h", "beneficiary_customers_24h"]])
                model = IsolationForest(contamination=0.15, random_state=42)
                df["AI_Anomaly"] = (model.fit_predict(X) == -1).astype(int)

                df["Risk_Score"] = (df["Rule_Structuring"] * 30 + df["Rule_FanOut"] * 30 + df["Rule_SharedBen"] * 30 + df["AI_Anomaly"] * 10)

                result = df[df["Risk_Score"] > 0].copy().sort_values(["Risk_Score", "time"], ascending=[False, True])
                cols = [c for c in ["customer", "counterparty", "time", "amount", "Risk_Score", "pattern_label", "Rule_Structuring", "Rule_FanOut", "Rule_SharedBen", "AI_Anomaly"] if c in result.columns]

                # Lưu vào Session State
                st.session_state['processed_df'] = result[cols].copy()
                cust_summary = result.groupby('customer')['Risk_Score'].max().reset_index()
                st.session_state['customer_list'] = [f"{row['customer']} (Risk: {row['Risk_Score']})" for _, row in cust_summary.sort_values(by='Risk_Score', ascending=False).iterrows()]

                st.success("✅ Đã xử lý thành công!")
            except Exception as e:
                st.error(f"Lỗi hệ thống: {str(e)}")

    if not st.session_state['processed_df'].empty:
        with col1_2:
            st.dataframe(st.session_state['processed_df'].head(100), use_container_width=True)

        with col1_1:
            false_positive = (st.session_state['processed_df']["pattern_label"].str.lower().eq("normal").sum()) if "pattern_label" in st.session_state['processed_df'].columns else 0
            st.info(f"Tổng số cảnh báo: {len(st.session_state['processed_df'])}\n\nFalse Positive: {false_positive}")
            excel_data = convert_df_to_excel(st.session_state['processed_df'])
            st.download_button(label="📤 Tải xuống kết quả (.xlsx)", data=excel_data, file_name="Ket_Qua_AML_Nang_Cap.xlsx", mime="application/vnd.ms-excel", use_container_width=True)

# ==============================================================================
# TAB 2: NETWORK GRAPH
# ==============================================================================
with tab2:
    st.subheader("Trực Quan Hóa Mạng Lưới")
    if st.session_state['customer_list']:
        cust_graph_input = st.selectbox("👤 Chọn Khách hàng để vẽ sơ đồ:", st.session_state['customer_list'], key="sel_graph")
        if st.button("📊 Vẽ Sơ Đồ", type="primary"):
            with st.spinner("Đang render sơ đồ NetworkX..."):
                actual_id = str(cust_graph_input).split(" (")[0].strip()
                df_store = st.session_state['processed_df']
                cust_data = df_store[df_store['customer'] == actual_id]

                if not cust_data.empty:
                    pattern = cust_data['pattern_label'].iloc[0] if 'pattern_label' in cust_data.columns else "Anomaly"
                    targets = cust_data['counterparty'].unique()
                    subset = df_store[(df_store['customer'] == actual_id) | (df_store['counterparty'].isin(targets))]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    G = nx.DiGraph()
                    agg_edges = subset.groupby(['customer', 'counterparty']).agg(tx_count=('amount', 'count'), total_amount=('amount', 'sum')).reset_index()
                    for _, row in agg_edges.iterrows():
                        G.add_edge(row['customer'], row['counterparty'], tx_count=row['tx_count'], total_amount=row['total_amount'])
                    for node in G.nodes():
                        G.nodes[node]['layer'] = 1 if node in targets else 0

                    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")
                    color_map = ['#ff4d4d' if node == actual_id else '#99ccff' if node in targets else '#ffb366' for node in G.nodes()]

                    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=3500, font_size=10, font_weight='bold', arrows=True, arrowsize=20, ax=ax)
                    edge_labels = {(u, v): f"{d['tx_count']} txns\n${d['total_amount']:,.0f}" for u, v, d in G.edges(data=True)}
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', ax=ax)
                    ax.set_title(f"Hồ sơ: {actual_id} | Mẫu hành vi: {pattern}", fontsize=15, fontweight='bold', color='#b30000', pad=20)
                    fig.tight_layout()

                    st.pyplot(fig)
    else:
        st.warning("Vui lòng tải và quét dữ liệu ở Tab 1 trước.")

# ==============================================================================
# TAB 3: LLM COPILOT
# ==============================================================================
AML_PROMPT_TEMPLATE = """
Hồ sơ điều tra (Case):
- Khách hàng: {customer_id}
- Tổng số lượng giao dịch: {total_tx}
- Tổng giá trị: {total_amount:,.2f} USD
- Điểm rủi ro: {risk_score}

BẰNG CHỨNG GIAO DỊCH:
{transactions_text}

Hãy sinh BÁO CÁO ĐIỀU TRA gồm 5 phần:
1. TÓM TẮT HỒ SƠ
2. NHẬN DIỆN BẤT THƯỜNG
3. TRÍCH XUẤT BẰNG CHỨNG
4. GIẢ THUYẾT RỦI RO (Không kết luận rửa tiền)
5. 3 CÂU HỎI PHỎNG VẤN KHÁCH HÀNG
"""

with tab3:
    col3_1, col3_2 = st.columns([1, 2])
    with col3_1:
        st.subheader("Cấu Hình Trợ Lý AI")
        cust_llm_input = st.selectbox("👤 Chọn Khách hàng:", st.session_state['customer_list'] if st.session_state['customer_list'] else ["Chưa có dữ liệu"], key="sel_llm")
        api_key_input = st.text_input("🔑 Gemini API Key", type="password")
        btn_gen_llm = st.button("📝 Sinh Báo Cáo & Đánh Giá QA", type="primary", use_container_width=True)

    if btn_gen_llm:
        if not api_key_input:
            st.error("Lỗi: Thiếu API Key.")
        elif not st.session_state['customer_list']:
            st.error("Lỗi: Chưa chạy dữ liệu ở Tab 1.")
        else:
            with st.spinner("AI đang tạo báo cáo và hệ thống QA đang chấm điểm..."):
                actual_id = str(cust_llm_input).split(" (")[0].strip()
                df_store = st.session_state['processed_df']
                df_case = df_store[df_store['customer'] == actual_id]

                total_tx, total_amount, risk_score = len(df_case), float(df_case['amount'].sum()), int(df_case['Risk_Score'].max())
                tx_lines = [f" - {row['time']}: Chuyển {row['amount']:,.2f} USD cho {row['counterparty']}" for _, row in df_case.iterrows()]
                user_prompt = AML_PROMPT_TEMPLATE.format(customer_id=actual_id, total_tx=total_tx, total_amount=total_amount, risk_score=risk_score, transactions_text="\n".join(tx_lines))

                try:
                    client = genai.Client(api_key=api_key_input)
                    response = client.models.generate_content(
                        model='gemini-3.5-flash-lite',
                        contents=user_prompt,
                        config=types.GenerateContentConfig(system_instruction="Không kết luận rửa tiền. Trích xuất đúng số. Phải có từ khóa Structuring, Fan-out, Shared Beneficiary", temperature=0.2)
                    )
                    llm_report = response.text
                    st.session_state['llm_report'] = llm_report

                    # Chấm QA
                    evaluator = AMLReportEvaluator(df_case)
                    eval_res = evaluator.evaluate(llm_report)
                    qa_html = f"""
                    ### 🛡️ KẾT QUẢ KIỂM ĐỊNH AI (QA AUDIT)
                    - **Kết luận:** {eval_res['verdict']}
                    - **Điểm tin cậy tổng hợp:** {eval_res['confidence_score']}/100
                    - **Traceability (Khớp số liệu Excel):** {eval_res['t_score']}/100
                    - **Usefulness (Hữu ích nghiệp vụ):** {eval_res['u_score']}/100
                    """
                    if eval_res['ai_hallucinated']:
                        qa_html += f"\n- **CẢNH BÁO BỊA ĐẶT (Hallucination):** Phát hiện các số tiền không khớp Excel gốc: {eval_res['unmatched_figures']}"

                    st.session_state['qa_result'] = qa_html
                except Exception as e:
                    st.error(f"Lỗi API: {e}")

    with col3_2:
        st.subheader("Bản Đánh Giá & Báo Cáo")
        if st.session_state['qa_result']:
            st.info(st.session_state['qa_result'])
            st.markdown("---")
            st.markdown(st.session_state['llm_report'])
        else:
            st.markdown("*(Nội dung báo cáo và chứng chỉ QA sẽ hiện tại đây)*")

# ==============================================================================
# TAB 4: VECTOR SEARCH
# ==============================================================================
with tab4:
    col4_1, col4_2 = st.columns([1, 2])
    with col4_1:
        st.subheader("Kho Dữ Liệu Án Lệ")
        # BỎ ĐI CÁI HỘP FILE UPLOADER
        st.info("📂 Đã kết nối với CSDL Lịch sử (Historical_Data.xlsx)")

        case_input_for_search = st.text_area("📝 Báo cáo vụ án hiện tại (Tự động cập nhật từ Tab 3):",
                                             value=st.session_state.get('llm_report', ''), height=200)
        btn_search = st.button("🔍 Tìm Án Lịch Sử", type="primary", use_container_width=True)

    with col4_2:
        st.subheader("Top 3 Case Tương Đồng")
        if btn_search:
            if not case_input_for_search.strip():
                st.error("Báo cáo vụ án hiện tại đang trống!")
            else:
                with st.spinner("Đang mã hóa Vector và tính độ tương đồng (Cosine Similarity)..."):
                    try:
                        # ĐỌC TRỰC TIẾP FILE TỪ GITHUB
                        df_historical = pd.read_excel("Historical_Data.xlsx")

                        if 'case_summary' not in df_historical.columns:
                            st.error("File Excel của bạn đang thiếu cột 'case_summary'.")
                        else:
                            @st.cache_resource
                            def load_vector_model():
                                return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

                            embedding_model = load_vector_model()
                            historical_texts = df_historical['case_summary'].astype(str).tolist()
                            historical_embeddings = embedding_model.encode(historical_texts)
                            current_embedding = embedding_model.encode([case_input_for_search])

                            similarities = cosine_similarity(current_embedding, historical_embeddings)[0]
                            df_historical['similarity_score'] = similarities
                            top_cases = df_historical.sort_values(by='similarity_score', ascending=False).head(3)

                            cols_to_show = ['historical_case_id', 'pattern_type', 'case_summary', 'investigation_result', 'similarity_score']
                            cols_available = [c for c in cols_to_show if c in top_cases.columns]
                            result_df = top_cases[cols_available].copy()
                            result_df['similarity_score'] = result_df['similarity_score'].apply(lambda x: f"{x:.2%}")

                            st.dataframe(result_df, use_container_width=True)

                            excel_history = convert_df_to_excel(result_df)
                            st.download_button("📤 Tải xuống Top 3 (.xlsx)", data=excel_history, file_name="Top3_Similar_Cases.xlsx", mime="application/vnd.ms-excel")
                    except Exception as e:
                        st.error(f"Lỗi Vector Search: {str(e)}")