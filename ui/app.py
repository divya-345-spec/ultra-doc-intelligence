import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Ultra Doc Intelligence",
    page_icon="📄",
    layout="wide",
)

# ── Custom CSS ──
st.markdown("""
<style>
    .main > div { max-width: 1100px; margin: auto; }
    .stMetricValue { font-size: 1.8rem !important; }
    div[data-testid="stExpander"] { border: 1px solid #e0e0e0; border-radius: 8px; }
    .confidence-high { color: #28a745; font-weight: 700; }
    .confidence-mid  { color: #ffc107; font-weight: 700; }
    .confidence-low  { color: #dc3545; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("# 📄 Ultra Doc Intelligence")
st.caption("AI-powered logistics document analysis — upload, ask, extract.")

# ── Sidebar ──
with st.sidebar:
    st.markdown("### How to use")
    st.markdown(
        "1. **Upload** a logistics document (PDF, DOCX, TXT)\n"
        "2. **Ask** natural language questions\n"
        "3. **Extract** structured shipment data as JSON"
    )
    st.divider()
    st.markdown("**Supported fields:**")
    st.code(
        "shipment_id, shipper, consignee,\n"
        "pickup_datetime, delivery_datetime,\n"
        "equipment_type, mode, rate,\n"
        "currency, weight, carrier_name",
        language=None,
    )
    st.divider()
    st.caption("Powered by RAG + LLM with hallucination guardrails")

# ── Tabs ──
tab_upload, tab_ask, tab_extract = st.tabs(["Upload", "Ask Questions", "Structured Extraction"])

# ═══════════════════════════════════════════
# TAB 1 — UPLOAD
# ═══════════════════════════════════════════
with tab_upload:
    st.markdown("### Upload Logistics Documents")

    uploaded_files = st.file_uploader(
        "Drag and drop or browse files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        progress = st.progress(0, text="Uploading...")
        total = len(uploaded_files)

        for idx, uploaded_file in enumerate(uploaded_files):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/octet-stream",
                )
            }
            try:
                resp = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                if resp.status_code == 200:
                    st.success(f"Uploaded: **{uploaded_file.name}**")
                    data = resp.json()
                    fields = data.get("structured_fields", {})
                    non_null = sum(1 for v in fields.values() if v is not None)
                    st.info(f"Extracted **{non_null}/{len(fields)}** structured fields on upload.")
                else:
                    st.error(f"Upload failed for {uploaded_file.name}")
            except Exception as e:
                st.error(f"Connection error: {e}")

            progress.progress((idx + 1) / total, text=f"Processed {idx + 1}/{total}")

        progress.empty()

# ═══════════════════════════════════════════
# TAB 2 — ASK QUESTIONS
# ═══════════════════════════════════════════
with tab_ask:
    st.markdown("### Ask a Question")

    question = st.text_input(
        "Type your question about the uploaded document",
        placeholder="e.g. What is the carrier rate? Who is the consignee?",
        label_visibility="collapsed",
    )

    ask_btn = st.button("Ask", type="primary", use_container_width=True)

    if ask_btn and question.strip():
        with st.spinner("Searching document..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data.get("answer", "No answer")
                    confidence = data.get("confidence", 0)
                    sources = data.get("sources", [])

                    # Answer card
                    st.markdown("---")
                    st.markdown("#### Answer")
                    st.markdown(f"> {answer}")

                    # Confidence metric
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if confidence >= 0.7:
                            color = "confidence-high"
                        elif confidence >= 0.4:
                            color = "confidence-mid"
                        else:
                            color = "confidence-low"
                        st.markdown(f"**Confidence:** <span class='{color}'>{confidence:.0%}</span>", unsafe_allow_html=True)
                    with col2:
                        st.progress(min(confidence, 1.0))

                    # Sources
                    if sources:
                        with st.expander(f"Sources ({len(sources)} chunks)", expanded=False):
                            for i, src in enumerate(sources, 1):
                                st.markdown(f"**[Page {src.get('page', '?')}]** {src.get('text', '')}")
                                if i < len(sources):
                                    st.divider()
                else:
                    st.error("Failed to get answer from API.")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")

# ═══════════════════════════════════════════
# TAB 3 — STRUCTURED EXTRACTION
# ═══════════════════════════════════════════
with tab_extract:
    st.markdown("### Structured Shipment Data")
    st.caption("Extract all 11 logistics fields from the uploaded document as JSON.")

    extract_btn = st.button("Extract Structured Data", type="primary", use_container_width=True)

    if extract_btn:
        with st.spinner("Extracting fields..."):
            try:
                resp = requests.post(f"{API_URL}/extract", timeout=120)

                if resp.status_code == 200:
                    data = resp.json()

                    non_null = {k: v for k, v in data.items() if v is not None}
                    null_keys = [k for k, v in data.items() if v is None]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Fields", len(data))
                    col2.metric("Extracted", len(non_null))
                    col3.metric("Missing", len(null_keys))

                    st.markdown("---")

                    if non_null:
                        st.markdown("#### Extracted Values")
                        df = pd.DataFrame(
                            [{"Field": k, "Value": v} for k, v in non_null.items()]
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)

                    if null_keys:
                        st.markdown("#### Missing Fields")
                        st.warning(", ".join(null_keys))

                    with st.expander("Raw JSON", expanded=False):
                        st.json(data)
                else:
                    st.error("Extraction failed.")

            except requests.ConnectionError:
                st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")
