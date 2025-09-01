import os, io, re, time, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import streamlit as st
from urllib.parse import quote
from openai import OpenAI

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="BSE M&A/JV — OpenAI PDF Summaries", layout="wide")
st.title("📈 BSE Company Update — M&A / Merger / Scheme / JV")
st.caption("Filter BSE announcements → fetch attachments → upload PDFs to OpenAI → get clean bullet summaries (no local PDF parsing).")

# ===============================
# Small utilities
# ===============================
def _norm(s): return re.sub(r"\s+", " ", str(s or "")).strip()
def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns: return n
    return None

def _get_openai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    if not api_key:
        st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
        st.stop()
    return OpenAI(api_key=api_key)

# ===============================
# BSE fetch + filters (unchanged logic)
# ===============================
def filter_announcements(df_in: pd.DataFrame, category_filter="Company Update") -> pd.DataFrame:
    if df_in.empty: return df_in.copy()
    cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
    if not cat_col: return df_in.copy()
    df2 = df_in.copy()
    df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
    out = df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])
    return out

def fetch_bse_announcements_strict(start_yyyymmdd: str, end_yyyymmdd: str, request_timeout: int = 25) -> pd.DataFrame:
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd

    base_page = "https://www.bseindia.com/corporates/ann.html"
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })
    try: s.get(base_page, timeout=15)
    except Exception: pass

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for v in variants:
        params = {
            "pageno": 1, "strCat": "-1",
            "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strscrip": "", "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            if "application/json" not in (r.headers.get("content-type","")):
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try: total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception: total = None
            if not table: break
            params["pageno"] += 1
            page += 1
            time.sleep(0.25)
            if total and len(rows) >= total: break
        if rows:
            all_rows = rows
            break

    if not all_rows: return pd.DataFrame()

    # Normalize frame
    all_keys = set()
    for r in all_rows: all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # Exact filters
    df = filter_announcements(df, category_filter="Company Update")
    df = df.loc[
        df.filter(
            ["NEWSSUB","SUBCATEGORY","SUBCATEGORYNAME","NEWS_SUBCATEGORY","NEWS_SUB"], axis=1
        ).astype(str).apply(
            lambda col: col.str.contains(r"(Acquisition|Amalgamation\s*/\s*Merger|Scheme of Arrangement|Joint Venture)",
                                         case=False, na=False)
        ).any(axis=1)
    ]
    return df

# ===============================
# Attachment URL candidates
# ===============================
def _candidate_urls(row):
    seen, out = set(), []
    bases = [
        "https://www.bseindia.com/xml-data/corpfiling/AttachHis/",
        "https://www.bseindia.com/xml-data/corpfiling/Attach/",
        "https://www.bseindia.com/xml-data/corpfiling/AttachLive/",
        "https://www.bseindia.com/xml-data/corpfiling/AttachStar/",
        "https://www.bseindia.com/xml-data/corpfiling/AttchPDF/",
    ]
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        att_q = "/".join(quote(p, safe="._-") for p in att.split("/"))
        for base in bases:
            u = base + att_q
            if u not in seen: out.append(u); seen.add(u)

    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        u = ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/")
        if u not in seen: out.append(u); seen.add(u)
    return out

# ===============================
# OpenAI: upload + summarize PDF (no local parsing)
# ===============================
SUMMARY_SYS = (
    "You are a capital-markets analyst. Read the attached PDF and summarize it "
    "into precise Markdown bullets. Preserve exact figures (₹, %, counts), names, and concrete dates. "
    "Do not invent missing details."
)

def summarize_pdf_with_openai(client: OpenAI, model: str, pdf_bytes: bytes, filename: str, bullets: int, temperature: float) -> str:
    """
    Upload the PDF, then ask the Responses API to summarize it.
    We rely entirely on OpenAI to read/extract text from the PDF.
    """
    # Save to a temp file so the SDK can upload a file handle cleanly
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
        tf.write(pdf_bytes)
        tf.flush()
        temp_path = tf.name

    try:
        with open(temp_path, "rb") as fh:
            up = client.files.create(file=fh, purpose="assistants")  # file will be referenced in the Response
        file_id = up.id

        prompt = (
            f"Summarize this PDF in <= {bullets} bullets.\n"
            f"Focus: deal type (acquisition/merger/JV/scheme), counterparties, stake %, consideration/valuation, "
            f"structure (share swap/slump sale), key dates (appointed/effective/record/closing), approvals/conditions, "
            f"and immediate implications. Output only Markdown bullets starting with '- '."
        )

        # Use Responses API with file input
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_file", "file_id": file_id},
                ],
            }],
            instructions=SUMMARY_SYS,
            temperature=temperature,
        )

        # Robust extraction across SDK variants
        try:
            out = resp.output_text.strip()
        except Exception:
            out = resp.output[0].content[0].text.strip()
        return out or "- (empty summary)"
    except Exception as e:
        return f"- [OpenAI error] {e}"
    finally:
        try: os.remove(temp_path)
        except Exception: pass

# ===============================
# Sidebar controls
# ===============================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    st.divider()
    st.subheader("OpenAI")
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"],
        index=0,
        help="Uses the Responses API with direct PDF input."
    )
    bullets = st.slider("Bullets per summary", 4, 10, 7)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.0, 0.1)
    llm_parallel = st.slider("Parallel OpenAI requests", 1, 4, 2, help="Concurrent summaries (watch your rate limits).")
    max_pdfs = st.number_input("Max PDFs to summarize (cap)", min_value=1, max_value=50, value=20, step=1)

    run = st.button("🚀 Fetch & Summarize", type="primary")

def _get_openai_client():
    api_key = (
        st.session_state.get("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    )
    if not api_key:
        st.error("Missing OPENAI_API_KEY (set env var, add to Secrets, or enter it in the sidebar).")
        st.stop()
    return OpenAI(api_key=api_key)



# ===============================
# Run pipeline
# ===============================
def _fmt(d: datetime.date) -> str: return d.strftime("%Y%m%d")

if run:
    client = _get_openai_client()
    start_str, end_str = _fmt(start_date), _fmt(end_date)

    with st.status("Fetching announcements…", expanded=True):
        df_hits = fetch_bse_announcements_strict(start_str, end_str)
        st.write(f"Matched rows after filters: **{len(df_hits)}**")

    if df_hits.empty:
        st.warning("No matching announcements in this window.")
        st.stop()

    # Build attachment list and download bytes (no local parsing of PDF)
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
    })

    rows = []
    for _, r in df_hits.iterrows():
        urls = _candidate_urls(r)
        rows.append({**r.to_dict(), "_cand_urls": urls})

    st.info(f"Trying to fetch PDFs for {len(rows)} rows…")

    def fetch_pdf(i, cand_urls):
        for u in cand_urls:
            try:
                resp = s.get(u, timeout=25)
                if resp.status_code == 200:
                    ctype = (resp.headers.get("content-type","") or "").lower()
                    if "pdf" in ctype or u.lower().endswith(".pdf"):
                        return i, u, resp.content
            except Exception:
                continue
        return i, "", b""

    pdf_blobs = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_pdf, i, r["_cand_urls"]) for i, r in enumerate(rows)]
        for fut in as_completed(futs):
            i, url, blob = fut.result()
            rows[i]["pdf_url"] = url
            pdf_blobs[i] = blob

    with st.status("Uploading to OpenAI & summarizing…", expanded=True):
        summaries = [""] * len(rows)

        def llm_worker(i):
            r = rows[i]
            blob = pdf_blobs[i]
            if not blob or not r.get("pdf_url"):
                return i, "- PDF not accessible from attachment links."
            fname = f"{_norm(r.get('SLONGNAME') or 'file')[:40]}_{i}.pdf"
            summ = summarize_pdf_with_openai(
                client=client, model=model, pdf_bytes=blob, filename=fname,
                bullets=bullets, temperature=temperature
            )
            return i, summ

        # respect the cap
        indices = [i for i in range(len(rows))][:int(max_pdfs)]
        with ThreadPoolExecutor(max_workers=int(llm_parallel)) as ex:
            futs = [ex.submit(llm_worker, i) for i in indices]
            for fut in as_completed(futs):
                i, out = fut.result()
                summaries[i] = out

        df = pd.DataFrame(rows)
        df["openai_summary"] = summaries

    # ========== UI Rendering ==========
    show_cols = [c for c in [
        "NEWS_DT","SLONGNAME","HEADLINE","CATEGORYNAME","SUBCATEGORYNAME",
        "NEWSSUB","pdf_url","openai_summary"
    ] if c in df.columns]

    st.subheader("📑 Results (with OpenAI bullet summaries)")
    st.dataframe(df[show_cols].fillna(""), use_container_width=True, hide_index=True)

    st.subheader("📝 Summaries")
    for i, rr in df.iterrows():
        title = f"{rr.get('NEWS_DT','')} — {rr.get('SLONGNAME','')} — {rr.get('HEADLINE','')}"
        with st.expander(title):
            if rr.get("pdf_url"):
                st.markdown(f"**Attachment:** {rr['pdf_url']}")
            st.markdown(rr.get("openai_summary") or "- (no summary)")

    st.download_button(
        "⬇️ Download CSV",
        data=df[show_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"company_update_mna_jv_{start_str}_{end_str}_openai.csv",
        mime="text/csv"
    )
    st.subheader("OpenAI API key")
    api_key_input = st.text_input("Enter key (kept only for this session)", type="password")
    if api_key_input:
        st.session_state["OPENAI_API_KEY"] = api_key_input.strip()

else:
    st.info("Pick your date range and click **Fetch & Summarize**.")
