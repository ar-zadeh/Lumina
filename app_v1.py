import io
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import gradio as gr
import pdfplumber
import pandas as pd


# ----------------------------
# Utilities
# ----------------------------
def read_pdf_text(file_bytes: bytes) -> str:
    """Extract plain text from all pages of a PDF using pdfplumber."""
    texts = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for p in pdf.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
    # Keep page breaks to help regex that expects line starts
    return "\n\n".join(texts)


def normalize_space(s: str) -> str:
    return re.sub(r"[ \t]+", " ", s).replace("\u00a0", " ").strip()


def find_label_value_block(text: str, label: str, window: int = 150) -> Optional[str]:
    """
    Heuristic: find 'label' (case-insensitive), return the nearby span after it.
    Useful for patterns like 'Invoice Number: 12345' or 'Patient ID  = ABC-1'.
    """
    pattern = re.compile(rf"(?i)\b{re.escape(label)}\b\s*[:=]?\s*(.+)", re.IGNORECASE)
    m = pattern.search(text)
    if m:
        # Crop a short window after the label line
        val = m.group(1)[:window]
        # Stop at common hard breaks
        val = re.split(r"[\r\n]", val)[0]
        return normalize_space(val)
    return None


def type_postprocess(val: str, ftype: str) -> str:
    if val is None:
        return ""
    val = val.strip()
    if not val:
        return val

    ftype = (ftype or "").lower()

    if ftype in {"int", "integer"}:
        m = re.search(r"[-+]?\d+", val.replace(",", ""))
        return m.group(0) if m else ""
    if ftype in {"float", "number", "decimal"}:
        m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", val.replace(",", ""))
        return m.group(0) if m else ""
    if ftype in {"date", "datetime"}:
        # naive date pick (YYYY-MM-DD, MM/DD/YYYY, DD Mon YYYY etc.)
        m = re.search(
            r"(\d{4}-\d{1,2}-\d{1,2})|(\d{1,2}/\d{1,2}/\d{2,4})|(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})",
            val,
        )
        return m.group(0) if m else val
    return val


# ----------------------------
# Agent: field extraction
# ----------------------------
def extract_field_from_text(
    text: str,
    field: str,
    description: str,
    ftype: str = "",
    regex: str = ""
) -> str:
    """
    Priority:
      1) If regex is provided, use it (first capturing group if present, else whole match).
      2) Try exact field label, then description keywords.
      3) Fallback: keyword window search by important tokens from description.
    Finally run type-based postprocess.
    """
    # 1) Regex path
    if regex and regex.strip():
        try:
            m = re.search(regex, text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if m:
                val = m.group(1) if m.groups() else m.group(0)
                return type_postprocess(normalize_space(val), ftype)
        except re.error:
            # invalid regex -> ignore and continue
            pass

    # 2) Label-based
    for label in [field, field.replace("_", " "), field.replace("-", " ")]:
        val = find_label_value_block(text, label)
        if val:
            return type_postprocess(val, ftype)

    # 3) Description keyword hunt
    # Pick a few strong tokens from description
    desc_tokens = re.findall(r"[A-Za-z]{4,}", description or "")
    # Try top 3 tokens as pseudo-labels
    for tok in desc_tokens[:3]:
        val = find_label_value_block(text, tok)
        if val:
            return type_postprocess(val, ftype)

    # 4) Soft window search: nearest token, then take next word-ish chunk
    for tok in [field] + desc_tokens[:3]:
        i = text.lower().find(tok.lower())
        if i != -1:
            window = text[i:i+160]
            # remove the token itself
            window = re.sub(re.escape(tok), "", window, flags=re.IGNORECASE)
            # pick first plausible token sequence (letters/digits/.-/_)
            m = re.search(r"([A-Za-z0-9][A-Za-z0-9._\-\/ ]{1,60})", window)
            if m:
                return type_postprocess(normalize_space(m.group(1)), ftype)

    return ""


# ----------------------------
# Core run: per-file extraction
# ----------------------------
def run_extraction_on_file(file_obj, schema_rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Returns one dict representing a row for the result table.
    """
    # Load bytes + filename
    if isinstance(file_obj, dict) and "data" in file_obj:
        file_bytes = file_obj["data"]
        fname = file_obj.get("name", "uploaded.pdf")
    else:
        path = file_obj if isinstance(file_obj, str) else file_obj.name
        with open(path, "rb") as fh:
            file_bytes = fh.read()
        fname = path.split("/")[-1]

    text = read_pdf_text(file_bytes)

    row = {"source_file": fname}
    for r in schema_rows:
        field = (r.get("Field") or "").strip()
        if not field:
            continue
        desc  = (r.get("Description") or "").strip()
        ftype = (r.get("Type") or "").strip()
        rgx   = (r.get("Regex (optional)") or "").strip()

        val = extract_field_from_text(text, field, desc, ftype, rgx)
        row[field] = val

    return row


# ----------------------------
# Gradio callbacks
# ----------------------------
def on_extract(files, schema_df: pd.DataFrame) -> Tuple[str, pd.DataFrame, str]:
    """
    Returns:
      - status message
      - preview dataframe
      - hidden JSON payload of full dataframe
    """
    if not files:
        return ("Please upload at least one PDF.", pd.DataFrame(), "")

    # Clean schema
    if schema_df is None or schema_df.empty:
        return ("Define at least one column in the schema.", pd.DataFrame(), "")

    # Normalize schema column names and rows
    expected_cols = ["Field", "Description", "Type", "Regex (optional)"]
    # Upgrade user’s columns if they changed headers accidentally
    rename_map = {c: c.strip() for c in schema_df.columns}
    schema_df = schema_df.rename(columns=rename_map)
    for col in expected_cols:
        if col not in schema_df.columns:
            schema_df[col] = ""
    schema_df = schema_df[expected_cols]

    # Drop empty field names
    schema_df = schema_df[ schema_df["Field"].astype(str).str.strip() != "" ]
    if schema_df.empty:
        return ("Your schema has no valid 'Field' names.", pd.DataFrame(), "")

    schema_rows = schema_df.fillna("").to_dict(orient="records")

    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(run_extraction_on_file, f, schema_rows) for f in files]
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({"source_file": "<error>", "error": str(e)})

    if not results:
        return ("No output rows produced.", pd.DataFrame(), "")

    # Order columns: meta first, then schema order
    cols = ["source_file"] + [r["Field"] for r in schema_rows]
    df = pd.DataFrame(results)
    # Ensure columns exist even if some fields were missing
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols]

    # Preview + stash
    json_payload = df.to_json(orient="split")
    status = f"Extracted {len(df)} row(s) • {df['source_file'].nunique()} file(s)."
    return (status, df.head(200), json_payload)


def on_download(json_payload: str) -> Tuple[Any, str]:
    if not json_payload:
        return (gr.File.update(value=None), "Nothing to download. Run extraction first.")
    df = pd.read_json(io.StringIO(json_payload), orient="split")
    path = "extracted_from_pdfs.csv"
    df.to_csv(path, index=False)
    return (path, f"Saved {len(df)} rows to extracted_from_pdfs.csv")


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(title="PDF → Table (Schema-Driven)") as demo:
    gr.Markdown("# 🧠 PDF ➜ Table (Schema-Driven Extraction)")
    gr.Markdown(
        "1) Upload PDFs\n"
        "2) Define your **schema** (Field + Description + Type + optional Regex)\n"
        "3) Click **Extract** to build a row per PDF.\n\n"
        "Tip: Provide a Regex when you can — it's the most reliable. Otherwise the agent will try label/keyword heuristics."
    )

    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(
                label="Drop PDFs",
                file_count="multiple",
                file_types=[".pdf"],
                type="filepath"
            )

            gr.Markdown("### Schema")
            schema = gr.Dataframe(
                headers=["Field", "Description", "Type", "Regex (optional)"],
                value=[
                    ["invoice_number", "Exact invoice # (e.g., near 'Invoice Number')", "int", r"invoice\s*number\s*[:#]\s*([A-Za-z0-9\-]+)"],
                    ["invoice_date", "Date the invoice was issued", "date", ""],
                    ["total_amount", "Grand total / amount due", "float", r"(?:total|amount due)\s*[:$]?\s*\$?\s*([-+]?\d[\d,]*(?:\.\d+)?)"],
                ],
                row_count=(3, "dynamic"),
                col_count=(4, "fixed"),
                wrap=True,
                interactive=True,
                # height=240
            )

            extract_btn = gr.Button("🚀 Extract", variant="primary")
            status = gr.Markdown()

            hidden_json = gr.State("")

            download_btn = gr.Button("⬇️ Download CSV")
            download_file = gr.File(label="Your CSV", interactive=False)
            dl_status = gr.Markdown()

        with gr.Column(scale=2):
            table = gr.Dataframe(
                label="Preview (top 200 rows)",
                value=pd.DataFrame(),
                wrap=True,
                # height=560
            )

    extract_btn.click(
        fn=on_extract,
        inputs=[files, schema],
        outputs=[status, table, hidden_json],
        api_name="extract"
    )

    download_btn.click(
        fn=on_download,
        inputs=[hidden_json],
        outputs=[download_file, dl_status],
        api_name="download_csv"
    )

if __name__ == "__main__":
    demo.launch()