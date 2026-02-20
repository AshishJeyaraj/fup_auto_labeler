
import os
import re
import json
import csv
import math
import time
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import pandas as pd
import streamlit as st
import urllib3

# =========================
# SECURITY / NETWORKING
# =========================
# Disable SSL verification as requested (internal self-signed chain).
# NOTE: This reduces TLS security; prefer providing a corporate CA bundle when possible.
VERIFY_SSL: bool = False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DEFAULT_TIMEOUT = int(st.secrets.get("timeout_seconds", 60))

DASHPROACH_BASE_URL = st.secrets.get("dashproach_base_url", "https://dashproach.amadeus.net/api").rstrip("/")
APROACH_BASE_URL    = st.secrets.get("aproach_base_url", "https://aproach-api.muc.amadeus.net/api/v2/json").rstrip("/")

AUTH_USERNAME = st.secrets.get("auth_username", "")
AUTH_PASSWORD = st.secrets.get("auth_password", "")

# LLM: local OpenAI-compatible endpoint (recommended with vLLM)
LLM_ENABLED  = bool(st.secrets.get("use_llm", True))
LLM_BASE_URL = st.secrets.get("llm_base_url", "http://localhost:8000/v1").rstrip("/")
LLM_API_KEY  = st.secrets.get("llm_api_key", "local")  # dummy ok for local servers
LLM_MODEL    = st.secrets.get("llm_model", "gpt-oss-20b")

# If you want a strict "no LLM" mode, set use_llm=false in secrets.
# The app will fall back to heuristic labeling.

# =========================
# TEMPLATES (Question sets)
# =========================
# We store only the question lines (compact) for matching + completeness checks.
# Optional items are marked with "(optional)" and should NOT be required for Q-COMPLETE.
TEMPLATES: Dict[str, List[str]] = {
    "Fallback Collapse": [
        "Why was the fallback performed to recover the issue?",
        "Was the change tested in QCP environment? (optional)",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Capacity Change Collapse": [
        "Why was a capacity change needed to fix the issue?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Configuration Change Collapse": [
        "Why was a configuration change needed to fix the issue?",
        "Was the issue detected in QCP? (optional)",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Reboot Collapse": [
        "Why was a reboot performed to recover the issue?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Application Restart Collapse": [
        "Does Operational Monitoring indicate a problem leading to a re-occurrence? YES/NO",
        "Why was the application(s) restarted to recover the issue?",
        "Is this a re-occurrence?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent re-occurrence?",
    ],
    "Other Collapse": [
        "Why did the recovery action solve the issue?",
        "Was the change tested in QCP environment? (optional)",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Reroute Collapse": [
        "Why was rerouting done to recover the issue?",
        "Is this routing temporary?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Database Restore Collapse": [
        "Why was the database restored to recover the issue?",
        "What is the root cause of the issue?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Database Patch Collapse": [
        "Why was the database patch applied to recover the issue?",
        "What is the root cause of the issue?",
        "Was there any issue with the database application server/nodes?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Software Correction Collapse": [
        "Why was the software correction (load) performed to solve the issue?",
        "What is the root cause of the issue?",
        "Did any recent change trigger the issue?",
        "Is the software correction performed and validated in QCP?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Hardware Collapse": [
        "Why did the recovery action solve the issue?",
        "What is the root cause of the issue?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "System Patch Collapse": [
        "Why did the recovery action solve the issue?",
        "What is the root cause of the issue?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Restart File Transfer Collapse": [
        "Why did the recovery action solve the issue?",
        "Was the delay due to a planned software load? Was the outage communicated?",
        "Was there any automatic alert received via TOPX? provide reference",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Modify/File Log Collapse": [
        "Why did the recovery action solve the issue?",
        "Was the log file generated successfully? size and location",
        "Is there any exception message seen on the log file/s?",
        "How will it be fixed permanently? (provide fix reference)",
        "List countermeasures taken to prevent reoccurrence?",
    ],
    "Unknown/None Collapse": [
        "What caused the error/change in behavior of the system?",
        "Were any recovery actions taken/known?",
        "How will it be fixed permanently? (provide fix reference)",
        "Are there any countermeasures to prevent reoccurrence?",
    ],
}

QUESTIONNAIRE_MARKER = "problem management questionnaire"

# =========================
# DATA MODELS
# =========================
@dataclass
class FupCandidate:
    rnid: str
    title: str
    keywords: str
    logged_date: Optional[str] = None
    closed_date: Optional[str] = None
    severity: Optional[str] = None
    raw: Any = None

@dataclass
class LabelResult:
    rnid: str
    label: str  # Q-COMPLETE / Q-PARTIAL / Q-NONE / Q-MISSING
    template: Optional[str]
    confidence: float
    answered_questions: Optional[int] = None
    required_questions: Optional[int] = None
    missing_questions: Optional[List[int]] = None
    rationale: Optional[str] = None

# =========================
# HELPERS
# =========================
def month_range(start: dt.date, end: dt.date) -> List[Tuple[int, int]]:
    """Return list of (year, month) pairs covering [start, end]."""
    months = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        months.append((y, m))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return months

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    return s.strip()

# =========================
# API CLIENTS
# =========================
def _session(auth: Tuple[str, str]) -> requests.Session:
    sess = requests.Session()
    sess.auth = auth
    sess.headers.update({"Accept": "*/*"})
    return sess

def dashproach_teamactivity(year: int, month: Optional[int], auth: Tuple[str, str]) -> Any:
    """
    Calls:
      https://dashproach.amadeus.net/api/record/DAPPATC/teamactivity?year=YYYY&month=M
      or ...?year=YYYY
    Response can be JSON or CSV-like text depending on server.
    """
    url = f"{DASHPROACH_BASE_URL}/record/DAPPATC/teamactivity"
    params = {"year": year}
    if month is not None:
        params["month"] = month

    r = requests.get(url, params=params, auth=auth, timeout=DEFAULT_TIMEOUT, verify=VERIFY_SSL)
    r.raise_for_status()

    ctype = (r.headers.get("Content-Type") or "").lower()
    if "application/json" in ctype:
        return r.json()

    # Otherwise treat as text (CSV/lines)
    text = r.text.strip()
    return text

def parse_teamactivity_payload(payload: Any) -> List[FupCandidate]:
    """
    Robust parsing:
    - If payload is JSON: try common shapes (list/dict with 'records', etc.)
    - If payload is text: parse CSV-ish lines.
    """
    candidates: List[FupCandidate] = []

    if isinstance(payload, dict):
        # common patterns: {"records":[...]} or {"data":[...]} or {"items":[...]}
        for key in ["records", "data", "items", "result"]:
            if key in payload and isinstance(payload[key], list):
                for item in payload[key]:
                    candidates.extend(parse_teamactivity_payload(item))
                return candidates

        # single record dict
        item = payload
        rnid = str(item.get("rnid") or item.get("id") or "")
        keywords = str(item.get("keywords") or item.get("keyword") or "")
        title = str(item.get("title") or item.get("summary") or "")
        sev = str(item.get("severity") or item.get("severityid") or item.get("sev") or "")
        logged = str(item.get("logged_date") or item.get("loggedDate") or item.get("created") or "")
        closed = str(item.get("closed_date") or item.get("closedDate") or item.get("closed") or "")
        if rnid:
            candidates.append(FupCandidate(rnid=rnid, title=title, keywords=keywords, severity=sev, logged_date=logged, closed_date=closed, raw=item))
        return candidates

    if isinstance(payload, list):
        for item in payload:
            candidates.extend(parse_teamactivity_payload(item))
        return candidates

    if isinstance(payload, str):
        # CSV-like lines. Example given:
        # 29220443,PTR,sev3,...,"ATC-FUP ...",2024-11-21 11:02:20,2025-01-27 10:17:26,...
        lines = [ln for ln in payload.splitlines() if ln.strip()]
        for ln in lines:
            # attempt CSV parse with python csv (handles quotes)
            try:
                row = next(csv.reader([ln]))
            except Exception:
                row = ln.split(",")

            if not row:
                continue
            rnid = row[0].strip() if len(row) >= 1 else ""
            title = ""
            keywords = ""
            sev = ""

            # Best-effort guess by scanning columns
            for cell in row:
                if "ATC-FUP" in cell:
                    keywords = cell.strip().strip('"')
                if cell.lower().startswith("sev"):
                    sev = cell.strip()
            # title tends to be a long column containing "Follow-up"
            for cell in row:
                if "Follow-up" in cell or "Follow up" in cell:
                    title = cell.strip().strip('"')
                    break

            logged = None
            closed = None
            # find timestamps in row
            ts = [c for c in row if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", c)]
            if len(ts) >= 1:
                logged = ts[0].strip().strip('"')
            if len(ts) >= 2:
                closed = ts[1].strip().strip('"')

            if rnid.isdigit():
                candidates.append(FupCandidate(rnid=rnid, title=title, keywords=keywords, severity=sev, logged_date=logged, closed_date=closed, raw=row))
        return candidates

    return candidates

def aproach_full_record(rnid: str, auth: Tuple[str, str]) -> Dict[str, Any]:
    # https://aproach-api.muc.amadeus.net/api/v2/json/records/?rnid=29690823&with=full
    url = f"{APROACH_BASE_URL}/records/"
    params = {"rnid": rnid, "with": "full"}
    r = requests.get(url, params=params, auth=auth, timeout=DEFAULT_TIMEOUT, verify=VERIFY_SSL)
    r.raise_for_status()
    return r.json()

# =========================
# RECORD TEXT EXTRACTION
# =========================
def record_to_text(record_json: Dict[str, Any]) -> str:
    """
    Build a readable text from a full Aproach record:
    - title
    - keywords
    - ffts chronologically with timestamps
    """
    records = record_json.get("records") or []
    if not records:
        return ""

    rec = records[0]
    common = rec.get("common") or {}
    title = common.get("title") or ""
    keywords = common.get("keywords") or ""
    rnid = common.get("rnid") or rec.get("main", {}).get("rnid") or ""

    parts = []
    parts.append(f"RNID: {rnid}")
    parts.append(f"TITLE: {title}")
    parts.append(f"KEYWORDS: {keywords}")
    parts.append("")

    ffts = rec.get("ffts") or []
    # sort by update_date when present
    def _key(x):
        return x.get("update_date") or x.get("correction_date") or ""
    ffts_sorted = sorted(ffts, key=_key)

    for i, f in enumerate(ffts_sorted, start=1):
        upd = f.get("update_date") or f.get("correction_date") or ""
        txt = f.get("fft") or ""
        if not isinstance(txt, str):
            try:
                txt = json.dumps(txt, ensure_ascii=False)
            except Exception:
                txt = str(txt)
        txt = normalize_ws(txt)
        parts.append(f"--- FFT #{i} | {upd} ---")
        parts.append(txt)
        parts.append("")

    return "\n".join(parts).strip()

def extract_questionnaire_context(full_text: str, window_chars: int = 12000) -> str:
    """
    Extract the part around 'Problem Management Questionnaire' and also include
    the latest content (answers often appear in later FFTs).
    """
    low = full_text.lower()
    idx = low.rfind(QUESTIONNAIRE_MARKER)
    if idx == -1:
        return full_text[-window_chars:] if len(full_text) > window_chars else full_text

    start = max(0, idx - 4000)
    end = min(len(full_text), idx + window_chars)
    ctx = full_text[start:end]

    # Always include the tail (latest FFTs) to capture actual answers
    tail = full_text[-4000:] if len(full_text) > 4000 else full_text
    combined = ctx + "\n\n[TAIL]\n" + tail
    return combined

# =========================
# TEMPLATE DETECTION (light)
# =========================
def detect_template(full_text: str) -> Optional[str]:
    """
    Scores templates by presence of distinctive phrases.
    Not perfect, but helps LLM + reporting.
    """
    t = full_text.lower()
    scores = {}
    for name, qs in TEMPLATES.items():
        score = 0
        # Score by key phrases in questions
        for q in qs[:2]:  # first two usually distinctive
            qk = re.sub(r"\(optional\)", "", q, flags=re.I).strip().lower()
            # take 2-4 keywords from the question
            kws = [w for w in re.findall(r"[a-z]{4,}", qk) if w not in {"provide", "please", "fixed", "permanently", "countermeasures"}]
            for kw in kws[:4]:
                if kw in t:
                    score += 1
        scores[name] = score

    best = max(scores.items(), key=lambda kv: kv[1])
    return best[0] if best[1] > 0 else None

# =========================
# LLM (GPT-OSS) CLASSIFIER
# =========================
def llm_classify_questionnaire(questionnaire_text: str) -> Dict[str, Any]:
    """
    Uses a local OpenAI-compatible server (vLLM recommended) running gpt-oss-20b.
    Returns strict JSON in a dict.
    """
    # Lazy import to keep requirements light if LLM disabled
    from openai import OpenAI

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    templates_compact = {k: v for k, v in TEMPLATES.items()}

    rubric = """
You must label a Follow-up PTR questionnaire into exactly one of:
- Q-COMPLETE: questionnaire exists and all REQUIRED questions are answered meaningfully.
- Q-PARTIAL: questionnaire exists and at least one required question is missing OR answers are vague/weak (e.g., "N/A", "done", "ok") OR lacks evidence where asked (references/links).
- Q-NONE: questionnaire exists but no real answers (template only, greetings only, or "no response").
- Q-MISSING: questionnaire is not present at all.

Rules:
- Treat questions marked "(optional)" as NOT required.
- If the PTR says "[Remove/ignore for QCP IR]" that question is optional.
- "Meaningful" answer: non-empty, not just "NA", "none", "done", "ok", "TBD". Should include explanation; if asked for reference, must include an id or ticket/IR/Win@proach reference (or clearly state not yet available with plan).
- The record may contain multiple FFT updates; prefer the LATEST answers for each question.
Return JSON only. No markdown.
"""

    prompt = {
        "templates": templates_compact,
        "record_excerpt": questionnaire_text
    }

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You are a strict JSON-only classifier for Amadeus Follow-up PTR questionnaires."},
            {"role": "user", "content": rubric + "\n\nINPUT:\n" + json.dumps(prompt, ensure_ascii=False)}
        ],
        temperature=0.1,
        max_tokens=900
    )

    content = resp.choices[0].message.content.strip()

    # Strict JSON parse with small recovery
    try:
        return json.loads(content)
    except Exception:
        # Try to extract first JSON object
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            return json.loads(m.group(0))
        raise

def heuristic_label(full_text: str) -> LabelResult:
    """
    Very lightweight fallback if LLM is disabled/unavailable.
    """
    low = full_text.lower()
    if QUESTIONNAIRE_MARKER not in low:
        return LabelResult(rnid="?", label="Q-MISSING", template=detect_template(full_text), confidence=0.55,
                           rationale="No 'Problem Management Questionnaire' section found.")
    # If only template questions but no apparent answers:
    # heuristic: look for question lines AND a following non-empty line that is not another question.
    lines = [ln.strip() for ln in full_text.splitlines()]
    q_idx = [i for i, ln in enumerate(lines) if re.match(r"^\s*\d+\s*-\s*", ln)]
    answered = 0
    for i in q_idx:
        # scan next 1-3 lines for an answer
        for j in range(i+1, min(i+4, len(lines))):
            nxt = lines[j]
            if not nxt:
                continue
            if re.match(r"^\s*\d+\s*-\s*", nxt):
                break
            # ignore greetings / sign-offs
            if nxt.lower().startswith(("hello", "thanks", "regards")):
                continue
            # ignore pure template markers
            if nxt.lower().startswith(("<complete executive summary>", "!objective")):
                continue
            # ignore placeholders
            if nxt.lower() in {"na", "n/a", "tbd", "none"}:
                continue
            answered += 1
            break

    template = detect_template(full_text)
    required = None
    if template and template in TEMPLATES:
        required = sum(1 for q in TEMPLATES[template] if "(optional)" not in q.lower())

    if answered == 0:
        return LabelResult(rnid="?", label="Q-NONE", template=template, confidence=0.55,
                           rationale="Questionnaire found but no detected answers (heuristic).",
                           answered_questions=0, required_questions=required)
    if required is not None and answered >= required:
        return LabelResult(rnid="?", label="Q-COMPLETE", template=template, confidence=0.55,
                           rationale="All required questions appear answered (heuristic).",
                           answered_questions=answered, required_questions=required)
    return LabelResult(rnid="?", label="Q-PARTIAL", template=template, confidence=0.55,
                       rationale="Some answers detected but not all required (heuristic).",
                       answered_questions=answered, required_questions=required)

def label_record(rnid: str, record_json: Dict[str, Any]) -> LabelResult:
    full_text = record_to_text(record_json)
    template_guess = detect_template(full_text)
    if QUESTIONNAIRE_MARKER not in full_text.lower():
        return LabelResult(rnid=rnid, label="Q-MISSING", template=template_guess, confidence=0.80,
                           rationale="No 'Problem Management Questionnaire' section found in record text.")

    ctx = extract_questionnaire_context(full_text)

    if not LLM_ENABLED:
        h = heuristic_label(ctx)
        h.rnid = rnid
        h.template = template_guess
        return h

    try:
        out = llm_classify_questionnaire(ctx)
    except Exception as e:
        # fallback
        h = heuristic_label(ctx)
        h.rnid = rnid
        h.template = template_guess
        h.rationale = f"LLM failed, used heuristic. Error: {type(e).__name__}: {e}"
        return h

    # Expected keys from model
    label = out.get("label") or out.get("classification") or out.get("result")
    if label not in {"Q-COMPLETE", "Q-PARTIAL", "Q-NONE", "Q-MISSING"}:
        # Normalize common variants
        mapping = {
            "complete": "Q-COMPLETE",
            "partial": "Q-PARTIAL",
            "none": "Q-NONE",
            "missing": "Q-MISSING"
        }
        label = mapping.get(str(label).strip().lower(), "Q-PARTIAL")

    tpl = out.get("template") or template_guess
    conf = float(out.get("confidence", 0.75))
    aq = out.get("answered_questions")
    rq = out.get("required_questions")
    miss = out.get("missing_questions")
    rationale = out.get("rationale") or out.get("notes")

    return LabelResult(
        rnid=rnid,
        label=label,
        template=tpl,
        confidence=max(0.0, min(1.0, conf)),
        answered_questions=aq if isinstance(aq, int) else None,
        required_questions=rq if isinstance(rq, int) else None,
        missing_questions=miss if isinstance(miss, list) else None,
        rationale=rationale if isinstance(rationale, str) else None
    )

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="FUP Questionnaire Auto-Labeler (GPT-OSS)", layout="wide")
st.title("FUP Questionnaire Auto-Labeler (GPT-OSS)")

st.info(
    "Labels:\n"
    "- **Q-COMPLETE**: ideal closure (all required questions answered meaningfully)\n"
    "- **Q-PARTIAL**: answered but weak / missing checks\n"
    "- **Q-NONE**: no response (template only / no real answers)\n"
    "- **Q-MISSING**: questionnaire not added\n\n"
    "SSL verification is **disabled** in this script as requested."
)

with st.sidebar:
    st.header("Input")
    if not (AUTH_USERNAME and AUTH_PASSWORD):
        st.warning("Set auth_username/auth_password in .streamlit/secrets.toml (do NOT hardcode creds).")

    start_date = st.date_input("Start date", value=dt.date(dt.date.today().year, 1, 1))
    end_date = st.date_input("End date", value=dt.date.today())
    max_records = st.number_input("Max FUP PTRs to process", min_value=1, max_value=5000, value=200, step=50)
    run_llm = st.checkbox("Use GPT-OSS model (LLM)", value=LLM_ENABLED)
    st.caption("If unchecked, uses heuristic labeling.")
    LLM_ENABLED = run_llm  # override from UI

    fetch_btn = st.button("Fetch FUP PTRs")

auth = (AUTH_USERNAME, AUTH_PASSWORD)

@st.cache_data(show_spinner=False, ttl=3600)
def cached_teamactivity(year: int, month: Optional[int], user: str, pwd: str):
    return dashproach_teamactivity(year, month, auth=(user, pwd))

@st.cache_data(show_spinner=False, ttl=3600)
def cached_full_record(rnid: str, user: str, pwd: str):
    return aproach_full_record(rnid, auth=(user, pwd))

def is_fup(candidate: FupCandidate) -> bool:
    return "ATC-FUP" in (candidate.keywords or "") or "ATC-FUP" in (str(candidate.raw) if candidate.raw is not None else "")

if fetch_btn:
    if start_date > end_date:
        st.error("Start date must be <= End date.")
        st.stop()

    months = month_range(start_date, end_date)
    st.write(f"Fetching teamactivity for months: {', '.join([f'{y}-{m:02d}' for y,m in months])}")

    all_candidates: List[FupCandidate] = []
    for (y, m) in months:
        try:
            payload = cached_teamactivity(y, m, AUTH_USERNAME, AUTH_PASSWORD)
        except Exception as e:
            st.error(f"Dashproach fetch failed for {y}-{m:02d}: {type(e).__name__}: {e}")
            st.stop()

        cands = parse_teamactivity_payload(payload)
        all_candidates.extend(cands)

    # Filter only FUP PTRs by keywords
    fups = [c for c in all_candidates if is_fup(c)]
    st.success(f"Found {len(fups)} PTRs with ATC-FUP keyword (before interval filtering).")

    # Filter by closed_date/logged_date if present (best-effort)
    def in_interval(c: FupCandidate) -> bool:
        for dstr in [c.closed_date, c.logged_date]:
            if not dstr:
                continue
            # parse 'YYYY-MM-DD HH:MM:SS' or ISO
            try:
                if "T" in dstr:
                    d = dt.datetime.fromisoformat(dstr.replace("Z", "+00:00")).date()
                else:
                    d = dt.datetime.strptime(dstr[:19], "%Y-%m-%d %H:%M:%S").date()
                return start_date <= d <= end_date
            except Exception:
                continue
        # if no parsable date, keep it (user can still label)
        return True

    fups = [c for c in fups if in_interval(c)]
    fups = fups[: int(max_records)]
    st.write(f"Processing {len(fups)} FUP PTRs (limited by Max).")

    # Display candidates
    df_cand = pd.DataFrame([{
        "rnid": c.rnid,
        "severity": c.severity,
        "logged_date": c.logged_date,
        "closed_date": c.closed_date,
        "title": c.title,
        "keywords": c.keywords
    } for c in fups])
    st.dataframe(df_cand, use_container_width=True, height=260)

    # Labeling
    results: List[Dict[str, Any]] = []
    prog = st.progress(0)
    status = st.empty()

    for i, c in enumerate(fups, start=1):
        status.write(f"({i}/{len(fups)}) Fetching + labeling RNID {c.rnid} ...")
        try:
            rec_json = cached_full_record(c.rnid, AUTH_USERNAME, AUTH_PASSWORD)
            res = label_record(c.rnid, rec_json)
        except Exception as e:
            res = LabelResult(rnid=c.rnid, label="Q-PARTIAL", template=None, confidence=0.20,
                              rationale=f"Error fetching/labeling: {type(e).__name__}: {e}")

        results.append({
            "rnid": res.rnid,
            "label": res.label,
            "template": res.template,
            "confidence": res.confidence,
            "answered_questions": res.answered_questions,
            "required_questions": res.required_questions,
            "missing_questions": json.dumps(res.missing_questions, ensure_ascii=False) if res.missing_questions else "",
            "rationale": res.rationale or "",
            "title": c.title,
            "severity": c.severity,
            "logged_date": c.logged_date,
            "closed_date": c.closed_date,
            "keywords": c.keywords,
        })

        prog.progress(i / len(fups))

    status.write("Done.")

    df_out = pd.DataFrame(results)

    # Summary metrics
    st.subheader("Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Label distribution")
        st.dataframe(df_out["label"].value_counts(dropna=False).to_frame("count"), use_container_width=True)
    with col2:
        st.write("Avg confidence by label")
        st.dataframe(df_out.groupby("label")["confidence"].mean().sort_values(ascending=False).to_frame("avg_confidence"),
                     use_container_width=True)

    st.subheader("Detailed results")
    st.dataframe(df_out, use_container_width=True, height=420)

    # Download
    csv_bytes = df_out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv_bytes, file_name=f"fup_labels_{start_date}_{end_date}.csv", mime="text/csv")

    st.caption("Tip: If you want to write these labels back into Kibana/Elastic, add a write-back function once the target index & API are confirmed.")
