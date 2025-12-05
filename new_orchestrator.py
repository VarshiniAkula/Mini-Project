# new_orchestrator.py
# Orchestrator utilities to pick ONE retriever using an LLM with a strict two-key JSON output.

from __future__ import annotations
import os, re, json
from pathlib import Path
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ALLOWED = ["bm25", "tfidf", "svm", "faiss", "time_weighted"]

# ------------------------
# Guide loader
# ------------------------
def _load_selection_guide(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data

# ------------------------
# Signals from query text
# ------------------------
TEMPORAL_PAT = re.compile(
    r"\b(today|yesterday|tonight|this (morning|afternoon|evening|week|month|year)|"
    r"last (night|week|month|year|weekend)|next (week|month|year)|"
    r"now|currently|recently|since|before|after|during|"
    r"\b(19\d{2}|20\d{2})\b|\b\d{1,2}:\d{2}\b)\b",
    flags=re.I
)
UPDATE_VERBS = re.compile(r"\b(now|changed|switch(?:ed|ing)?|update(?:d)?|still|anymore)\b", flags=re.I)
# Detect uppercase codes (e.g., ACME-42) or quoted phrases
QUOTE_LIKE = re.compile(r'[A-Z]{2,}[-_]*\d{2,}|"[^"]+"|\'[^\']+\'')

# Detect currency + number (e.g., $5)
MONEY = re.compile(r"[$€£]\s?\d")

PROPER       = re.compile(r"\b([A-Z][a-z]+){1,}\b")

def extract_signals(query: str) -> Dict[str, Any]:
    toks = re.findall(r"[A-Za-z0-9$€£:_\-]+", query or "")
    return {
        "len_tokens": len(toks),
        "short_query": len(toks),
        "has_temporal_cue": bool(TEMPORAL_PAT.search(query or "")),
        "update_verbs": bool(UPDATE_VERBS.search(query or "")),
        "num_digits": sum(ch.isdigit() for ch in (query or "")),
        "has_money": bool(MONEY.search(query or "")),
        "proper_noun": len(PROPER.findall(query or "")) > 0,
        "quote_like": bool(QUOTE_LIKE.search(query or "")),
        "idf_like_high": sum(1 for t in toks if len(t) >= 6 or any(c.isupper() for c in t)) >= 2,
        "idf_like_low":  sum(1 for t in toks if len(t) <= 3) >= max(2, max(1, len(toks)//3)),
        "paraphrase_synonym_hint": any(k in (query or "").lower()
                                       for k in ["talk about", "discuss", "mention", "describe", "summary", "overall", "similar", "like the"])
    }

# ------------------------
# Few-shot selection (from guide)
# ------------------------
def _pick_few_shots(query: str, guide: Dict[str, Any], k: int = 6) -> List[Dict[str, Any]]:
    shots = guide.get("few_shots", []) or []
    if not shots:
        return []
    texts = [s["q"] for s in shots]
    vect = TfidfVectorizer(stop_words="english")
    X = vect.fit_transform(texts + [query])
    sims = cosine_similarity(X[-1], X[:-1])[0]
    order = sims.argsort()[::-1][:k]
    return [shots[i] for i in order]

# ------------------------
# Prompt builder
# ------------------------
def build_selector_prompt(query: str, signals: Dict[str, Any], guide: Dict[str, Any], k_shots: int = 6) -> str:
    allowed = guide.get("allowed_values", ALLOWED)
    rubric = guide.get("decision_rubric", [])
    ties   = guide.get("tie_breakers", [])
    anti   = guide.get("anti_patterns", [])
    shots  = _pick_few_shots(query, guide, k=k_shots)

    shots_block = "\n".join(
        [f'Q: {s["q"]}\n→ {{"retriever":"{s["label"]}","CoT reasoning":"{s["why"]}"}}' for s in shots]
    )

    prompt = (
            "You are a routing policy that selects exactly one retriever for personal-memory QA.\n\n"
            "Retrievers (choose one):\n"
            "- bm25: exact/rare terms, numbers, quoted phrases\n"
            "- tfidf: general lexical overlap, breadth aggregation\n"
            "- faiss: paraphrase/synonyms, semantic similarity\n"
            "- svm: short, pattern-like user turns\n"
            "- time_weighted: time-sensitive/recency, explicit dates/times\n\n"
            "Decision rubric (priority order):\n"
            + "\n".join([f"{i+1}) {r}" for i, r in enumerate(rubric)]) + "\n\n"
            "Tie-breakers:\n" + ("\n".join([f"- {t}" for t in ties]) or "- (none)") + "\n\n"
            "Anti-patterns:\n" + ("\n".join([f"- {a}" for a in anti]) or "- (none)") + "\n\n"
            "Output JSON only: {\"retriever\":\"<one>\", \"CoT reasoning\":\"<= 18 words\"}\n"
            f"Allowed retrievers: {allowed}\n"
            "If a tie, pick the earlier rule.\n\n"
            "FEW-SHOT\n" + (shots_block or "(none)") + "\n\n"
            f"USER\nQuery: {query}\nSignals: {json.dumps(signals, ensure_ascii=False)}\n"
            "Return the JSON only."
    )
    return prompt

# ------------------------
# Robust JSON sanitization
# ------------------------
def _parse_two_key_json(raw: str) -> Dict[str, str]:
    try:
        data = json.loads((raw or "").strip())
        if not isinstance(data, dict):
            raise ValueError("not dict")
    except Exception:
        return {"retriever": "tfidf", "CoT reasoning": "Fallback: parse error"}
    # coerce
    ret = data.get("retriever")
    reas = data.get("CoT reasoning") or data.get("reasoning") or data.get("reason") or ""
    if ret not in ALLOWED:
        ret = "tfidf"
        reas = "Fallback: invalid retriever"
    # cap reason to 18 words
    words = (str(reas).strip().split())
    if len(words) > 18:
        reas = " ".join(words[:18])
    return {"retriever": ret, "CoT reasoning": reas}

# ------------------------
# Main selection entry-point
# ------------------------
def llm_select_retriever_two_key(query: str,
                                 question_docs: List[Dict[str, Any]],
                                 guide: Dict[str, Any],
                                 llm_fn,
                                 fewshot_k: int = 6) -> Dict[str, str]:
    """
    Compose the improved prompt with signals + few-shots, call llm_fn(prompt),
    and return exactly two keys: {"retriever": ..., "CoT reasoning": ...}
    """
    # We don't need the docs' content here; we only use the query. (You can add doc-derived features if you like.)
    signals = extract_signals(query)
    prompt = build_selector_prompt(query, signals, guide, k_shots=fewshot_k)

    raw = ""
    try:
        raw = llm_fn(prompt)
    except Exception:
        return {"retriever": "tfidf", "CoT reasoning": "Fallback: llm_fn error"}

    return _parse_two_key_json(raw)

# ------------------------
# Optional default llm_fn (safe fallback)
# ------------------------
def default_llm_fn(prompt: str) -> str:
    """
    If you haven't wired a real LLM yet, this returns a safe default decision.
    To use a backend, replace this with your Gemini/HF/OpenAI client that returns a JSON string.
    """
    return json.dumps({"retriever": "tfidf", "CoT reasoning": "No LLM backend configured"})
