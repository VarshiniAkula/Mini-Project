# new_runner.py
# Runs the improved LLM-based orchestrator on LongMemEval, using Google Gemini 2.5 Flash‑Lite.

from __future__ import annotations
import os, re, json, math, logging, warnings, sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# -----------------------
# Project imports
# -----------------------
from dataloader import CustomData
from new_orchestrator import (
    _load_selection_guide,            # loads the JSON guide (policy, rules, few shots)
    llm_select_retriever_two_key,     # builds prompt + selects ONE retriever
)

# -----------------------
# Retrieval stack
# -----------------------
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import SVMRetriever
from langchain_classic.retrievers.time_weighted_retriever import TimeWeightedVectorStoreRetriever

# Quiet noisy libs
os.environ["LANGCHAIN_VERBOSE"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("langchain_community").setLevel(logging.ERROR)
logging.getLogger("langchain_classic").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# =========================
# Config
# =========================
DATA_LONGMEM_PATH = "data/longmemeval_s_cleaned.json"
GUIDE_PATH        = "improved_training_guide.json"   # Uses your upgraded guide.  :contentReference[oaicite:1]{index=1}
LIMIT_Q: Optional[int] = 100   # None for all
K = 5                           # Top-k

# =========================
# Gemini LLM for orchestration
# =========================
def llm_fn_gemini(prompt: str) -> str:
    """
    Calls Google Gemini 2.5 Flash‑Lite and returns STRICT JSON as a string:
      {"retriever": "<one of bm25|tfidf|svm|faiss|time_weighted>", "CoT reasoning": "<<=18 words>"}
    If the call fails, returns a safe fallback JSON.
    """
    import json as _json
    try:
        import google.generativeai as genai
    except Exception as e:
        # SDK not installed
        return _json.dumps({
            "retriever": "tfidf",
            "CoT reasoning": "google-generativeai not installed; defaulting to TF-IDF."
        }, ensure_ascii=False)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return _json.dumps({
            "retriever": "tfidf",
            "CoT reasoning": "GOOGLE_API_KEY missing; defaulting to TF-IDF."
        }, ensure_ascii=False)

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")

    try:
        model = genai.GenerativeModel(model_name)
        generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 128,
            "response_mime_type": "application/json",
        }
        resp = model.generate_content(prompt, generation_config=generation_config)

        # Prefer resp.text; fall back to first candidate's parts
        raw = (getattr(resp, "text", None) or "").strip()
        if not raw and getattr(resp, "candidates", None):
            parts = getattr(resp.candidates[0].content, "parts", [])
            raw = "".join(getattr(p, "text", "") for p in parts).strip()

        if raw:
            # Return raw; orchestrator will sanitize to exactly two keys.
            return raw

        # Empty model response
        return _json.dumps({
            "retriever": "tfidf",
            "CoT reasoning": "Gemini returned empty; defaulting to TF-IDF."
        }, ensure_ascii=False)
    except Exception:
        # Network/model/parse error → safe fallback
        return _json.dumps({
            "retriever": "tfidf",
            "CoT reasoning": "Gemini call failed; defaulting to TF-IDF."
        }, ensure_ascii=False)


# =========================
# Text utils
# =========================
def preprocess(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return [w for w in text.split() if w]

def _parse_timestamp(timestamp_str: str) -> datetime:
    if not timestamp_str:
        return datetime(2000, 1, 1)
    try:
        parts = (timestamp_str or "").split()
        if len(parts) >= 3:
            date_part = parts[0]  # "YYYY/MM/DD"
            time_part = parts[2]  # "HH:MM"
            date_obj = datetime.strptime(date_part, "%Y/%m/%d")
            hh, mm = map(int, time_part.split(":"))
            return date_obj.replace(hour=hh, minute=mm)
        return datetime(2000, 1, 1)
    except Exception:
        return datetime(2000, 1, 1)

# =========================
# Index building (per Q)
# =========================
_EMB = None
def _embeddings():
    global _EMB
    if _EMB is None:
        print("Loading embeddings model (this may take a moment)...", flush=True)
        _EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMB

def build_indexes_for_question(question_docs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not question_docs:
        return None
    texts = [d["text"] for d in question_docs]

    bm25 = BM25Okapi([preprocess(t) for t in texts])

    tfidf_vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    lc_docs = []
    for i, doc in enumerate(question_docs):
        ts = doc.get("timestamp", "")
        last = _parse_timestamp(ts)
        lc_docs.append(Document(
            page_content=doc["text"],
            metadata={
                "buffer_idx": i,
                "last_accessed_at": last,
                "session_id": doc.get("session_id",""),
                "timestamp": ts,
                "original_dict": doc
            }
        ))

    emb = _embeddings()
    svm_retriever = SVMRetriever.from_texts(texts, emb) if texts else None

    if lc_docs:
        vectorstore = FAISS.from_documents(lc_docs, emb)
        time_weighted = TimeWeightedVectorStoreRetriever(
            vectorstore=vectorstore,
            memory_stream=lc_docs,
            search_kwargs={"k": min(50, len(lc_docs))}
        )
    else:
        vectorstore, time_weighted = None, None

    return {
        "bm25": bm25,
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "svm_retriever": svm_retriever,
        "vectorstore": vectorstore,
        "time_weighted_retriever": time_weighted,
        "question_docs": question_docs
    }

# =========================
# Retrievals
# =========================
def r_bm25(q: str, k: int, idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    scores = idx["bm25"].get_scores(preprocess(q))
    order = (-scores).argsort()[:k]
    return [idx["question_docs"][i] for i in order]

def r_tfidf(q: str, k: int, idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = idx["tfidf_vectorizer"].transform([q])
    scores = cosine_similarity(v, idx["tfidf_matrix"])[0]
    order = scores.argsort()[::-1][:k]
    return [idx["question_docs"][i] for i in order]

def r_svm(q: str, k: int, idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = idx["svm_retriever"].invoke(q)[:k]
    map_by_text = {d["text"]: d for d in idx["question_docs"]}
    out = []
    for d in docs:
        pc = getattr(d, "page_content", "")
        out.append(map_by_text.get(pc, {"text": pc, "session_id": getattr(d, "metadata", {}).get("session_id","")}))
    return out

def r_faiss(q: str, k: int, idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    res = idx["vectorstore"].similarity_search_with_score(q, k=k)
    return [doc.metadata.get("original_dict", {}) for doc, _ in res]

def r_time(q: str, k: int, idx: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = idx["time_weighted_retriever"].invoke(q)[:k]
    return [d.metadata.get("original_dict", {}) for d in docs]

RUN_MAP = {
    "bm25": r_bm25,
    "tfidf": r_tfidf,
    "svm": r_svm,
    "faiss": r_faiss,
    "time_weighted": r_time
}

# =========================
# Metrics
# =========================
def recall_at_k(retrieved_ids: List[str], relevant_ids: set, k: int = 5) -> float:
    if not relevant_ids: return 0.0
    top = set((retrieved_ids or [])[:k])
    return len(top & set(relevant_ids)) / float(len(relevant_ids))

def ndcg_at_k(retrieved_ids: List[str], relevant_ids: set, k: int = 5) -> float:
    if not relevant_ids: return 0.0
    dcg = 0.0; seen = set()
    for i, sid in enumerate((retrieved_ids or [])[:k], start=1):
        rel = 1 if (sid in relevant_ids and sid not in seen) else 0
        if rel:
            seen.add(sid)
            dcg += (2**rel - 1) / math.log2(i + 1)
    idcg = sum((2**1 - 1) / math.log2(i + 1) for i in range(1, min(k, len(relevant_ids)) + 1))
    return (dcg / idcg) if idcg else 0.0

# =========================
# Robust doc/label getters
# =========================
def get_relevant_session_ids(qentry: Dict[str, Any]) -> set:
    # Your QA uses 'evidence_ids'. Fall back to others if needed.
    ids = qentry.get("evidence_ids") or qentry.get("answer_session_ids") or qentry.get("answer_ids") or []
    ids = [s for s in ids or [] if isinstance(s, str) and s.strip()]
    return set(ids)

def docs_for_question(question_id: str,
                      qentry: Dict[str, Any],
                      col2docs: Dict[str, List[Dict[str, Any]]],
                      sess2docs: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    key = f"vectordb_collection_{question_id}"
    docs = col2docs.get(key, [])
    if docs:
        return docs
    # fallback: build from ground-truth session ids if collection missing
    merged = []
    for sid in get_relevant_session_ids(qentry):
        merged.extend(sess2docs.get(sid, []))
    return merged

# =========================
# Main
# =========================
def main():
    # 1) Load selection guide (policy, rubric, few-shots)
    guide = _load_selection_guide(GUIDE_PATH)  # improved guide v2  :contentReference[oaicite:2]{index=2}

    # 2) Load data
    print("Loading data...", flush=True)
    data = CustomData(path_locomo=None, path_longmem_eval=DATA_LONGMEM_PATH)
    print("Processing longmem data...", flush=True)
    longmem = data.process_longmem_eval()

    # 3) Build user-message docs (mirror longmem_approach.py)
    print("Creating documents from sessions...", flush=True)
    documents: List[Dict[str, Any]] = []
    for collection_key, session_entries in longmem.items():
        for e in session_entries:
            sess = e.get("haystack_session", []) or []
            ts = e.get("haystack_date", "")
            sid = e.get("haystack_session_id", "")
            for msg in sess:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    documents.append({
                        "text": msg.get("content",""),
                        "timestamp": ts,
                        "session_id": sid,
                        "collection_key": collection_key,
                        "role": "user"
                    })
    print(f"Created {len(documents)} documents", flush=True)

    # 4) Load QA
    qa_all = data.load_source_qa_longmem_eval()
    if LIMIT_Q is not None:
        qa = {}
        count = 0
        for qid, qs in qa_all.items():
            if count >= LIMIT_Q: break
            for q in qs:
                if count >= LIMIT_Q: break
                qa.setdefault(qid, []).append(q)
                count += 1
        print(f"Limiting to {count} questions")
    else:
        qa = qa_all

    # 5) Build mappings
    col2docs = defaultdict(list)
    for d in documents:
        col2docs[d.get("collection_key","")].append(d)

    sess2docs = defaultdict(list)
    for d in documents:
        sid = d.get("session_id","")
        if sid:
            sess2docs[sid].append(d)

    # metadata (optional)
    qmeta = {}
    for item in data.longmem_eval_data:
        qid = item.get("question_id")
        if qid:
            qmeta[qid] = {
                "question_type": item.get("question_type",""),
                "question_date": item.get("question_date","")
            }

    # 6) Evaluate
    recalls, ndcgs = [], []
    rows: List[Dict[str, Any]] = []

    total = sum(len(v) for v in qa.values())
    processed = 0
    print(f"Starting evaluation of {total} questions...\n")

    for qid, qlist in qa.items():
        for qentry in qlist:
            gt = get_relevant_session_ids(qentry)
            if not gt:
                continue
            qdocs = docs_for_question(qid, qentry, col2docs, sess2docs)
            if not qdocs:
                continue
            query = qentry.get("question","")
            if not query:
                continue

            idx = build_indexes_for_question(qdocs)
            if idx is None:
                continue

            # --- Use Gemini to choose ONE retriever (strict two keys) ---
            decision = llm_select_retriever_two_key(query, qdocs, guide, llm_fn=llm_fn_gemini)
            chosen = decision["retriever"]
            run = RUN_MAP.get(chosen, r_tfidf)  # safety

            # --- Run chosen retriever ---
            top_docs = run(query, K, idx)
            top_ids = [d.get("session_id","") for d in top_docs]

            r5 = recall_at_k(top_ids, gt, k=K)
            n5 = ndcg_at_k(top_ids, gt, k=K)
            recalls.append(r5); ndcgs.append(n5)

            meta = qmeta.get(qid, {})
            rows.append({
                "question_id": qid,
                "question_type": meta.get("question_type",""),
                "question_date": meta.get("question_date",""),
                "question": query,
                "answer": qentry.get("answer",""),
                "evidence_ids": list(gt),
                "Retriever": {
                    "Orchestrator": {
                        "decision": decision,
                        "Top5_retrieved_ids": top_ids,
                        "recall@5": r5,
                        "ndcg@5": n5
                    }
                }
            })

            processed += 1
            if processed % 10 == 0:
                avg_r = sum(recalls)/len(recalls) if recalls else 0.0
                avg_n = sum(ndcgs)/len(ndcgs) if ndcgs else 0.0
                print(f"Processed {processed}/{total} | Orchestrator Recall@5={avg_r:.4f} NDCG@5={avg_n:.4f}", flush=True)

    # 7) Final report
    print(f"\nProcessed {processed}/{total}.\n")
    avg_r = sum(recalls)/len(recalls) if recalls else 0.0
    avg_n = sum(ndcgs)/len(ndcgs) if ndcgs else 0.0
    print("="*54)
    print("FINAL RESULTS — ORCHESTRATOR (Gemini 2.5 Flash‑Lite)")
    print("="*54)
    print(f"Total Questions Evaluated: {processed}")
    print(f"Recall@5: {avg_r:.4f}")
    print(f"NDCG@5 : {avg_n:.4f}")
    print("="*54)

    out = "new_longmem_orchestrator_results.json"
    Path(out).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved per-question results to {out}")

if __name__ == "__main__":
    main()
