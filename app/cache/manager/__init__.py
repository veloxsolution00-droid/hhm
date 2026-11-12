"""Bounded process-local cache. It never owns durable document data."""
from __future__ import annotations
import hashlib, json, re, threading, time, math
from collections import OrderedDict
from copy import deepcopy
from app.core.config import settings

_lock=threading.RLock(); _results=OrderedDict(); _embeddings=OrderedDict()

def _key(*parts): return hashlib.sha256(json.dumps(parts,sort_keys=True,default=str,separators=(",",":")).encode()).hexdigest()
def normalize_query(query): return " ".join(str(query or "").lower().split())
def config_fingerprint(kind, config): return _key(kind,config)
def corpus_fingerprint(documents, indexed):
    rows=[(d.get("document_id"),d.get("updated_at"),d.get("chunk_count"),d.get("indexing_status")) for d in documents or []]
    return _key(sorted(rows),sorted(indexed or []))
def _prune(cache):
    now=time.time()
    for k in list(cache):
        if cache[k].get("expires",0)<=now: cache.pop(k,None)
    while len(cache)>max(1,int(settings.cache_max_entries)): cache.popitem(last=False)
def _anchors(text):
    """Protect numeric values and explicitly capitalized acronyms in semantic reuse."""
    value = str(text or "")
    numeric = set(re.findall(r"\b\d+(?:[.,/]\d+)*\b|\b(?:19|20)\d{2}\b", normalize_query(value)))
    acronyms = {token.lower() for token in re.findall(r"\b[A-Z][A-Z0-9&-]{1,}\b", value)}
    return numeric | {f"acronym:{token}" for token in acronyms}

def _cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    dot=sum(float(x)*float(y) for x,y in zip(a,b)); na=math.sqrt(sum(float(x)**2 for x in a)); nb=math.sqrt(sum(float(y)**2 for y in b))
    return dot/(na*nb) if na and nb else 0.0

def lookup_result(kind, query, owner_id, corpus_version, config_version, query_vector=None, question_type=None, semantic_threshold=None):
    if not settings.cache_enabled: return None
    normalized=normalize_query(query); key=_key(kind,owner_id,corpus_version,config_version,normalized,question_type)
    with _lock:
        _prune(_results)
        item=_results.get(key)
        if item:
            _results.move_to_end(key)
            return {"payload":deepcopy(item["payload"]),"match_kind":"exact","similarity":1.0}
        if query_vector is None or semantic_threshold is None: return None
        if kind=="answer" and not settings.semantic_answer_cache_enabled: return None
        current_anchors=_anchors(query)
        for cached in reversed(_results.values()):
            if cached.get("kind")!=kind or cached.get("owner")!=owner_id or cached.get("corpus")!=corpus_version or cached.get("config")!=config_version or cached.get("question_type")!=question_type: continue
            if current_anchors != cached.get("anchors",set()): continue
            score=_cosine(query_vector,cached.get("query_vector"))
            if score >= float(semantic_threshold):
                return {"payload":deepcopy(cached["payload"]),"match_kind":"semantic","similarity":score,"cached_query":cached.get("query")}
    return None

def put_result(kind, query, owner_id, corpus_version, config_version, payload, query_vector=None, question_type=None):
    if not settings.cache_enabled: return
    # Never cache an ungrounded/failed answer. Retrieval payloads are cacheable only when they contain evidence.
    if kind=="answer":
        answer=str(payload.get("answer", "")); refs=re.findall(r"\[(\d+)\]",answer)
        citation = payload.get("citation_validation") or {}
        verification = payload.get("verification") or {}
        if (payload.get("status") != "success" or payload.get("abstained") or not refs
                or not payload.get("sources") or not citation.get("valid")
                or not (verification.get("passed") or verification.get("verdict") == "EXACT_PASSAGE_MATCH")):
            return
    if kind=="retrieval" and not (payload.get("hits") or payload.get("sources")): return
    normalized=normalize_query(query); key=_key(kind,owner_id,corpus_version,config_version,normalized,question_type)
    with _lock:
        _results[key]={"kind":kind,"owner":owner_id,"corpus":corpus_version,"config":config_version,
                       "query":normalized,"anchors":_anchors(query),"query_vector":list(query_vector) if query_vector is not None else None,
                       "question_type":question_type,"payload":deepcopy(payload),"expires":time.time()+settings.cache_ttl_seconds}
        _results.move_to_end(key); _prune(_results)

def get_embedding(text, model="default"):
    if not settings.cache_enabled: return None
    key=_key(model,normalize_query(text))
    with _lock:
        _prune(_embeddings); item=_embeddings.get(key)
        if item: _embeddings.move_to_end(key); return list(item["value"])
    return None

def put_embedding(text, model, vector):
    if not settings.cache_enabled: return
    key=_key(model,normalize_query(text))
    with _lock:
        _embeddings[key]={"value":list(vector),"expires":time.time()+settings.cache_ttl_seconds}
        _embeddings.move_to_end(key); _prune(_embeddings)

def stats():
    with _lock:
        _prune(_results); _prune(_embeddings)
        return {"enabled":bool(settings.cache_enabled),"entries":len(_results)+len(_embeddings),"result_entries":len(_results),"embedding_entries":len(_embeddings),"max_entries":settings.cache_max_entries}


def invalidate_document(document_id):
    """Remove cached retrievals/answers that contain evidence from a document."""
    target = str(document_id)
    removed = 0
    with _lock:
        for key, item in list(_results.items()):
            payload = item.get("payload") or {}
            evidence_ids = set()
            for hit in payload.get("hits") or []:
                doc_id = (hit.get("metadata") or {}).get("document_id")
                if doc_id:
                    evidence_ids.add(str(doc_id))
            for source in payload.get("sources") or []:
                doc_id = source.get("document_id")
                if doc_id:
                    evidence_ids.add(str(doc_id))
            if target in evidence_ids:
                _results.pop(key, None)
                removed += 1
    return removed


def clear():
    """Clear process-local caches (primarily useful for isolated tests)."""
    with _lock:
        _results.clear()
        _embeddings.clear()
