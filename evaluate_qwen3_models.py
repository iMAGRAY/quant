#!/usr/bin/env python
"""evaluate_qwen3_models.py

Оценивает качество пары моделей:
  • эмбеддинговой (feature-extraction) – например, Qwen3-Embedding-0.6B
  • реранкера (sequence-classification) – например, Qwen3-Reranker-0.6B

Скрипт принимает заранее подготовленный датасет в формате BEIR/pyserini:
  dataset/
    ├── corpus.jsonl   # {"_id": str, "text": str}
    ├── queries.jsonl  # {"_id": str, "text": str}
    └── qrels.tsv      # query_id \t doc_id

Метрики:
  • Recall@{1,10,100}
  • MRR@10

Пример запуска:

    python evaluate_qwen3_models.py \
        --dataset ./data/msmarco-mini \
        --embedding onnx/qwen3/model.int8.onnx \
        --reranker  onnx/qwen3_reranker/model.int8.onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import onnxruntime as ort
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# ─────────────────────────────────────────────────────────── utils ──

def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        return {d["_id"]: d for d in map(json.loads, fh)}


def _read_qrels(path: Path) -> dict[str, set[str]]:
    rels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            qid, did, *_ = line.strip().split("\t")
            rels[qid].add(did)
    return rels


# ─────────────────────────────────────────────── inference ──

class OnnxEncoder:
    """Обёртка для эмбеддинговой модели (sentence-level)."""

    def __init__(self, model_path: Path, device: str = "cpu", batch_size: int = 32):
        self.session = ort.InferenceSession(model_path.as_posix(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"])
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.parent, trust_remote_code=True)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def _mean_pool(self, last_hidden: np.ndarray, mask: np.ndarray):
        mask_exp = mask[..., None]
        sum_emb = (last_hidden * mask_exp).sum(axis=1)
        lengths = mask_exp.sum(axis=1)
        return sum_emb / np.clip(lengths, a_min=1e-9, a_max=None)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return 2-D float32 array (n, d)."""
        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tok = self.tokenizer(batch, padding=True, truncation=True, return_tensors="np")
            inputs = {k: v for k, v in tok.items() if k in self.input_names}
            outs = self.session.run(None, inputs)
            last_hidden = outs[0]  # shape (B, L, D)
            pooled = self._mean_pool(last_hidden, tok["attention_mask"])
            embeddings.append(pooled.astype(np.float32))
        return np.vstack(embeddings)


class OnnxReranker:
    """Sequence-classification scorer: возвращает logit[0]."""

    def __init__(self, model_path: Path, device: str = "cpu", batch_size: int = 32):
        self.session = ort.InferenceSession(model_path.as_posix(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"])
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.parent, trust_remote_code=True)
        self.input_names = {i.name for i in self.session.get_inputs()}

    def score(self, pairs: List[tuple[str, str]]) -> List[float]:
        scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch_pairs = pairs[i : i + self.batch_size]
            texts_a, texts_b = zip(*batch_pairs)
            toks = self.tokenizer(list(texts_a), list(texts_b), padding=True, truncation=True, return_tensors="np")
            inputs = {k: v for k, v in toks.items() if k in self.input_names}
            logits = self.session.run(None, inputs)[0]  # (B, 1) or (B, 2)
            # Если две логиты – берём score для класса 1
            if logits.shape[1] == 1:
                scores.extend(logits.squeeze(1).tolist())
            else:
                scores.extend(logits[:, 1].tolist())
        return scores


# ─────────────────────────────────────────────── metrics ──


def recall_at_k(ranked_lists: List[List[str]], qrels: dict[str, set[str]], k: int) -> float:
    hits = 0
    total = len(ranked_lists)
    for qid, docs in ranked_lists:
        hits += any(did in qrels[qid] for did in docs[:k])
    return hits / total if total else 0.0


def mrr_at_k(ranked_lists: List[List[str]], qrels: dict[str, set[str]], k: int) -> float:
    total = 0.0
    for qid, docs in ranked_lists:
        for rank, did in enumerate(docs[:k], 1):
            if did in qrels[qid]:
                total += 1.0 / rank
                break
    return total / len(ranked_lists) if ranked_lists else 0.0


# ─────────────────────────────────────────────── main ──

def evaluate(dataset_dir: Path, emb_model: Path, rerank_model: Path | None, device: str = "cpu", top_k: int = 100):
    LOGGER = logging.getLogger("eval")
    LOGGER.info("Loading dataset from %s", dataset_dir)
    corpus = _read_jsonl(dataset_dir / "corpus.jsonl")
    queries = _read_jsonl(dataset_dir / "queries.jsonl")
    qrels = _read_qrels(dataset_dir / "qrels.tsv")

    emb = OnnxEncoder(emb_model, device)

    query_ids = list(queries.keys())
    doc_ids = list(corpus.keys())

    LOGGER.info("Encoding corpus (%d documents)…", len(doc_ids))
    doc_embeddings = emb.encode([corpus[did]["text"] for did in tqdm(doc_ids)])
    LOGGER.info("Encoding queries (%d)…", len(query_ids))
    query_embeddings = emb.encode([queries[qid]["text"] for qid in tqdm(query_ids)])

    # Retrieval (dot-product similarity)
    LOGGER.info("Retrieving top-%d docs per query…", top_k)
    ranked_lists: List[tuple[str, List[str]]] = []

    # Pre-compute doc norms for cosine similarity if needed
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    docs_unit = doc_embeddings / np.clip(doc_norms, a_min=1e-9, a_max=None)

    for qid, qvec in tqdm(zip(query_ids, query_embeddings), total=len(query_ids)):
        sims = (docs_unit @ (qvec / np.linalg.norm(qvec))).astype(np.float32)
        top_idx = np.argpartition(-sims, top_k)[:top_k]
        top_sorted = top_idx[np.argsort(-sims[top_idx])]
        ranked_lists.append((qid, [doc_ids[i] for i in top_sorted]))

    rec1 = recall_at_k(ranked_lists, qrels, 1)
    rec10 = recall_at_k(ranked_lists, qrels, 10)
    rec100 = recall_at_k(ranked_lists, qrels, 100)
    mrr10 = mrr_at_k(ranked_lists, qrels, 10)

    LOGGER.info("EMBEDDING MODEL RESULTS → Recall@1: %.4f  @10: %.4f  @100: %.4f  MRR@10: %.4f", rec1, rec10, rec100, mrr10)

    # ── Rerank ────────────────────────────────────────────
    if rerank_model is None:
        return

    reranker = OnnxReranker(rerank_model, device)
    LOGGER.info("Reranking top-%d…", top_k)

    reranked_lists: List[tuple[str, List[str]]] = []
    for qid, docs in tqdm(ranked_lists):
        texts_q = queries[qid]["text"]
        pairs = [(texts_q, corpus[did]["text"]) for did in docs]
        scores = reranker.score(pairs)
        best = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked_lists.append((qid, [d for d, _ in best]))

    rec1_rr = recall_at_k(reranked_lists, qrels, 1)
    rec10_rr = recall_at_k(reranked_lists, qrels, 10)
    mrr10_rr = mrr_at_k(reranked_lists, qrels, 10)

    LOGGER.info(
        "RERANKER RESULTS → Recall@1: %.4f (+%.4f)  Recall@10: %.4f (+%.4f)  MRR@10: %.4f (+%.4f)",
        rec1_rr,
        rec1_rr - rec1,
        rec10_rr,
        rec10_rr - rec10,
        mrr10_rr,
        mrr10_rr - mrr10,
    )


# ───────────────────────────────────────────── argparse ──

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate embedding & reranker ONNX models (Qwen3)")
    p.add_argument("--dataset", required=True, help="Path to dataset folder (corpus.jsonl / queries.jsonl / qrels.tsv)")
    p.add_argument("--embedding", required=True, help="Path to embedding model.onnx")
    p.add_argument("--reranker", help="Path to reranker model.onnx (optional)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device (default: cpu)")
    p.add_argument("--top-k", type=int, default=100, help="Retrieve top-k docs before reranking (default: 100)")
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO, stream=sys.stdout)
    args = _parse_args()
    evaluate(
        dataset_dir=Path(args.dataset),
        emb_model=Path(args.embedding),
        rerank_model=Path(args.reranker) if args.reranker else None,
        device=args.device,
        top_k=args.top_k,
    )