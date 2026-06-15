# Copyright 2025 Model AI Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modeled on the Search-R1 retrieval server (github.com/PeterGriffinJin/Search-R1), Apache-2.0.
"""
Lightweight retrieval server for Search-R1 style question answering.

Supports:
- BM25 retrieval via Pyserini
- Dense retrieval via FAISS and e5 embeddings
- FastAPI endpoint compatible with Search-R1 format
"""

import argparse
import json
import logging
import warnings

import datasets
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str):
    """Load corpus from jsonl file."""
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)  # nosec B615
    return corpus


def load_docs(corpus, doc_idxs):
    """Load documents by indices from corpus."""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


class BM25Retriever:
    """BM25 retriever using Pyserini."""

    def __init__(self, index_path: str, corpus_path: str = None, topk: int = 10):
        from pyserini.search.lucene import LuceneSearcher

        self.index_path = index_path
        self.corpus_path = corpus_path
        self.topk = topk

        logger.info(f"Loading BM25 index from {index_path}")
        self.searcher = LuceneSearcher(index_path)

        # Check if index contains documents or if we need external corpus
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            if corpus_path is None:
                raise ValueError("Corpus path required when index doesn't contain documents")
            logger.info(f"Loading external corpus from {corpus_path}")
            self.corpus = load_corpus(corpus_path)

        logger.info("BM25 retriever initialized")

    def _check_contain_doc(self):
        """Check if index contains full documents."""
        try:
            doc = self.searcher.doc(0)
            if doc is None:
                return False
            raw = doc.raw()
            return raw is not None and len(raw) > 0
        except Exception as e:
            logger.warning(f"Could not check if index contains documents: {e}")
            return False

    def search(self, query: str, num: int = None, return_score: bool = False):
        """Search for a single query."""
        if num is None:
            num = self.topk

        hits = self.searcher.search(query, num)

        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []

        scores = [hit.score for hit in hits]

        if len(hits) < num:
            warnings.warn(f"Only {len(hits)} documents retrieved for query: {query[:50]}...", stacklevel=2)
        else:
            hits = hits[:num]

        # Extract documents
        if self.contain_doc:
            # Documents are in the index
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            # Load from external corpus
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        """Search for multiple queries."""
        if num is None:
            num = self.topk

        # Use Pyserini's native batch search
        qids = [str(i) for i in range(len(query_list))]
        hits_dict = self.searcher.batch_search(query_list, qids, k=num, threads=8)

        results = []
        scores = []

        for qid in qids:
            hits = hits_dict.get(qid, [])

            if len(hits) < 1:
                results.append([])
                scores.append([])
                continue

            item_scores = [hit.score for hit in hits]

            if self.contain_doc:
                all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
                item_results = [
                    {
                        "title": content.split("\n")[0].strip('"'),
                        "text": "\n".join(content.split("\n")[1:]),
                        "contents": content,
                    }
                    for content in all_contents
                ]
            else:
                item_results = load_docs(self.corpus, [hit.docid for hit in hits])

            results.append(item_results)
            scores.append(item_scores)

        if return_score:
            return results, scores
        else:
            return results


class E5Retriever:
    """Dense retriever using E5 embeddings and FAISS index."""

    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        retriever_model: str = "intfloat/e5-base-v2",
        topk: int = 10,
        faiss_gpu: bool = False,
        batch_size: int = 64,
    ):
        import faiss
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.index_path = index_path
        self.corpus_path = corpus_path
        self.topk = topk
        self.faiss_gpu = faiss_gpu
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() and faiss_gpu else "cpu"

        logger.info(f"Loading E5 model from {retriever_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(retriever_model)  # nosec B615
        self.model = AutoModel.from_pretrained(retriever_model)  # nosec B615
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(index_path)

        # Move index to GPU if requested
        if faiss_gpu and torch.cuda.is_available():
            logger.info("Moving FAISS index to GPU")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        logger.info(f"Loading corpus from {corpus_path}")
        self.corpus = load_corpus(corpus_path)

        logger.info(f"E5 retriever initialized with {self.index.ntotal} vectors on {self.device}")

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to embedding vector."""
        import torch

        # E5 requires "query: " prefix for queries
        query_text = f"query: {query}"

        # Tokenize and encode
        inputs = self.tokenizer(query_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode multiple queries to embedding vectors."""
        import torch

        all_embeddings = []

        # Process in batches
        for i in range(0, len(queries), self.batch_size):
            batch_queries = queries[i : i + self.batch_size]
            # E5 requires "query: " prefix for queries
            batch_texts = [f"query: {q}" for q in batch_queries]

            # Tokenize and encode
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask."""
        import torch

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def search(self, query: str, num: int = None, return_score: bool = False):
        """Search for a single query."""
        if num is None:
            num = self.topk

        # Encode query
        query_embedding = self.encode_query(query)
        query_embedding = query_embedding.reshape(1, -1).astype("float32")

        # Search
        scores, doc_idxs = self.index.search(query_embedding, num)
        scores = scores[0]
        doc_idxs = doc_idxs[0]

        # Filter out invalid indices (-1 means no result)
        valid_mask = doc_idxs >= 0
        doc_idxs = doc_idxs[valid_mask]
        scores = scores[valid_mask]

        if len(doc_idxs) < 1:
            if return_score:
                return [], []
            else:
                return []

        if len(doc_idxs) < num:
            warnings.warn(f"Only {len(doc_idxs)} documents retrieved for query: {query[:50]}...", stacklevel=2)

        # Load documents from corpus
        results = load_docs(self.corpus, doc_idxs)

        if return_score:
            return results, scores.tolist()
        else:
            return results

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        """Search for multiple queries."""
        if num is None:
            num = self.topk

        # Encode all queries
        query_embeddings = self.encode_queries(query_list)
        query_embeddings = query_embeddings.astype("float32")

        # Batch search
        all_scores, all_doc_idxs = self.index.search(query_embeddings, num)

        results = []
        scores = []

        for i, (doc_idxs, doc_scores) in enumerate(zip(all_doc_idxs, all_scores, strict=False)):
            # Filter out invalid indices
            valid_mask = doc_idxs >= 0
            doc_idxs = doc_idxs[valid_mask]
            doc_scores = doc_scores[valid_mask]

            if len(doc_idxs) < 1:
                results.append([])
                scores.append([])
                continue

            if len(doc_idxs) < num:
                warnings.warn(
                    f"Only {len(doc_idxs)} documents retrieved for query {i}: {query_list[i][:50]}...", stacklevel=2
                )

            # Load documents from corpus
            query_results = load_docs(self.corpus, doc_idxs)
            results.append(query_results)
            scores.append(doc_scores.tolist())

        if return_score:
            return results, scores
        else:
            return results


class QueryRequest(BaseModel):
    """Request format for retrieval endpoint."""

    queries: list[str]
    topk: int | None = None
    return_scores: bool = False


def create_app(retriever, default_topk: int = 3):
    """Create FastAPI application with retriever."""
    app = FastAPI(title="Axon Retrieval Server")

    @app.get("/health")
    def health_check():
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post("/retrieve")
    def retrieve_endpoint(request: QueryRequest):
        """
        Retrieval endpoint compatible with Search-R1 format.

        Input:
        {
            "queries": ["What is Python?", "Tell me about AI."],
            "topk": 3,
            "return_scores": true
        }

        Output:
        {
            "result": [
                [
                    {"document": {...}, "score": 10.5},
                    {"document": {...}, "score": 9.2},
                    ...
                ],
                ...
            ]
        }
        """
        topk = request.topk if request.topk else default_topk

        # Perform batch retrieval
        if request.return_scores:
            results, scores = retriever.batch_search(query_list=request.queries, num=topk, return_score=True)
        else:
            results = retriever.batch_search(query_list=request.queries, num=topk, return_score=False)
            scores = None

        # Format response
        resp = []
        for i, single_result in enumerate(results):
            if request.return_scores and scores:
                # Combine documents with scores
                combined = []
                for doc, score in zip(single_result, scores[i], strict=False):
                    combined.append({"document": doc, "score": score})
                resp.append(combined)
            else:
                # Just documents, wrap each in "document" key for consistency
                resp.append([{"document": doc} for doc in single_result])

        return {"result": resp}

    return app


def main():
    parser = argparse.ArgumentParser(description="Axon Retrieval Server")
    parser.add_argument(
        "--index_path", type=str, required=True, help="Path to index directory (BM25) or index file (e5)"
    )
    parser.add_argument("--corpus_path", type=str, default=None, help="Path to corpus jsonl file")
    parser.add_argument(
        "--retriever_name",
        type=str,
        default="bm25",
        choices=["bm25", "e5"],
        help="Type of retriever to use (bm25 or e5)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Model name for e5 retriever (e.g., intfloat/e5-base-v2)",
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for FAISS index (e5 only)")
    parser.add_argument("--topk", type=int, default=3, help="Default number of documents to retrieve")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for encoding queries (e5 only)")

    args = parser.parse_args()

    # Initialize retriever based on type
    if args.retriever_name == "bm25":
        logger.info("Initializing BM25 retriever")
        retriever = BM25Retriever(index_path=args.index_path, corpus_path=args.corpus_path, topk=args.topk)
    elif args.retriever_name == "e5":
        logger.info("Initializing E5 dense retriever")
        if args.corpus_path is None:
            raise ValueError("--corpus_path is required for e5 retriever")
        retriever = E5Retriever(
            index_path=args.index_path,
            corpus_path=args.corpus_path,
            retriever_model=args.retriever_model,
            topk=args.topk,
            faiss_gpu=args.faiss_gpu,
            batch_size=args.batch_size,
        )
    else:
        raise ValueError(f"Unknown retriever type: {args.retriever_name}")

    # Create and launch app
    app = create_app(retriever, default_topk=args.topk)

    logger.info(f"Starting {args.retriever_name} retrieval server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
