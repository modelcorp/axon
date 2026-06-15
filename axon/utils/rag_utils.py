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
"""Retrieval-Augmented Generation (RAG) utilities."""

import torch
from sentence_transformers import SentenceTransformer, util


class RAG:
    """
    A simple RAG system using sentence transformers for embedding-based retrieval.
    """

    def __init__(self, docs: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the RAG system.

        Args:
            docs (List[str]): A list of documents to encode.
            model (str): The SentenceTransformer model to use.
        """
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model)
        self.docs = docs
        # Compute embeddings
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def top_k(self, query, k=1):
        """
        Retrieve top-k most similar documents for a query.

        Args:
            query (str): The query text.
            k (int): Number of top documents to retrieve.

        Returns:
            List[dict]: List of dicts with keys 'score', 'text', and 'idx'.
        """
        # Create embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity [1 x N]
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Extract top_k indices
        top_results = torch.topk(cos_scores, k=k)

        # Prepare a list of (score, problem_text)
        results = []
        for score, idx in zip(top_results.values, top_results.indices, strict=False):
            results.append(
                {
                    "score": score,
                    "text": self.docs[int(idx)],
                    "idx": int(idx),
                }
            )
        return results


__all__ = ["RAG"]
