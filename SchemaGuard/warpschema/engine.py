"""
WarpSchema Transformation Engine
Copyright (c) 2026 Tarek Clarke. All Rights Reserved.

Uses BERT embeddings + cosine similarity to auto-map drifted schema keys
to a canonical target schema in real-time.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import TARGET_SCHEMAS, SchemaGuardConfig

logger = logging.getLogger("warpschema.engine")


class TransformationEngine:
    """Low-latency, BERT-powered schema remediation engine."""

    def __init__(self, config: SchemaGuardConfig | None = None) -> None:
        self.config = config or SchemaGuardConfig()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Loading model '%s' on %s", self.config.model_name, self._device
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModel.from_pretrained(self.config.model_name).to(
            self._device
        )
        self._model.eval()

        # Pre-compute embeddings for every canonical key in every schema.
        self._schema_embeddings: dict[str, dict[str, np.ndarray]] = {}
        for schema_name, schema_fields in TARGET_SCHEMAS.items():
            self._schema_embeddings[schema_name] = {
                key: self._embed(key) for key in schema_fields
            }

        if self.config.pre_warm:
            self._warm_up()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self,
        payload: dict[str, Any],
        schema_name: str,
    ) -> NormalizationResult:
        """Map *payload* keys to the canonical *schema_name* and return the
        remediated payload together with a detailed mapping report."""

        if schema_name not in self._schema_embeddings:
            raise ValueError(
                f"Unknown schema '{schema_name}'. "
                f"Available: {list(self._schema_embeddings)}"
            )

        canonical_embeddings = self._schema_embeddings[schema_name]
        canonical_keys = list(canonical_embeddings.keys())
        canonical_matrix = np.stack(
            [canonical_embeddings[k] for k in canonical_keys]
        )

        mapped_payload: dict[str, Any] = {}
        mapping_report: list[KeyMapping] = []
        unmapped_keys: dict[str, Any] = {}

        for incoming_key, value in payload.items():
            # Fast path: exact match — skip embedding entirely.
            if incoming_key in canonical_embeddings:
                mapped_payload[incoming_key] = value
                mapping_report.append(
                    KeyMapping(
                        source=incoming_key,
                        target=incoming_key,
                        similarity=1.0,
                        exact=True,
                    )
                )
                continue

            # Compute embedding for the incoming key and find best match.
            incoming_emb = self._embed(incoming_key)
            similarities = self._cosine_similarity_batch(
                incoming_emb, canonical_matrix
            )
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            best_key = canonical_keys[best_idx]

            if best_score >= self.config.similarity_threshold:
                mapped_payload[best_key] = value
                mapping_report.append(
                    KeyMapping(
                        source=incoming_key,
                        target=best_key,
                        similarity=best_score,
                        exact=False,
                    )
                )
                logger.info(
                    "Mapped '%s' -> '%s' (score=%.4f)",
                    incoming_key,
                    best_key,
                    best_score,
                )
            else:
                unmapped_keys[incoming_key] = value
                mapping_report.append(
                    KeyMapping(
                        source=incoming_key,
                        target=None,
                        similarity=best_score,
                        exact=False,
                    )
                )
                logger.warning(
                    "No match for '%s' (best='%s', score=%.4f, threshold=%.2f)",
                    incoming_key,
                    best_key,
                    best_score,
                    self.config.similarity_threshold,
                )

        if not self.config.drop_unmapped and unmapped_keys:
            mapped_payload["_unmapped"] = unmapped_keys

        return NormalizationResult(
            data=mapped_payload,
            mappings=mapping_report,
            unmapped_count=len(unmapped_keys),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @lru_cache(maxsize=2048)
    def _embed(self, text: str) -> np.ndarray:
        """Return the mean-pooled [CLS] embedding for *text*.

        Uses ``@lru_cache`` so repeated / canonical keys are never
        re-encoded."""
        # Normalise underscores & camelCase to space-separated words
        cleaned = self._clean_key(text)

        tokens = self._tokenizer(
            cleaned,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_token_length,
        ).to(self._device)

        with torch.no_grad():
            output = self._model(**tokens)

        # Mean-pool over the token dimension for a richer representation
        # than using [CLS] alone.
        embedding = (
            output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        )
        return embedding / (np.linalg.norm(embedding) + 1e-10)

    @staticmethod
    def _cosine_similarity_batch(
        vector: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        """Cosine similarity between a single *vector* and each row of *matrix*.

        Both inputs are assumed to be L2-normalised."""
        return matrix @ vector

    @staticmethod
    def _clean_key(key: str) -> str:
        """Convert ``engine_rpm`` or ``engineRpm`` into ``engine rpm``."""
        import re

        # camelCase -> space separated
        key = re.sub(r"([a-z])([A-Z])", r"\1 \2", key)
        # underscores / hyphens -> spaces
        key = key.replace("_", " ").replace("-", " ")
        return key.lower().strip()

    def _warm_up(self) -> None:
        """Run a throwaway inference to JIT-compile any lazy initialisations."""
        logger.info("Pre-warming model …")
        self._embed("warm_up_token")
        logger.info("Model ready.")


# ------------------------------------------------------------------
# Data classes for results
# ------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class KeyMapping:
    source: str
    target: str | None
    similarity: float
    exact: bool


@dataclass
class NormalizationResult:
    data: dict[str, Any]
    mappings: list[KeyMapping]
    unmapped_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "data": self.data,
            "mappings": [
                {
                    "source": m.source,
                    "target": m.target,
                    "similarity": round(m.similarity, 4),
                    "exact_match": m.exact,
                }
                for m in self.mappings
            ],
            "unmapped_count": self.unmapped_count,
        }
