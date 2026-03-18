"""Endee vector database client — single source of truth for all API calls."""

import json
import time
from typing import Optional

import msgpack
import numpy as np
import requests

from config import ENDEE_URL, INDEX_NAME, EMBEDDING_DIM, ENDEE_AUTH_TOKEN


class EndeeClient:
    def __init__(
        self,
        base_url: str = ENDEE_URL,
        index_name: str = INDEX_NAME,
        dim: int = EMBEDDING_DIM,
    ):
        self.base_url = base_url
        self.index_name = index_name
        self.dim = dim

    # ── helpers ──────────────────────────────────────────────

    def _headers(self, json_content: bool = False) -> dict:
        h = {}
        if ENDEE_AUTH_TOKEN:
            h["Authorization"] = ENDEE_AUTH_TOKEN
        if json_content:
            h["Content-Type"] = "application/json"
        return h

    # ── health ───────────────────────────────────────────────

    def is_healthy(self) -> bool:
        try:
            r = requests.get(
                f"{self.base_url}/api/v1/health",
                headers=self._headers(),
                timeout=3,
            )
            return r.ok
        except Exception:
            return False

    # ── index management ─────────────────────────────────────

    def create_index(self) -> bool:
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/index/create",
                json={
                    "index_name": self.index_name,
                    "dim": self.dim,
                    "space_type": "cosine",
                    "precision": "float32",
                    "M": 16,
                    "ef_con": 200,
                },
                headers=self._headers(json_content=True),
                timeout=10,
            )
            return r.ok or "already exists" in r.text.lower()
        except Exception:
            return False

    def delete_index(self) -> bool:
        try:
            r = requests.delete(
                f"{self.base_url}/api/v1/index/{self.index_name}/delete",
                headers=self._headers(),
                timeout=10,
            )
            return r.status_code in (200, 404)
        except Exception:
            return False

    def ensure_index(self) -> bool:
        return self.create_index()

    # ── insert ───────────────────────────────────────────────

    def insert_vectors(
        self,
        ids: list[str],
        vectors: np.ndarray,
        metadatas: list[dict],
    ) -> bool:
        """Insert vectors with metadata. Handles index-recovery on ephemeral hosts."""
        payload = []
        for vid, vec, meta in zip(ids, vectors, metadatas):
            payload.append(
                {
                    "id": str(vid),
                    "vector": vec.astype(np.float32).tolist(),
                    "meta": json.dumps(meta),
                }
            )

        try:
            r = requests.post(
                f"{self.base_url}/api/v1/index/{self.index_name}/vector/insert",
                json=payload,
                headers=self._headers(json_content=True),
                timeout=45,
            )
            if r.ok:
                return True

            # auto-recover from missing index files (Render free-tier restarts)
            if r.status_code == 400 and "required files missing" in r.text.lower():
                self.delete_index()
                if self.create_index():
                    retry = requests.post(
                        f"{self.base_url}/api/v1/index/{self.index_name}/vector/insert",
                        json=payload,
                        headers=self._headers(json_content=True),
                        timeout=45,
                    )
                    return retry.ok
            return False
        except Exception:
            return False

    # ── search ───────────────────────────────────────────────

    def search(
        self, vector: np.ndarray, top_k: int = 3
    ) -> list[dict]:
        """Search and return list of {"doc": ..., "text": ..., "score": ...}."""
        try:
            r = requests.post(
                f"{self.base_url}/api/v1/index/{self.index_name}/search",
                json={
                    "vector": vector.astype(np.float32).tolist(),
                    "k": top_k,
                },
                headers=self._headers(json_content=True),
                timeout=20,
            )
            if not r.ok:
                return []

            raw = self._decode_response(r)
            return self._normalize_results(raw)
        except Exception:
            return []

    def _decode_response(self, r: requests.Response):
        """Try msgpack first, then JSON."""
        content_type = r.headers.get("Content-Type", "")
        if "msgpack" in content_type or "octet-stream" in content_type:
            try:
                return msgpack.unpackb(r.content, raw=False)
            except Exception:
                pass
        try:
            return r.json()
        except Exception:
            pass
        # last resort: try msgpack on any content
        try:
            return msgpack.unpackb(r.content, raw=False)
        except Exception:
            return []

    @staticmethod
    def _normalize_results(raw) -> list[dict]:
        """Convert whatever Endee returns into a uniform list."""
        results = []

        if isinstance(raw, dict):
            raw = raw.get("results", raw.get("data", []))

        if not isinstance(raw, list):
            return results

        for item in raw:
            score = None
            meta_val = ""

            if isinstance(item, dict):
                meta_val = item.get("meta", item.get("metadata", ""))
                score = item.get("score", item.get("distance"))
            elif isinstance(item, (list, tuple)) and len(item) > 2:
                score = item[1]
                meta_val = item[2]
            else:
                continue

            doc, text = EndeeClient._parse_meta(meta_val)
            results.append({"doc": doc, "text": text, "score": score})

        return results

    @staticmethod
    def _parse_meta(meta_val) -> tuple[str, str]:
        if isinstance(meta_val, bytes):
            meta_val = meta_val.decode("utf-8", errors="replace")
        if isinstance(meta_val, dict):
            return str(meta_val.get("doc", "Document")), str(
                meta_val.get("text", "")
            )
        if isinstance(meta_val, str):
            try:
                parsed = json.loads(meta_val)
                if isinstance(parsed, dict):
                    return str(parsed.get("doc", "Document")), str(
                        parsed.get("text", "")
                    )
            except Exception:
                return "Document", meta_val
        return "Document", str(meta_val)