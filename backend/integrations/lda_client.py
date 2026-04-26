"""
LDA Legal Data Hub Integration
================================
Connects to the Legal Data Hub API (https://docs.legal-data-analytics.com)
for German legal search, semantic search, QnA, and clause checking.

Requires: LDA_CLIENT_ID and LDA_CLIENT_SECRET in .env
API Base: https://api.legal-data-hub.com (or https://otto-schmidt.legal-data-hub.com)
"""

import os
import json
import time
import requests
from typing import Optional


class LDAClient:
    """Client for the Legal Data Hub API by LDA Legal Data Analytics GmbH."""

    TOKEN_URL = "https://online.otto-schmidt.de/token"
    API_BASE = "https://api.legal-data-hub.com"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.client_id = client_id or os.getenv("LDA_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("LDA_CLIENT_SECRET", "")
        self.api_base = api_base or os.getenv("LDA_API_BASE", self.API_BASE)
        self._token = None
        self._token_expiry = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def _get_token(self) -> str:
        """Retrieve or refresh the Bearer token."""
        if self._token and time.time() < self._token_expiry:
            return self._token

        if not self.is_configured:
            raise ValueError(
                "LDA_CLIENT_ID and LDA_CLIENT_SECRET must be set in .env"
            )

        resp = requests.post(
            self.TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "authorization_code",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        self._token = data.get("access_token", "")
        # Token typically valid for ~3600s, refresh 60s early
        self._token_expiry = time.time() + data.get("expires_in", 3600) - 60
        return self._token

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_token()}",
        }

    # ─── Search (keyword / Elasticsearch DSL) ───
    def search(
        self,
        query: str,
        data_asset: str = "Beratermodul Miet- und WEG-Recht",
        size: int = 10,
    ) -> dict:
        """Keyword search across a data asset."""
        payload = {
            "size": size,
            "_source": {
                "includes": [
                    "metadata.aktenzeichen",
                    "metadata.datum",
                    "metadata.dokumententyp",
                    "metadata.ecli",
                    "metadata.ebene0",
                    "metadata.ebene1",
                    "metadata.leitsatz",
                    "metadata.normenkette",
                    "metadata.oso_url",
                    "text",
                ]
            },
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "metadata.aktenzeichen^3",
                                    "metadata.dokumententyp^1",
                                    "metadata.leitsatz^1",
                                    "metadata.ebene0^1",
                                    "text^1",
                                ],
                                "type": "best_fields",
                                "operator": "and",
                            }
                        },
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "metadata.leitsatz^2",
                                    "text^1",
                                ],
                                "type": "phrase",
                            }
                        },
                    ]
                }
            },
            "highlight": {"fragment_size": 300, "fields": {"text": {}}},
            "from": 0,
            "sort": [{"_score": "desc"}],
        }

        resp = requests.post(
            f"{self.api_base}/api/search/{requests.utils.quote(data_asset)}/_search",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ─── Semantic Search ───
    def semantic_search(
        self,
        query: str,
        data_asset: str = "Aktionsmodul Familienrecht",
        candidates: int = 5,
        filters: Optional[list] = None,
    ) -> dict:
        """AI-powered semantic search across a data asset."""
        payload = {
            "candidates": candidates,
            "data_asset": data_asset,
            "filter": filters or [{}],
            "search_query": query,
        }

        resp = requests.post(
            f"{self.api_base}/api/semantic-search",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    # ─── QnA (Question and Answer) ───
    def qna(
        self,
        question: str,
        data_asset: str = "Beratermodul Miet- und WEG-Recht",
        mode: str = "attribution",
        filters: Optional[list] = None,
    ) -> dict:
        """Ask a legal question and get an AI-generated answer with sources."""
        payload = {
            "data_asset": data_asset,
            "filter": filters or [{}],
            "mode": mode,
            "prompt": question,
        }

        resp = requests.post(
            f"{self.api_base}/api/qna",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    # ─── Clause Check ───
    def clause_check(
        self,
        clause_text: str,
        data_asset: str = "Aktionsmodul Arbeitsrecht",
        mode: str = "check",
        filters: Optional[list] = None,
    ) -> dict:
        """Check a contract clause for validity and appropriateness."""
        payload = {
            "data_asset": data_asset,
            "prompt": clause_text,
            "mode": mode,
            "filter": filters or [],
        }

        resp = requests.post(
            f"{self.api_base}/api/analyzer/clause-check",
            headers=self._headers(),
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    # ─── List Data Assets ───
    def list_data_assets(self) -> dict:
        """List all available data assets."""
        resp = requests.get(
            f"{self.api_base}/api/data-assets",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
