"""Guard check: search() returns [] (not raises) when the Qdrant collection is missing.

A fresh deploy has no `drag` collection until the first /ingest; query_points then
raises UnexpectedResponse(404). The guard in search() must swallow that 404 so the
chat path returns the existing "No relevant context found" instead of HTTP 500. Also
asserts a non-404 (e.g. 500) still propagates, so real Qdrant errors aren't masked.

Runnable as a script (matches tests/test_ingest.py); stdlib only — no pytest dep.
"""
import asyncio
import pathlib
import sys
from unittest.mock import AsyncMock, patch

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import query  # noqa: E402


def _err(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code, reason_phrase="x", content=b"{}", headers=httpx.Headers({})
    )


def _patched_client(status_code: int) -> AsyncMock:
    client = AsyncMock()
    client.query_points = AsyncMock(side_effect=_err(status_code))
    client.close = AsyncMock()
    return client


def test_missing_collection_returns_empty():
    client = _patched_client(404)
    with patch("app.query._client", new=AsyncMock(return_value=client)), patch(
        "app.ingest.embed_texts", new=AsyncMock(return_value=[[0.0] * 1024])
    ):
        result = asyncio.run(query.search("how do grapples work"))
    assert result == [], "404 (missing collection) must yield [], not raise"
    client.close.assert_awaited()


def test_non404_propagates():
    client = _patched_client(500)
    with patch("app.query._client", new=AsyncMock(return_value=client)), patch(
        "app.ingest.embed_texts", new=AsyncMock(return_value=[[0.0] * 1024])
    ):
        try:
            asyncio.run(query.search("q"))
        except UnexpectedResponse as e:
            assert e.status_code == 500
            return
    raise AssertionError("non-404 UnexpectedResponse must propagate, not be swallowed")


def main():
    test_missing_collection_returns_empty()
    print("PASS: missing collection -> [] (no 500)")
    test_non404_propagates()
    print("PASS: non-404 propagates")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
