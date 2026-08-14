"""Guard check: the /books + DELETE /books/{book} endpoints tolerate a missing collection.

A fresh deploy has no `drag` collection until the first /ingest; scroll/delete then
raise UnexpectedResponse(404). Each handler must swallow that 404 so a fresh stack
returns [] (GET /books) / idempotent {"deleted": book} (DELETE) instead of HTTP 500.
Also asserts a non-404 (e.g. 500) still propagates.

Runnable as a script (matches tests/test_ingest.py); no pytest dependency — but needs
the project venv (imports httpx + qdrant_client from the runtime deps).
"""
import asyncio
import pathlib
import sys
from unittest.mock import AsyncMock, patch

import httpx
from qdrant_client.http.exceptions import UnexpectedResponse

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app import api  # noqa: E402


def _err(status_code: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status_code, reason_phrase="x", content=b"{}", headers=httpx.Headers({})
    )


def _fake_client(status_code: int) -> AsyncMock:
    client = AsyncMock()
    client.scroll = AsyncMock(side_effect=_err(status_code))
    client.delete = AsyncMock(side_effect=_err(status_code))
    client.close = AsyncMock()
    return client


def test_books_returns_empty_when_collection_missing():
    client = _fake_client(404)
    with patch("app.api.AsyncQdrantClient", return_value=client):
        result = asyncio.run(api.books_endpoint())
    assert result == [], "404 (missing collection) on /books must yield [], not raise"
    client.close.assert_awaited()


def test_delete_is_idempotent_when_collection_missing():
    client = _fake_client(404)
    with patch("app.api.AsyncQdrantClient", return_value=client):
        result = asyncio.run(api.delete_book("some-book"))
    assert result == {"deleted": "some-book"}, "404 on DELETE must be idempotent success, not raise"
    client.close.assert_awaited()


def test_non404_propagates():
    client = _fake_client(500)
    with patch("app.api.AsyncQdrantClient", return_value=client):
        try:
            asyncio.run(api.books_endpoint())
        except UnexpectedResponse as e:
            assert e.status_code == 500
            return
    raise AssertionError("non-404 UnexpectedResponse must propagate, not be swallowed")


def main():
    test_books_returns_empty_when_collection_missing()
    print("PASS: /books missing collection -> [] (no 500)")
    test_delete_is_idempotent_when_collection_missing()
    print("PASS: DELETE missing collection -> idempotent success (no 500)")
    test_non404_propagates()
    print("PASS: non-404 propagates")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
