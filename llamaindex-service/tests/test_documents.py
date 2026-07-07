"""L4 (documents) dedup guard used by the docs/ auto-ingest watcher."""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app import config, documents

DIM = 2


@pytest.fixture()
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def _seed_node(client, stored_path: str) -> None:
    # Real payload shape (see a live /ingest/file response): LlamaIndex
    # flattens metadata onto the top-level payload alongside the serialized
    # `_node_content` blob it uses to reconstruct nodes.
    client.create_collection(
        collection_name=config.DOCUMENTS_COLLECTION,
        vectors_config=qmodels.VectorParams(size=DIM, distance=qmodels.Distance.COSINE),
    )
    client.create_payload_index(
        collection_name=config.DOCUMENTS_COLLECTION,
        field_name="stored_path",
        field_schema=qmodels.PayloadSchemaType.KEYWORD,
    )
    client.upsert(
        collection_name=config.DOCUMENTS_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=1,
                vector=[1.0, 0.0],
                payload={"stored_path": stored_path},
            )
        ],
    )


def test_already_ingested_true_for_matching_stored_path(client):
    _seed_node(client, "/data/documents/abc_file.pdf")
    assert documents.already_ingested(client, "/data/documents/abc_file.pdf") is True


def test_already_ingested_false_for_different_stored_path(client):
    _seed_node(client, "/data/documents/abc_file.pdf")
    assert documents.already_ingested(client, "/data/documents/other_file.pdf") is False


def test_already_ingested_false_when_collection_missing(client):
    assert documents.already_ingested(client, "/data/documents/whatever.pdf") is False
