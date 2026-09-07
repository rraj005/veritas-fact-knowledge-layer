from veritas.store.embeddings import FakeEmbedder
from veritas.store.vectors import VectorIndex


def test_add_and_query_excludes_via_where(tmp_path):
    idx = VectorIndex(tmp_path, FakeEmbedder())
    idx.add(
        ["f1", "f2"],
        ["revenue was 100 cr", "gdp grew 6 percent"],
        [{"doc_id": "a"}, {"doc_id": "b"}],
    )
    res = idx.query("revenue 100 crore", top_k=1, where={"doc_id": "b"})
    assert res and res[0][2]["doc_id"] == "b"


def test_count(tmp_path):
    idx = VectorIndex(tmp_path, FakeEmbedder())
    assert idx.count() == 0
    idx.add(["x1", "x2"], ["foo", "bar"], [{"doc_id": "a"}, {"doc_id": "a"}])
    assert idx.count() == 2


def test_query_returns_id_distance_metadata(tmp_path):
    idx = VectorIndex(tmp_path, FakeEmbedder())
    idx.add(["id1"], ["some fact text"], [{"doc_id": "doc_a", "page": 3}])
    results = idx.query("some fact", top_k=1)
    assert len(results) == 1
    item_id, distance, meta = results[0]
    assert item_id == "id1"
    assert isinstance(distance, float)
    assert meta["doc_id"] == "doc_a"


def test_fake_embedder_deterministic():
    emb = FakeEmbedder(dim=16)
    v1 = emb.embed(["hello world"])
    v2 = emb.embed(["hello world"])
    assert v1 == v2
    assert len(v1[0]) == 16


def test_fake_embedder_different_texts():
    emb = FakeEmbedder(dim=16)
    vecs = emb.embed(["text one", "text two"])
    assert vecs[0] != vecs[1]


def test_where_ne_filter(tmp_path):
    """Support operator-style where dicts like {'doc_id': {'$ne': 'a'}}."""
    idx = VectorIndex(tmp_path, FakeEmbedder())
    idx.add(
        ["f1", "f2"],
        ["revenue was 100", "profit was 50"],
        [{"doc_id": "a"}, {"doc_id": "b"}],
    )
    res = idx.query("revenue", top_k=5, where={"doc_id": {"$ne": "a"}})
    assert all(r[2]["doc_id"] != "a" for r in res)
