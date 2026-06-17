import pytest

from rqa.graph import QuestionGraph


@pytest.fixture
def graph(tmp_path):
    g = QuestionGraph(tmp_path / "graph.db")
    yield g
    g.close()


def test_add_and_search(graph):
    graph.add_node("claim", "自己理解層の更新は安定性を高める", provenance="user")
    graph.add_node("question", "更新可能領域と禁止領域はどのデータ構造で分離すべきか?", provenance="self")
    graph.add_node("note", "completely unrelated text about cooking pasta", provenance="user")
    results = graph.search("自己理解層の更新と安定性", top_k=2)
    assert results
    assert results[0].text.startswith("自己理解層")


def test_status_transitions_enforced(graph):
    nid = graph.add_node("question", "未解決の問い?", provenance="self")
    graph.set_status(nid, "revisited")
    graph.set_status(nid, "resolved")
    with pytest.raises(ValueError):
        graph.set_status(nid, "open")  # no resurrection


def test_archived_excluded_from_search(graph):
    nid = graph.add_node("claim", "古い主張テキストアーカイブ対象", provenance="user")
    assert graph.search("古い主張テキストアーカイブ対象")
    graph.set_status(nid, "archived")
    assert not graph.search("古い主張テキストアーカイブ対象")


def test_append_only_no_delete_api(graph):
    assert not hasattr(graph, "delete_node")
    assert not hasattr(graph, "remove_node")


def test_audit_log_records_operations(graph):
    nid = graph.add_node("claim", "監査対象", provenance="user")
    graph.set_status(nid, "archived")
    stats = graph.stats()
    assert stats["audit_entries"] == 2


def test_invalid_kind_rejected(graph):
    with pytest.raises(ValueError):
        graph.add_node("policy", "should fail", provenance="self")
