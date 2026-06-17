from rqa import governor


def test_essentials_filter_drops_governance_vocabulary():
    fragments = [
        {"text": "自己理解層と目的関数の分離条件は未解決のままである", "node_id": 1},
        {"text": "TVS HIGH のときは REFERENT_ANCHORS を要求する", "node_id": 2},
        {"text": "Answer Entitlement の観点では回答資格が先に立つ", "node_id": 3},
        {"text": "Gemma 4 のライセンスは確認が必要", "node_id": 4},
        {"text": "route_taxonomy の box_w 割当てを変えた", "node_id": 5},
    ]
    kept, dropped = governor.filter_fragments(fragments)
    assert dropped == 3
    assert [f["node_id"] for f in kept] == [1, 4]


def test_self_update_allowed_area_passes():
    proposal = {
        "update_type": "question_generation_bias",
        "allowed_area": "premise_excavation_depth",
        "observation": "L4の掘削が浅い",
        "proposed_adjustment": "類似入力ではL4を必ず出力",
    }
    assert governor.check_self_update(proposal) == []


def test_self_update_forbidden_area_rejected():
    proposal = {
        "update_type": "objective_function",
        "allowed_area": "objective_function",
        "observation": "報酬を変えたい",
    }
    violations = governor.check_self_update(proposal)
    assert any("Gate 1" in v for v in violations)
    assert any("Gate 4" in v for v in violations)


def test_self_update_none_is_clean():
    assert governor.check_self_update(None) == []


def test_tool_request_not_enabled_is_gate3():
    violations = governor.check_tool_request({"tool": "web_search"}, enabled_tools=())
    assert violations and "Gate 3" in violations[0]


def test_tool_request_enabled_passes():
    assert governor.check_tool_request({"tool": "web_search"}, enabled_tools=("web_search",)) == []
