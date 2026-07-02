import json

from ultra.taskcraft_source import build_taskcraft_readiness_audit, build_taskcraft_source_probe


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_taskcraft_source_probe_filters_text_only_multihop_candidates(tmp_path):
    dataset_dir = tmp_path / "taskcraft"
    _write_jsonl(
        dataset_dir / "pure_qa.jsonl",
        [
            {
                "query": "Need three-hop PDF answer",
                "golden_answer": "Alpha",
                "valid_hop": 3,
                "tool": ["pdf_tool"],
                "domain": "financial_report_pdftool",
            },
            {
                "query": "Shallow web answer",
                "golden_answer": "Beta",
                "valid_hop": 1,
                "tool": ["web_tool"],
                "domain": "academic_html_webtool",
            },
            {
                "query": "Image answer",
                "golden_answer": "Gamma",
                "valid_hop": 5,
                "tool": ["image_tool"],
                "domain": "financial_report_imagetool",
            },
        ],
    )
    _write_jsonl(
        dataset_dir / "multihop_subtask_trace.jsonl",
        [
            {
                "domain": "academic_html_webtool",
                "trace": [
                    {"query": "Atomic lookup", "golden_answer": "Doc", "valid_hop": 1},
                    {"query": "Hard merged lookup", "golden_answer": "Delta", "valid_hop": 4},
                ],
            },
            {
                "domain": "biology_paper_imagetool",
                "trace": [
                    {"query": "Visual lookup", "golden_answer": "Epsilon", "valid_hop": 4},
                ],
            },
        ],
    )
    _write_jsonl(
        dataset_dir / "atomic_trace.jsonl",
        [
            {
                "query": "Trace row",
                "golden_answer": "Zeta",
                "tool_call_names": ["web_search", "crawl_pages"],
                "domain": "academic_html_webtool",
            }
        ],
    )

    report = build_taskcraft_source_probe(
        dataset_dir=dataset_dir,
        candidates_out=tmp_path / "out" / "candidates.jsonl",
        report_out=tmp_path / "out" / "report.json",
        limit=10,
    )

    candidates = [
        json.loads(line)
        for line in (tmp_path / "out" / "candidates.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert report["status"] == "candidate_source_not_grpo_ready"
    assert report["summaries"]["pure_qa"]["rows"] == 3
    assert report["summaries"]["pure_qa"]["image_domain_rows"] == 1
    assert report["candidate_count"] == 2
    assert {row["origin_file"] for row in candidates} == {"pure_qa.jsonl", "multihop_subtask_trace.jsonl"}
    assert all(row["grpo_ready"] is False for row in candidates)
    assert all(row["source_freeze_required"] is True for row in candidates)
    assert all("source_documents_not_frozen" in row["readiness_blockers"] for row in candidates)


def test_taskcraft_source_probe_stratifies_limit(tmp_path):
    dataset_dir = tmp_path / "taskcraft"
    _write_jsonl(
        dataset_dir / "pure_qa.jsonl",
        [
            {
                "query": f"Financial {idx}",
                "golden_answer": str(idx),
                "valid_hop": 3,
                "tool": ["pdf_tool"],
                "domain": "financial_report_pdftool",
            }
            for idx in range(4)
        ]
        + [
            {
                "query": "Academic",
                "golden_answer": "A",
                "valid_hop": 6,
                "tool": ["web_tool"],
                "domain": "academic_html_webtool",
            }
        ],
    )
    _write_jsonl(dataset_dir / "multihop_subtask_trace.jsonl", [])
    _write_jsonl(dataset_dir / "atomic_trace.jsonl", [])

    report = build_taskcraft_source_probe(
        dataset_dir=dataset_dir,
        candidates_out=tmp_path / "candidates.jsonl",
        report_out=tmp_path / "report.json",
        limit=2,
    )

    candidates = [json.loads(line) for line in (tmp_path / "candidates.jsonl").read_text().splitlines()]
    assert report["candidate_count_before_limit"] == 5
    assert report["candidate_count"] == 2
    assert {row["domain"] for row in candidates} == {"financial_report_pdftool", "academic_html_webtool"}


def test_taskcraft_readiness_audit_links_atomic_evidence_but_blocks_grpo(tmp_path):
    dataset_dir = tmp_path / "taskcraft"
    candidate_query = "Which report gives the answer?"
    atomic_query = "According to the source, what is the answer?"
    _write_jsonl(
        dataset_dir / "atomic_trace.jsonl",
        [
            {
                "query": atomic_query,
                "golden_answer": "Alpha",
                "content_identifier": "Stable Source",
                "tool_call_names": ["web_search", "crawl_pages"],
                "tool_calls_count": 2,
                "ans_from_agent": repr(
                    {
                        "answer": "Alpha",
                        "trace": {
                            "actions": [
                                {
                                    "tool_name": "crawl_pages",
                                    "observation": "Fetched https://example.test/source. The answer is Alpha.",
                                }
                            ]
                        },
                    }
                ),
                "score": {"agent_answer_score": 1.0},
            }
        ],
    )
    _write_jsonl(
        tmp_path / "candidates.jsonl",
        [
            {
                "candidate_id": "taskcraft_multihop_000001",
                "origin_file": "multihop_subtask_trace.jsonl",
                "domain": "academic_html_webtool",
                "valid_hop": 3,
                "query": candidate_query,
                "golden_answer": "Alpha",
                "subtasks": [
                    {"query": atomic_query, "golden_answer": "Alpha", "valid_hop": 1},
                    {"query": candidate_query, "golden_answer": "Alpha", "valid_hop": 3},
                ],
            }
        ],
    )

    report = build_taskcraft_readiness_audit(
        dataset_dir=dataset_dir,
        candidates_jsonl=tmp_path / "candidates.jsonl",
        evidence_out=tmp_path / "evidence.jsonl",
        report_out=tmp_path / "audit.json",
    )

    evidence = [json.loads(line) for line in (tmp_path / "evidence.jsonl").read_text().splitlines()]
    assert report["status"] == "audited_not_grpo_ready"
    assert report["raw_dataset_grpo_ready"] is False
    assert report["linkage_counts"]["any_atomic_match"] == 1
    assert report["linkage_counts"]["any_crawl_observation"] == 1
    assert report["freeze_priority_count"] == 1
    assert report["decision"]["promote_to_grpo"] is False
    assert evidence[0]["grpo_ready"] is False
    assert "source_documents_not_frozen" in evidence[0]["readiness_blockers"]
    assert "final_query_not_atomic_trace_anchored" in evidence[0]["readiness_blockers"]
