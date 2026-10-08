"""Boundary tests for offline series assembly, identity and source clocks."""
from copy import deepcopy
import math

import pytest

from noisefloor.adapters import from_financial_evidence, from_records


def records(**kwargs):
    base = {"series_id": "demo-bank-rate", "kind": "rate", "unit": "%",
            "source": {"id": "demo-publisher", "rights": "permitted"},
            "max_age_seconds": 86400}
    base.update(kwargs)
    return base


def row(**kwargs):
    result = {"timestamp": "2026-10-01T00:00:00Z", "value": 4.5,
              "entity_id": "BANK-A", "metric": "rate"}
    result.update(kwargs)
    return result


def fe_row(**kwargs):
    result = {"dataset": "money_market_history", "product": "Seiche",
              "entity_id": "US-USD", "entity_name": "Synthetic USD benchmark",
              "metric": "DEMO_RATE", "unit": "%", "value": 4.5,
              "as_of": "2026-10-01", "knowledge_time": "2026-10-02T12:00:00Z",
              "published_at": "2026-10-02T11:00:00Z", "retrieved_at": "2026-10-03T00:00:00Z",
              "source_url": "https://example.invalid/benchmarks",
              "source_field": "/markets/0/benchmark/history/0/1",
              "content_sha256": "sha256:synthetic-not-real-evidence",
              "rights_status": "allowed", "source_status": "FRESH",
              "availability": "AVAILABLE", "transport_status": "complete",
              "evidence_status": "not_evaluated", "carrier_verification": "not_performed"}
    result.update(kwargs)
    return result


def table(*rows, **kwargs):
    result = {"schema": "liquidity-lab.openbb-table.v1", "results": list(rows),
              "total_rows": len(rows), "returned_rows": len(rows), "offset": 0, "next_offset": None,
              "transport_status": "complete"}
    result.update(kwargs)
    return result


def fe(input_table, **kwargs):
    return from_financial_evidence(input_table, kind="rate", max_age_seconds=86400,
                                   allow_date_only=True, **kwargs)


def test_missing_values_remain_missing_with_input_retained():
    missing = row()
    missing.pop("value")
    original = [missing, row(timestamp="2026-10-02T00:00:00Z", value=None)]
    result = from_records(original, **records())
    assert [point["value"] for point in result["series"][0]["observations"]] == [None, None]
    assert result["issues"][0]["code"] == "missing_value_preserved_as_null"
    assert result["provenance"]["records"] == original
    result["provenance"]["records"][0]["value"] = 100
    assert "value" not in original[0]


@pytest.mark.parametrize("field,value", [("entity_id", "BANK-B"), ("metric", "other_rate"), ("symbol", "XYZ")])
def test_mixed_institutions_metrics_or_instruments_are_not_silently_combined(field, value):
    with pytest.raises(ValueError, match="mixed"):
        from_records([row(), row(timestamp="2026-10-02T00:00:00Z", **{field: value})], **records())


@pytest.mark.parametrize("unit", ["basis_points", None, "USD"])
def test_conflicting_units_need_explicit_conversion(unit):
    with pytest.raises(ValueError, match="unit conflicts"):
        from_records([row(unit=unit)], **records())


@pytest.mark.parametrize("source", ["other-publisher", {"id": "other-publisher"},
                                    {"id": "demo-publisher", "url": "https://other.invalid"}])
def test_mixed_sources_are_rejected(source):
    with pytest.raises(ValueError, match="sources|source URL"):
        from_records([row(source=source)], **records())


def test_rights_can_only_be_downgraded_by_rows_and_original_declaration_is_retained():
    arguments = records()
    original = deepcopy(arguments)
    result = from_records([row(rights="restricted")], **arguments)
    assert result["series"][0]["source"]["rights"] == "restricted"
    assert result["provenance"]["source_declaration"]["rights"] == "permitted"
    assert arguments == original
    assert any(issue["code"] == "rights_downgraded" for issue in result["issues"])


def test_date_only_requires_opt_in_and_does_not_create_knowledge_time():
    with pytest.raises(ValueError, match="allow_date_only"):
        from_records([row(timestamp="2026-10-01")], **records())
    result = from_records([row(timestamp="2026-10-01")], **records(allow_date_only=True))
    assert result["series"][0]["observations"] == [{"observed_at": "2026-10-01T00:00:00Z", "value": 4.5}]
    assert result["issues"][0]["code"] == "date_only_normalized"


def test_date_only_availability_is_never_assumed_at_midnight():
    with pytest.raises(ValueError, match="available_at"):
        from_records([row(available_at="2026-10-02")], **records(allow_date_only=True))


def test_sorting_preserves_fractional_second_order_and_records_that_it_sorted():
    result = from_records([row(timestamp="2026-10-01T00:00:00.2Z", value=2),
                           row(timestamp="2026-10-01T00:00:00Z", value=1)], **records())
    assert [point["value"] for point in result["series"][0]["observations"]] == [1, 2]
    assert any(issue["code"] == "sorted_by_observation_time" for issue in result["issues"])


def test_revisions_require_explicit_selection_even_if_timestamp_strings_differ():
    with pytest.raises(ValueError, match="select revisions explicitly"):
        from_records([row(value=4.5, revision=1),
                      row(timestamp="2026-10-01T00:00:00.000+00:00", value=4.6, revision=2)], **records())


def test_custom_mapping_is_explicit_and_preserves_unmapped_metadata():
    original = [{"date": "2026-10-01T00:00:00Z", "close": 7, "vendor_note": "synthetic"}]
    result = from_records(original, **records(mapping={"timestamp": "date", "value": "close"}))
    assert result["series"][0]["observations"][0]["value"] == 7
    assert result["provenance"]["records"] == original
    with pytest.raises(ValueError, match="one column"):
        from_records(original, **records(mapping={"timestamp": "close", "value": "close"}))


@pytest.mark.parametrize("value", [True, math.nan, math.inf, "4.5"])
def test_invalid_numbers_do_not_become_observations(value):
    with pytest.raises(ValueError):
        from_records([row(value=value)], **records())


def test_financial_evidence_retains_source_and_evidence_clocks_without_substitution():
    input_table = table(fe_row())
    result = fe(input_table)
    point = result["series"][0]["observations"][0]
    assert point["observed_at"] == "2026-10-01T00:00:00Z"
    assert point["available_at"] == "2026-10-02T12:00:00Z"
    assert result["provenance"]["input"] == input_table
    assert result["series"][0]["source"]["rights"] == "permitted"


def test_financial_evidence_unknown_availability_stays_unknown_despite_publication_and_retrieval():
    result = fe(table(fe_row(knowledge_time=None)))
    assert "available_at" not in result["series"][0]["observations"][0]
    assert any(issue["code"] == "availability_unknown" for issue in result["issues"])


def test_financial_evidence_missing_observation_clock_does_not_fall_back_to_retrieval():
    with pytest.raises(ValueError, match="timestamp"):
        fe(table(fe_row(as_of=None)))


def test_financial_evidence_partitions_institutions_metrics_units_and_sources():
    input_table = table(fe_row(), fe_row(entity_id="EU-EUR"), fe_row(metric="OTHER_RATE"),
                        fe_row(unit="basis_points"), fe_row(source_url="https://other.invalid"))
    result = fe(input_table)
    assert len(result["series"]) == 5
    assert len({item["id"] for item in result["series"]}) == 5
    assert all(len(item["observations"]) == 1 for item in result["series"])
    assert {item["entity_id"] for item in result["provenance"]["identities"]} == {"US-USD", "EU-EUR"}


def test_financial_evidence_revisions_are_not_implicitly_latest_wins():
    with pytest.raises(ValueError, match="select revisions explicitly"):
        fe(table(fe_row(), fe_row(value=4.6, knowledge_time="2026-10-03T12:00:00Z")))


@pytest.mark.parametrize("status", [None, "public", "unknown", "licensed"])
def test_financial_evidence_does_not_infer_source_rights(status):
    result = fe(table(fe_row(rights_status=status)))
    assert result["series"][0]["source"]["rights"] == "unknown"


@pytest.mark.parametrize("changes", [{"availability": "unavailable"}, {"source_status": "SOURCE_REVIEW_HOLD"},
                                     {"source_status": "STALE"}, {"transport_status": "error"}])
def test_financial_evidence_ineligible_values_stay_blocked(changes):
    result = fe(table(fe_row(**changes)))
    assert result["series"][0]["observations"][0]["value"] is None
    assert result["provenance"]["input"]["results"][0]["value"] == 4.5
    assert any(issue["code"] == "ineligible_value_withheld" for issue in result["issues"])


@pytest.mark.parametrize("changes", [{"next_offset": 1}, {"offset": 5}, {"total_rows": 2}, {"returned_rows": True}])
def test_financial_evidence_incomplete_pages_require_explicit_collection(changes):
    with pytest.raises(ValueError, match="incomplete"):
        fe(table(fe_row(), **changes))


def test_financial_evidence_agent_result_contract_supported():
    result = fe(table(fe_row(), schema="financial-evidence.agent-result.v1"))
    assert result["series"][0]["observations"][0]["value"] == 4.5


def test_financial_evidence_missing_institution_identity_is_not_pooled():
    with pytest.raises(ValueError, match="identity"):
        fe(table(fe_row(entity_id=None)))


@pytest.mark.parametrize("status", [
    "restricted", "withheld", "unavailable", "not_available", "denied", "pending",
    "review_required", "prohibited", "metadata_only", "derived_only", "SOURCE_REVIEW_HOLD",
    "blocked", "stale", "not_permitted", "SOURCE-REVIEW-HOLD", " Review Required ",
    "NOT AVAILABLE", "Derived-Only", "Metadata Only",
])
def test_native_source_restrictions_override_conflicting_available_and_allowed(status):
    original = table(fe_row(source_status=status, availability="AVAILABLE", rights_status="allowed"))
    result = fe(original)
    assert result["series"][0]["observations"][0]["value"] is None
    issue = next(issue for issue in result["issues"] if issue["code"] == "ineligible_value_withheld")
    assert "source_status_restricted" in issue["reasons"]
    assert result["provenance"]["input"]["results"][0]["value"] == 4.5
    assert original["results"][0]["source_status"] == status


@pytest.mark.parametrize("status", [None, "", "unknown", "not_reported", "new_unrecognized_source_state"])
def test_unknown_source_status_cannot_be_promoted_by_availability(status):
    result = fe(table(fe_row(source_status=status)))
    assert result["series"][0]["observations"][0]["value"] is None
    assert any("source_status_unknown" in issue.get("reasons", []) for issue in result["issues"])


@pytest.mark.parametrize("rights", ["restricted", "DERIVED ONLY", "Metadata-Only", "Review Required", "pending", "prohibited"])
def test_native_rights_gates_override_fresh_source_status(rights):
    result = fe(table(fe_row(rights_status=rights, source_status="FRESH")))
    assert result["series"][0]["source"]["rights"] == "restricted"
    assert result["series"][0]["observations"][0]["value"] is None


@pytest.mark.parametrize("status", [None, "unknown", "partial", "unavailable"])
def test_parent_transport_cannot_promote_complete_row_or_clear_source_gate(status):
    result = fe(table(fe_row(), fe_row(entity_id="blocked-bank", source_status="SOURCE_REVIEW_HOLD"),
                      transport_status=status))
    assert all(series["observations"][0]["value"] is None for series in result["series"])
    withheld = [issue for issue in result["issues"] if issue["code"] == "ineligible_value_withheld"]
    assert len(withheld) == 2
    assert all("table_transport_not_complete" in issue["reasons"] for issue in withheld)
    assert any("source_status_restricted" in issue["reasons"] for issue in withheld)


def test_missing_parent_transport_remains_unknown():
    original = table(fe_row())
    original.pop("transport_status")
    result = fe(original)
    assert result["series"][0]["observations"][0]["value"] is None
    assert any("table_transport_not_complete" in issue.get("reasons", []) for issue in result["issues"])


@pytest.mark.parametrize("parent", [False, True])
def test_explicit_publication_block_is_preserved(parent):
    original = table(fe_row())
    (original if parent else original["results"][0])["publication_allowed"] = False
    result = fe(original)
    assert result["series"][0]["observations"][0]["value"] is None
    assert any("publication_not_allowed" in issue.get("reasons", []) for issue in result["issues"])


def test_distinct_undertow_measures_never_form_one_time_series():
    first = fe_row(product="Undertow", dataset="market_liquidity", entity_id="UST",
                   entity_name="bid_ask_spread", metric="stress_percentile", unit="ratio")
    second = fe_row(product="Undertow", dataset="market_liquidity", entity_id="UST",
                    entity_name="market_depth", metric="stress_percentile", unit="ratio",
                    as_of="2026-10-02", value=0.9)
    result = fe(table(first, second))
    assert len(result["series"]) == 2
    assert all(len(series["observations"]) == 1 for series in result["series"])
    assert {identity["measure"] for identity in result["provenance"]["identities"]} == {"bid_ask_spread", "market_depth"}
    assert result["provenance"]["input"]["results"] == [first, second]


@pytest.mark.parametrize("name", [None, ""])
def test_undertow_requires_the_native_measure_identity(name):
    with pytest.raises(ValueError, match="liquidity measure identity"):
        fe(table(fe_row(product="Undertow", dataset="market_liquidity", entity_name=name)))
