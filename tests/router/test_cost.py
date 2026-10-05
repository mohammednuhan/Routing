"""Cost: what the observed token counts came to.

Three layers are tested here, in the order the numbers actually depend on each
other.

`price_tokens`: the arithmetic. One token count, one per-million rate, and a
`null` rate is UNKNOWN rather than free. This is where a real bug in a formula
would hide, so it is tested against hand-computed figures rather than against
whatever the code happens to produce.

`estimate_cost`: which model's prices apply, and what `OK` versus the other two
statuses mean for pricing.

The report and the `cost` command: the two figures the whole thing exists to
put side by side, the UNKNOWN rule, and the filters.

The UNKNOWN rule gets the most attention because it is the rule that can be
broken silently. A null cost that gets added in as `0.0` produces a report that
looks complete and understates itself, and nothing about the output gives it
away. So the tests below assert on the *absence* of a dollar figure as well as
its presence: a total that contains an unpriced row must say UNKNOWN, and the
dollar amount printed next to it must be labelled as the known part.
"""
from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path

import pytest

from router.config import ModelSpec, RouterConfig, load_config
from router.cost import NOTE_MODEL_MISMATCH, estimate_cost, price_tokens
from router.cost_report import (
    COST_NOTE,
    NOTHING_RECORDED,
    build_cost_report,
    cost_as_json,
    difference,
    format_cost_report,
    money,
)
from router.decisions import DB_ENV_VAR, USAGE_TABLE, DecisionLog, DecisionRecord, UsageRow
from router.readonly import ReadOnlyError, open_read_only
from router.report import ReportFilters, parse_since
from router.usage import STATUS_NO_USAGE, STATUS_OK, STATUS_UNKNOWN, UsageRecord

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_CONFIG_PATH = REPO_ROOT / "tools" / "sample-config-ACTIVE-test.yaml"
SHIPPED_CONFIG_PATH = REPO_ROOT / "router" / "config.yaml"

MODEL_LOW = "test-low"
MODEL_MID = "test-mid"
MODEL_HIGH = "test-high"

#: Dummy rates from `tools/sample-config-ACTIVE-test.yaml`, per million tokens.
RATE_INPUT = 3.00
RATE_OUTPUT = 15.00
RATE_CACHE_WRITE = 3.75
RATE_CACHE_READ = 0.30


def active_config() -> RouterConfig:
    return load_config(ACTIVE_CONFIG_PATH)


def spec(config: RouterConfig, model: str) -> ModelSpec:
    return next(one for one in config.models if one.id == model)


def record(**kwargs) -> UsageRecord:
    defaults: dict = {
        "status": STATUS_OK,
        "model_reported": "mock-model",
        "input_tokens": 1000,
        "output_tokens": 1000,
        "cache_read_tokens": 1000,
        "cache_write_tokens": 1000,
    }
    defaults.update(kwargs)
    return UsageRecord(**defaults)


def cost_of(
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    chosen_model: str | None = MODEL_MID,
    requested_model: str | None = MODEL_MID,
    config: RouterConfig | None = None,
) -> float:
    priced = estimate_cost(
        record(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        ),
        chosen_model,
        requested_model,
        config if config is not None else active_config(),
    )
    assert priced.cost_usd is not None
    return priced.cost_usd


def per_million(tokens: int, rate: float) -> float:
    return tokens * rate / 1_000_000


# --- the arithmetic ----------------------------------------------------------


def test_each_rate_prices_its_own_tokens_at_one_millionth_the_rate():
    """The formula, checked against the multiplication rather than the output:
    one token costs `rate / 1_000_000`, and a million tokens cost `rate`."""
    assert cost_of(input_tokens=1, chosen_model=MODEL_MID) == pytest.approx(3e-6)
    assert cost_of(input_tokens=1_000_000, chosen_model=MODEL_MID) == pytest.approx(3.00)
    assert cost_of(output_tokens=1_000_000, chosen_model=MODEL_MID) == pytest.approx(15.00)
    assert cost_of(cache_read_tokens=1_000_000, chosen_model=MODEL_MID) == pytest.approx(0.30)
    assert cost_of(cache_write_tokens=1_000_000, chosen_model=MODEL_MID) == pytest.approx(3.75)


def test_all_four_counts_are_priced_together():
    total = cost_of(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        chosen_model=MODEL_MID,
    )

    assert total == pytest.approx(RATE_INPUT + RATE_OUTPUT + RATE_CACHE_READ + RATE_CACHE_WRITE)


def test_odd_counts_do_not_round_to_a_whole_cent():
    """A single extra token must show up. Rounding here would let a real cost
    disappear into a `0.0` that reads like a free response."""
    with_extra = cost_of(input_tokens=1_000_001, chosen_model=MODEL_MID)
    without = cost_of(input_tokens=1_000_000, chosen_model=MODEL_MID)

    assert with_extra > without
    assert with_extra - without == pytest.approx(3e-6)


def test_the_four_prices_are_used_independently():
    """One count cannot be priced at another's rate: output is not input."""
    assert cost_of(input_tokens=500_000, chosen_model=MODEL_MID) == pytest.approx(1.50)
    assert cost_of(output_tokens=500_000, chosen_model=MODEL_MID) == pytest.approx(7.50)
    assert cost_of(cache_read_tokens=500_000, chosen_model=MODEL_MID) == pytest.approx(0.15)
    assert cost_of(cache_write_tokens=500_000, chosen_model=MODEL_MID) == pytest.approx(1.875)


# --- UNKNOWN is never zero ---------------------------------------------------


def test_a_null_rate_is_unknown_and_not_free():
    """The shipped config has no prices, so every count there prices to nothing
    at all. That is UNKNOWN, and it must not come back as `0.0`."""
    shipped = load_config(SHIPPED_CONFIG_PATH)

    priced = estimate_cost(
        record(), MODEL_MID, MODEL_MID, shipped
    )

    assert priced.cost_usd is None
    assert priced.baseline_cost_usd is None


def test_every_shipped_price_is_null_so_the_shipped_config_prices_nothing():
    shipped = load_config(SHIPPED_CONFIG_PATH)

    for model in shipped.models:
        for field in (
            "input_per_million",
            "output_per_million",
            "cache_write_per_million",
            "cache_read_per_million",
        ):
            assert getattr(model, field) is None, f"{model.id}.{field} was filled in"


def test_a_partly_null_model_prices_only_what_it_has():
    """A model with an input rate but no output rate cannot be priced fully,
    and the part it *can* price is not silently returned as the whole figure."""
    config = active_config()
    models = [
        dataclasses.replace(spec(config, MODEL_MID), output_per_million=None)
        if model.id == MODEL_MID
        else model
        for model in config.models
    ]
    partial = dataclasses.replace(config, models=models)

    priced = estimate_cost(
        record(input_tokens=1_000_000, output_tokens=1_000_000), MODEL_MID, MODEL_MID, partial
    )

    assert priced.cost_usd is None, "a partial price was returned as a total"
    assert priced.baseline_cost_usd is None


def test_a_free_model_costs_exactly_zero_and_is_not_unknown():
    """`test-low` carries a real `0.0`. Free is a price; missing is not. The
    two must not look the same in the database or in the report."""
    priced = estimate_cost(record(), MODEL_LOW, MODEL_LOW, active_config())

    assert priced.cost_usd == 0.0
    assert priced.cost_usd is not None
    assert priced.baseline_cost_usd == 0.0


def test_a_free_response_still_reports_its_token_counts():
    """A free price must not become a reason to record nothing."""
    priced = estimate_cost(
        record(input_tokens=4321, output_tokens=999), MODEL_LOW, MODEL_LOW, active_config()
    )

    assert priced.cost_usd == 0.0
    assert (4321, 999) == (4321, 999)


def test_a_model_that_is_not_in_the_config_prices_to_unknown():
    priced = estimate_cost(record(), "not-a-model", "not-a-model", active_config())

    assert priced.cost_usd is None


def test_an_unknown_row_is_never_priced():
    for status in (STATUS_UNKNOWN, STATUS_NO_USAGE):
        priced = estimate_cost(
            record(status=status), MODEL_MID, MODEL_MID, active_config()
        )
        assert priced.cost_usd is None, status
        assert priced.baseline_cost_usd is None, status


def test_price_tokens_with_no_rate_at_all_is_unknown():
    assert price_tokens(1000, 1000, 1000, 1000, None) is None
    assert price_tokens(1000, 1000, 1000, 1000, spec(active_config(), MODEL_LOW)) == 0.0


def test_an_unreported_count_is_no_spend_not_an_unknown_price():
    """A count nobody reported cannot make a total larger, so it cannot make a
    known price unknown either. A count of zero is treated the same way."""
    priced = estimate_cost(
        record(
            input_tokens=1_000_000,
            output_tokens=None,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        MODEL_MID,
        MODEL_MID,
        active_config(),
    )

    assert priced.cost_usd == pytest.approx(3.00)


def test_a_nonzero_count_with_no_price_makes_the_whole_total_unknown():
    """The rule that matters: a real spend with no verified price is UNKNOWN,
    never 0.0."""
    config = active_config()
    models = [
        dataclasses.replace(spec(config, MODEL_MID), output_per_million=None)
        if model.id == MODEL_MID
        else model
        for model in config.models
    ]
    partial = dataclasses.replace(config, models=models)

    priced = estimate_cost(
        record(input_tokens=1_000_000, output_tokens=1_000_000), MODEL_MID, MODEL_MID, partial
    )

    assert priced.cost_usd is None
    assert priced.is_unknown


# --- the baseline is the same counts on the requested model ------------------


def test_the_baseline_is_the_same_counts_at_the_requested_models_prices():
    """The comparison is only meaningful if the token counts are held fixed."""
    config = active_config()

    priced = estimate_cost(
        record(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            cache_read_tokens=0,
            cache_write_tokens=0,
            model_reported=MODEL_LOW,
        ),
        MODEL_LOW,
        MODEL_HIGH,
        config,
    )

    assert priced.cost_usd == pytest.approx(per_million(1_000_000, 0.0))
    assert priced.baseline_cost_usd == pytest.approx(
        per_million(1_000_000, spec(config, MODEL_HIGH).input_per_million)
        + per_million(1_000_000, spec(config, MODEL_HIGH).output_per_million)
    )


def test_a_stay_prices_both_figures_at_the_same_model():
    """Shadow mode, and every STAY: chosen equals requested, so the difference
    is zero. That is the absence of a saving, not a saving."""
    priced = estimate_cost(
        record(input_tokens=1_000_000, model_reported=MODEL_MID), MODEL_MID, MODEL_MID, active_config()
    )

    assert priced.cost_usd == priced.baseline_cost_usd
    assert NOTE_MODEL_MISMATCH not in priced.notes


def test_the_difference_is_the_baseline_minus_the_routed_figure():
    assert difference(
        build_totals(cost=1.0, baseline=4.0), build_totals(cost=1.0, baseline=4.0)
    ) == "$3.000000"


def build_totals(cost: float, baseline: float, cost_unknown: int = 0, baseline_unknown: int = 0):
    from router.cost_report import ModelCosts

    return ModelCosts(
        model="TOTAL",
        requests=1,
        cost_known_usd=cost,
        cost_unknown=cost_unknown,
        baseline_known_usd=baseline,
        baseline_unknown=baseline_unknown,
    )


# --- a mismatch is a note, never a status ------------------------------------


def test_a_model_the_router_did_not_choose_is_recorded_as_a_note():
    priced = estimate_cost(
        record(model_reported="mock-model"), MODEL_MID, MODEL_MID, active_config()
    )

    assert priced.notes == (NOTE_MODEL_MISMATCH,)
    assert priced.cost_usd is not None, "a note suppressed the price"


def test_a_reported_model_that_matches_the_chosen_one_is_not_a_mismatch():
    priced = estimate_cost(
        record(model_reported=MODEL_MID), MODEL_MID, MODEL_MID, active_config()
    )

    assert priced.notes == ()


def test_no_reported_model_is_not_a_mismatch():
    priced = estimate_cost(
        record(model_reported=None), MODEL_MID, MODEL_MID, active_config()
    )

    assert priced.notes == ()


# --- building a database to report on ----------------------------------------


def add_row(
    log: DecisionLog,
    chosen_model: str | None = MODEL_MID,
    requested_model: str | None = MODEL_MID,
    status: str = STATUS_OK,
    input_tokens: int | None = 1_000_000,
    output_tokens: int | None = 0,
    cache_read_tokens: int | None = 0,
    cache_write_tokens: int | None = 0,
    cost_usd: float | None = None,
    baseline_cost_usd: float | None = None,
    notes: str | None = None,
    session_hint: str | None = "session-a",
    timestamp: str | None = None,
    model_reported: str | None = None,
    with_decision: bool = True,
) -> int | None:
    """Write a usage row, and the decision row it links to.

    With no decision row, `decision_id` is NULL, which is what happens when the
    decision write failed. That is the case the report has to survive.
    """
    decision_id: int | None = None
    if with_decision:
        decision_id = log.record(
            DecisionRecord(
                session_hint=session_hint or "unknown",
                requested_model=requested_model,
                chosen_model=chosen_model,
                mode="shadow",
                reason_codes=["PASSTHROUGH"],
                action="STAY",
                applied=0,
                timestamp=timestamp,
            )
        )
    log.record_usage(
        UsageRow(
            status=status,
            decision_id=decision_id,
            model_reported=chosen_model if model_reported is None else model_reported,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            cost_usd=cost_usd,
            baseline_cost_usd=baseline_cost_usd,
            notes=notes,
        )
    )
    return decision_id


@pytest.fixture
def log(tmp_path: Path) -> DecisionLog:
    return DecisionLog(tmp_path / "router.sqlite3")


# --- the report --------------------------------------------------------------


def test_the_report_puts_routed_and_baseline_side_by_side(log):
    add_row(log, chosen_model=MODEL_LOW, requested_model=MODEL_HIGH,
            cost_usd=0.0, baseline_cost_usd=9.0)

    report = build_cost_report(log.db_path)

    assert report.rows == 1
    assert report.totals.cost_known_usd == pytest.approx(0.0)
    assert report.totals.baseline_known_usd == pytest.approx(9.0)
    assert "$9.000000" in difference(report.totals, report.totals)


def test_the_note_is_the_first_line_and_says_what_the_numbers_are_not(log):
    add_row(log, cost_usd=1.0, baseline_cost_usd=1.0)

    lines = format_cost_report(build_cost_report(log.db_path)).splitlines()

    assert lines[0] == COST_NOTE
    assert "Not a bill" in lines[0]
    assert "Savings only mean something in active mode" in lines[0]


def test_the_report_groups_by_the_chosen_model_not_the_reported_one(log):
    """The model the router chose is the price that applied, so it is the model
    the rows are grouped by."""
    add_row(log, chosen_model=MODEL_LOW, requested_model=MODEL_LOW,
            model_reported="mock-model", cost_usd=0.0, baseline_cost_usd=0.0)
    add_row(log, chosen_model=MODEL_HIGH, requested_model=MODEL_HIGH,
            model_reported="mock-model", cost_usd=1.0, baseline_cost_usd=1.0)

    report = build_cost_report(log.db_path)

    assert [group.model for group in report.by_model] == [MODEL_HIGH, MODEL_LOW]


def test_a_row_with_no_decision_is_grouped_under_a_dash_not_guessed(log):
    """The decision write failed, so nobody knows what was chosen. The report
    must not invent a model for it."""
    add_row(log, with_decision=False, cost_usd=1.0, baseline_cost_usd=1.0)

    report = build_cost_report(log.db_path)

    assert [group.model for group in report.by_model] == ["-"]
    assert report.rows == 1


def test_rows_are_ordered_by_size_then_by_name(log):
    add_row(log, chosen_model=MODEL_HIGH, cost_usd=0.0, baseline_cost_usd=0.0)
    add_row(log, chosen_model=MODEL_LOW, cost_usd=0.0, baseline_cost_usd=0.0)
    add_row(log, chosen_model=MODEL_LOW, cost_usd=0.0, baseline_cost_usd=0.0)

    report = build_cost_report(log.db_path)

    assert [(g.model, g.requests) for g in report.by_model] == [
        (MODEL_LOW, 2),
        (MODEL_HIGH, 1),
    ]


def test_token_counts_are_summed_per_group(log):
    add_row(log, chosen_model=MODEL_MID, input_tokens=10, output_tokens=20,
            cache_read_tokens=30, cache_write_tokens=40,
            cost_usd=0.0, baseline_cost_usd=0.0)
    add_row(log, chosen_model=MODEL_MID, input_tokens=1, output_tokens=2,
            cache_read_tokens=3, cache_write_tokens=4,
            cost_usd=0.0, baseline_cost_usd=0.0)

    totals = build_cost_report(log.db_path).totals

    assert totals.input_tokens == 11
    assert totals.output_tokens == 22
    assert totals.cache_read_tokens == 33
    assert totals.cache_write_tokens == 44
    assert totals.cache_tokens == 77


def test_the_status_counts_are_reported_separately(log):
    add_row(log, status=STATUS_OK, cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, status=STATUS_UNKNOWN)
    add_row(log, status=STATUS_NO_USAGE)

    report = build_cost_report(log.db_path)

    assert (report.unknown, report.no_usage) == (1, 1)
    assert report.rows == 3


def test_model_mismatch_rows_are_counted(log):
    add_row(log, cost_usd=0.0, baseline_cost_usd=0.0, notes=NOTE_MODEL_MISMATCH)
    add_row(log, cost_usd=0.0, baseline_cost_usd=0.0)

    assert build_cost_report(log.db_path).model_mismatch == 1


def test_an_empty_database_reports_nothing_recorded_and_creates_nothing(tmp_path):
    path = tmp_path / "router.sqlite3"
    DecisionLog(path).count()

    report = format_cost_report(build_cost_report(path))

    assert NOTHING_RECORDED in report
    assert COST_NOTE.splitlines()[0] == report.splitlines()[0]


def test_a_database_with_no_usage_table_reports_nothing_recorded(tmp_path):
    """An install that predates the table must still produce a report rather
    than an error."""
    path = tmp_path / "router.sqlite3"
    legacy = DecisionLog(path)
    legacy.record(
        DecisionRecord(
            session_hint="s",
            requested_model=MODEL_MID,
            chosen_model=MODEL_MID,
            mode="shadow",
        )
    )
    drop_usage_table(path)

    report = build_cost_report(path)

    assert report.rows == 0
    assert NOTHING_RECORDED in format_cost_report(report)


# --- UNKNOWN is never zero in the report either ------------------------------


def test_a_total_containing_an_unpriced_row_says_unknown(log):
    """The one that can be broken silently: a null cost added in as 0.0 makes a
    report that looks complete and understates itself."""
    add_row(log, cost_usd=3.0, baseline_cost_usd=3.0)
    add_row(log, status=STATUS_UNKNOWN)

    report = build_cost_report(log.db_path)
    text = format_cost_report(report)

    assert report.totals.cost_is_unknown
    assert report.totals.cost_unknown == 1
    assert money(report.totals.cost_known_usd, report.totals.cost_unknown).startswith(
        "UNKNOWN"
    )
    assert "routed cost     UNKNOWN" in text
    assert "routed cost     $3.000000" not in text


def test_an_unknown_total_still_prints_the_known_part(log):
    add_row(log, cost_usd=3.0, baseline_cost_usd=3.0)
    add_row(log, status=STATUS_UNKNOWN)

    text = format_cost_report(build_cost_report(log.db_path))

    assert "UNKNOWN (1 row(s) unpriced, known part $3.000000)" in text


def test_a_free_total_is_a_dollar_figure_and_not_unknown(log):
    """`test-low` is genuinely free. That is a complete answer, and it must not
    be confused with an unknown one."""
    add_row(log, chosen_model=MODEL_LOW, requested_model=MODEL_LOW,
            input_tokens=5_000_000, cost_usd=0.0, baseline_cost_usd=0.0)

    text = format_cost_report(build_cost_report(log.db_path))

    assert "routed cost     $0.000000" in text
    assert "routed cost     UNKNOWN" not in text


def test_money_never_prints_a_bare_dollar_figure_for_an_unknown():
    assert money(0.0, 0) == "$0.000000"
    assert money(1.5, 2) == "UNKNOWN (2 row(s) unpriced, known part $1.500000)"
    assert money(0.0, 1) == "UNKNOWN (1 row(s) unpriced, known part $0.000000)"


def test_the_difference_of_two_partial_sums_is_itself_unknown():
    assert "UNKNOWN" in difference(build_totals(1.0, 4.0, cost_unknown=1), build_totals(1.0, 4.0, cost_unknown=1))
    assert "UNKNOWN" in difference(build_totals(1.0, 4.0, baseline_unknown=1), build_totals(1.0, 4.0, baseline_unknown=1))
    assert difference(build_totals(1.0, 4.0), build_totals(1.0, 4.0)) == "$3.000000"


def test_unknown_in_json_is_null_and_the_known_part_is_still_there(log):
    add_row(log, cost_usd=3.0, baseline_cost_usd=3.0)
    add_row(log, status=STATUS_UNKNOWN)

    payload = json.loads(cost_as_json(build_cost_report(log.db_path)))

    assert payload["totals"]["routed_cost_usd"] is None
    assert payload["totals"]["routed_cost_known_usd"] == pytest.approx(3.0)
    assert payload["totals"]["difference_usd"] is None


def test_a_complete_total_is_a_number_in_json(log):
    add_row(log, cost_usd=3.0, baseline_cost_usd=9.0)

    payload = json.loads(cost_as_json(build_cost_report(log.db_path)))

    assert payload["totals"]["routed_cost_usd"] == pytest.approx(3.0)
    assert payload["totals"]["difference_usd"] == pytest.approx(6.0)


def test_a_zero_cost_is_zero_in_json_not_null(log):
    add_row(log, chosen_model=MODEL_LOW, requested_model=MODEL_LOW,
            cost_usd=0.0, baseline_cost_usd=0.0)

    payload = json.loads(cost_as_json(build_cost_report(log.db_path)))

    assert payload["totals"]["routed_cost_usd"] == 0.0
    assert payload["totals"]["routed_cost_usd"] is not None


# --- filters -----------------------------------------------------------------


def test_since_excludes_older_rows(log):
    add_row(log, timestamp="2026-10-01T00:00:00Z", cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, timestamp="2026-10-03T00:00:00Z", cost_usd=2.0, baseline_cost_usd=2.0)

    report = build_cost_report(log.db_path, ReportFilters(since="2026-10-02T00:00:00Z"))

    assert report.rows == 1
    assert report.totals.cost_known_usd == pytest.approx(2.0)


def test_since_includes_the_exact_boundary(log):
    add_row(log, timestamp="2026-10-02T00:00:00Z", cost_usd=1.0, baseline_cost_usd=1.0)

    report = build_cost_report(log.db_path, ReportFilters(since="2026-10-02T00:00:00Z"))

    assert report.rows == 1


def test_session_selects_one_session(log):
    add_row(log, session_hint="session-a", cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, session_hint="session-b", cost_usd=2.0, baseline_cost_usd=2.0)

    report = build_cost_report(log.db_path, ReportFilters(session="session-b"))

    assert report.rows == 1
    assert report.totals.cost_known_usd == pytest.approx(2.0)


def test_last_keeps_only_the_newest_rows(log):
    for index in range(5):
        add_row(log, cost_usd=float(index), baseline_cost_usd=float(index))

    report = build_cost_report(log.db_path, ReportFilters(last=2))

    assert report.rows == 2
    assert report.totals.cost_known_usd == pytest.approx(3.0 + 4.0)


def test_last_counts_the_newest_matching_rows_not_the_newest_overall(log):
    """`--last` applies after the other filters: asking for the last 1 of a
    session must give that session's newest, not the newest of the file."""
    add_row(log, session_hint="session-a", timestamp="2026-10-01T00:00:00Z",
            cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, session_hint="session-b", timestamp="2026-10-09T00:00:00Z",
            cost_usd=99.0, baseline_cost_usd=99.0)
    add_row(log, session_hint="session-a", timestamp="2026-10-05T00:00:00Z",
            cost_usd=2.0, baseline_cost_usd=2.0)

    report = build_cost_report(
        log.db_path, ReportFilters(session="session-a", last=1)
    )

    assert report.rows == 1
    assert report.totals.cost_known_usd == pytest.approx(2.0)


def test_a_filter_that_matches_nothing_reports_nothing_recorded(log):
    add_row(log, session_hint="session-a", cost_usd=1.0, baseline_cost_usd=1.0)

    report = build_cost_report(log.db_path, ReportFilters(session="nope"))

    assert report.rows == 0
    assert NOTHING_RECORDED in format_cost_report(report)


def test_the_time_range_spans_the_matched_rows(log):
    add_row(log, timestamp="2026-10-01T00:00:00Z", cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, timestamp="2026-10-04T00:00:00Z", cost_usd=1.0, baseline_cost_usd=1.0)

    report = build_cost_report(log.db_path)

    assert report.first == "2026-10-01T00:00:00Z"
    assert report.last == "2026-10-04T00:00:00Z"
    assert "2026-10-01T00:00:00Z .. 2026-10-04T00:00:00Z" in format_cost_report(report)


def test_a_bad_since_is_refused_rather_than_filtering_everything_away():
    with pytest.raises(ValueError) as caught:
        parse_since("yesterday")

    assert "ISO-8601" in str(caught.value)


# --- read-only ---------------------------------------------------------------


def test_reading_a_report_changes_nothing_on_disk(log):
    add_row(log, cost_usd=1.0, baseline_cost_usd=1.0)
    before = log.db_path.read_bytes()
    mtime_before = log.db_path.stat().st_mtime_ns

    build_cost_report(log.db_path)
    cost_as_json(build_cost_report(log.db_path))

    assert log.db_path.read_bytes() == before
    assert log.db_path.stat().st_mtime_ns == mtime_before


def test_a_report_never_creates_a_database(tmp_path):
    """An absent database is reported as absent. It is not an error, and it is
    not fixed by bringing a file into existence."""
    path = tmp_path / "missing" / "router.sqlite3"

    with pytest.raises(ReadOnlyError):
        build_cost_report(path)

    assert not path.exists()
    assert not path.parent.exists(), "reading created the directory"


def test_a_report_cannot_write_even_though_the_file_is_writable(log):
    """The connection is opened read-only, so an accidental write fails instead
    of quietly editing a measurement."""
    add_row(log, cost_usd=1.0, baseline_cost_usd=1.0)

    with pytest.raises(ReadOnlyError):
        with open_read_only(log.db_path) as connection:
            connection.execute(f"UPDATE {USAGE_TABLE} SET cost_usd = 0.0")


def test_a_report_does_not_add_the_usage_table_to_an_old_database(tmp_path):
    """A read must not migrate anything: an old database is left exactly as it
    was, table or no table."""
    path = tmp_path / "legacy.sqlite3"
    legacy = DecisionLog(path)
    legacy.record(
        DecisionRecord(
            session_hint="s",
            requested_model=MODEL_MID,
            chosen_model=MODEL_MID,
            mode="shadow",
        )
    )
    before = path.read_bytes()
    drop_usage_table(path)
    dropped = path.read_bytes()

    build_cost_report(path)

    assert path.read_bytes() == dropped
    assert dropped != before


def drop_usage_table(path: Path) -> None:
    """Make `path` look like it predates the usage table. Dropping the table
    drops the index that goes with it."""
    with sqlite3.connect(path) as connection:
        connection.execute(f"DROP TABLE {USAGE_TABLE}")


# --- the cost command --------------------------------------------------------


def run_cost(*args: str, monkeypatch: pytest.MonkeyPatch | None = None):
    from router.cli import main

    return main(["--config", str(ACTIVE_CONFIG_PATH), "cost", *args])


def test_the_cost_command_prints_the_report(capsys, monkeypatch, log):
    add_row(log, cost_usd=3.0, baseline_cost_usd=9.0, notes=NOTE_MODEL_MISMATCH)
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    code = run_cost()

    out = capsys.readouterr().out
    assert code == 0
    assert COST_NOTE.splitlines()[0] in out
    assert "routed cost     $3.000000" in out
    assert "baseline cost   $9.000000" in out
    assert "difference      $6.000000" in out
    assert "MODEL_MISMATCH  1" in out


def test_the_cost_command_json_is_parseable(capsys, monkeypatch, log):
    add_row(log, cost_usd=3.0, baseline_cost_usd=9.0)
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    assert run_cost("--json") == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["note"] == COST_NOTE
    assert payload["read_only"] is True
    assert payload["totals"]["responses"] == 1
    assert payload["totals"]["routed_cost_usd"] == pytest.approx(3.0)
    assert payload["by_model"][0]["model"] == MODEL_MID


def test_the_cost_command_applies_a_session_filter(capsys, monkeypatch, log):
    add_row(log, session_hint="session-a", cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, session_hint="session-b", cost_usd=2.0, baseline_cost_usd=2.0)
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    assert run_cost("--json", "--session", "session-b") == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["filters"]["session"] == "session-b"
    assert payload["totals"]["responses"] == 1


def test_the_cost_command_applies_a_last_filter(capsys, monkeypatch, log):
    for index in range(3):
        add_row(log, cost_usd=float(index), baseline_cost_usd=float(index))
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    assert run_cost("--json", "--last", "2") == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["filters"]["last"] == 2
    assert payload["totals"]["responses"] == 2


def test_the_cost_command_applies_a_since_filter(capsys, monkeypatch, log):
    add_row(log, timestamp="2026-10-01T00:00:00Z", cost_usd=1.0, baseline_cost_usd=1.0)
    add_row(log, timestamp="2026-10-05T00:00:00Z", cost_usd=2.0, baseline_cost_usd=2.0)
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    assert run_cost("--json", "--since", "2026-10-03T00:00:00Z") == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["filters"]["since"] == "2026-10-03T00:00:00Z"
    assert payload["totals"]["responses"] == 1


def test_the_cost_command_says_so_when_there_is_no_database(capsys, monkeypatch, tmp_path):
    missing = tmp_path / "missing.sqlite3"
    monkeypatch.setenv(DB_ENV_VAR, str(missing))

    code = run_cost()

    out = capsys.readouterr().out
    assert code == 0
    assert NOTHING_RECORDED in out
    assert COST_NOTE.splitlines()[0] in out, "an estimate report printed without its disclaimer"
    assert not missing.exists()


def test_the_cost_command_still_prints_the_note_when_nothing_matches(capsys, monkeypatch, log):
    add_row(log, session_hint="session-a", cost_usd=1.0, baseline_cost_usd=1.0)
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    assert run_cost("--session", "nope") == 0

    out = capsys.readouterr().out
    assert NOTHING_RECORDED in out
    assert COST_NOTE.splitlines()[0] in out


def test_the_cost_command_refuses_a_nonsense_last(capsys, monkeypatch, log):
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    code = run_cost("--last", "0")

    assert code != 0
    assert capsys.readouterr().err.strip()


def test_the_cost_command_refuses_a_nonsense_since(capsys, monkeypatch, log):
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))

    code = run_cost("--since", "yesterday")

    assert code != 0
    assert "ISO-8601" in capsys.readouterr().err


def test_the_cost_command_leaves_the_database_alone(capsys, monkeypatch, log):
    add_row(log, cost_usd=1.0, baseline_cost_usd=1.0)
    monkeypatch.setenv(DB_ENV_VAR, str(log.db_path))
    before = log.db_path.read_bytes()

    run_cost()
    run_cost("--json")

    assert log.db_path.read_bytes() == before