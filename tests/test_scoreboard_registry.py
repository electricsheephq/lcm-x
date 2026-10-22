"""The public scoreboard registry must stay valid: every row in bench/scoreboard/results.jsonl
passes the generator's schema checks (RESULTS-SCHEMA.md). The bench/scoreboard tests are outside
`testpaths`, so without this test a malformed row merges unnoticed."""

from bench.scoreboard import generate


def test_every_registry_row_validates():
    rows = generate.load_rows(generate.RESULTS_PATH)
    assert rows
    assert all(isinstance(row["breakdown"], str) for row in rows)
