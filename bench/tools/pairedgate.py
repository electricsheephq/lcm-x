#!/usr/bin/env python3
"""Register and read powered paired gates without guessing.

Usage::

    python bench/tools/pairedgate.py register gate.yaml --registry registry.jsonl
    python bench/tools/pairedgate.py power NAME --registry registry.jsonl --b-pool 24
    python bench/tools/pairedgate.py read NAME --registry registry.jsonl --inputs GATE-INPUTS.txt
    python bench/tools/pairedgate.py verify NAME --registry registry.jsonl

Registration files use the same JSON-compatible YAML convention as the other
bench tools.  Every command accepts ``--json``; a short status line is also
written to stderr.  ``read`` returns ``AMBIGUOUS`` whenever registered clauses
conflict for the observed cells instead of choosing a favourable band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable


REGISTRY_VERSION = 1
DEFAULT_REGISTRY = Path("gate-registry.jsonl")
NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})?(?:\.\d+)?")
KEY_VALUE_RE = re.compile(
    r"(?P<key>rung|b|c|net|ratio|token_ratio|tokens_off|tokens_on|off|on)"
    r"\s*[:=]\s*(?P<value>[^,|\s]+)",
    re.IGNORECASE,
)


def sha256_file(path: str | Path) -> str:
    """Return the sha256 of the registration bytes, not a re-serialisation."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} must use the JSON-compatible YAML form supported by bench tools"
        ) from exc


def load_gate(path: str | Path) -> dict[str, Any]:
    """Load a registration document without changing its source bytes."""
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError("gate registration must be a JSON object")
    return value


def _read_registry(path: str | Path) -> list[dict[str, Any]]:
    registry_path = Path(path)
    if not registry_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with registry_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{registry_path}:{line_number} is not JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{registry_path}:{line_number} is not an object")
            entries.append(value)
    return entries


def register_gate(
    gate_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Append a frozen registration; a name can never be registered twice."""
    source = Path(gate_path)
    gate = load_gate(source)
    name = gate.get("name")
    if name in (None, ""):
        raise ValueError("gate registration requires a non-empty name")
    name = str(name)
    registry = Path(registry_path)
    entries = _read_registry(registry)
    if any(str(entry.get("name")) == name for entry in entries):
        raise ValueError(f"gate name already registered: {name}")

    entry = {
        "version": REGISTRY_VERSION,
        "name": name,
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
    return entry


def _entry_for(name: str, registry_path: str | Path) -> dict[str, Any]:
    matches = [
        entry
        for entry in _read_registry(registry_path)
        if str(entry.get("name")) == str(name)
    ]
    if not matches:
        raise ValueError(f"gate is not registered: {name}")
    if len(matches) > 1:
        raise ValueError(f"registry contains duplicate gate name: {name}")
    return matches[0]


def _entry_path(entry: dict[str, Any], registry_path: str | Path) -> Path:
    path = Path(str(entry.get("path", "")))
    if not path.is_absolute():
        path = Path(registry_path).resolve().parent / path
    return path


def verify_registration(
    name: str, registry_path: str | Path = DEFAULT_REGISTRY
) -> dict[str, Any]:
    """Re-hash the frozen source and report post-registration edits."""
    entry = _entry_for(name, registry_path)
    path = _entry_path(entry, registry_path)
    current = sha256_file(path) if path.is_file() else None
    frozen = str(entry.get("sha256", ""))
    ok = current is not None and current == frozen
    return {
        "name": str(name),
        "path": str(path),
        "frozen_sha256": frozen,
        "current_sha256": current,
        "ok": ok,
        "status": "PASS" if ok else "FAIL",
    }


def _first(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    lowered = {str(key).lower().replace("-", "_"): value for key, value in mapping.items()}
    for key in keys:
        value = lowered.get(key.lower().replace("-", "_"))
        if value is not None:
            return value
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("×", "")
    match = NUMBER_RE.search(text)
    return float(match.group(0)) if match else None


def _integer(value: Any, label: str) -> int:
    number = _number(value)
    if number is None or not number.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(number)


def _source_count(gate: dict[str, Any], registry_path: str | Path) -> int | None:
    """Resolve an optional registered b-pool source without evaluating code."""
    source = _first(gate, "b_pool", "b_pool_source", "bpool")
    if source is None:
        return None
    if isinstance(source, (int, float)) and not isinstance(source, bool):
        return _integer(source, "b-pool")
    if not isinstance(source, dict):
        return None
    explicit = _first(source, "count", "n", "size")
    if explicit is not None:
        return _integer(explicit, "b-pool count")
    raw_path = _first(source, "jsonl", "path", "file")
    if raw_path is None:
        return None
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = Path(registry_path).resolve().parent / path
    expression = _first(source, "filter", "filter_expression", "where")
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            if _filter_matches(value, expression):
                count += 1
    return count


def _filter_matches(row: dict[str, Any], expression: Any) -> bool:
    if expression in (None, "", True):
        return True
    text = str(expression).strip()
    match = re.fullmatch(
        r"([A-Za-z_][\w.-]*)\s*(==|=|!=)\s*(.+)", text
    )
    if not match:
        raise ValueError(
            "b-pool filter must be a simple field == value or field != value expression"
        )
    field, operator, raw_want = match.groups()
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            current = None
            break
        current = current[part]
    want: Any = raw_want.strip().strip("\"'")
    parsed = _number(want)
    if parsed is not None and str(want) == str(parsed).rstrip("0").rstrip("."):
        want = int(parsed) if parsed.is_integer() else parsed
    equal = current == want or str(current) == str(want)
    return equal if operator in {"=", "=="} else not equal


def power_gate(
    name: str,
    registry_path: str | Path = DEFAULT_REGISTRY,
    *,
    b_pool: int | None = None,
    floor: int = 20,
    pass_net: int = 8,
) -> dict[str, Any]:
    """Build the pre-run power memo, refusing a denominator below the floor."""
    if floor < 0 or pass_net < 0:
        raise ValueError("floor and pass_net must be non-negative")
    registration = verify_registration(name, registry_path)
    if not registration["ok"]:
        raise ValueError(f"registered gate source drifted: {name}")
    gate = load_gate(registration["path"])
    observed = b_pool if b_pool is not None else _source_count(gate, registry_path)
    if observed is None:
        raise ValueError("power requires --b-pool or a registered b-pool source")
    if observed < 0:
        raise ValueError("b-pool must be non-negative")
    conversion = pass_net / observed if observed else None
    powered = observed >= floor
    return {
        "version": 1,
        "name": str(name),
        "registration_sha256": registration["frozen_sha256"],
        "b_pool": observed,
        "floor": floor,
        "floor_check": {
            "observed": observed,
            "required": floor,
            "ok": powered,
        },
        "pass_bar": {
            "required_net": pass_net,
            "required_conversion_rate": conversion,
        },
        "verdict": "READY" if powered else "UNDERPOWERED",
        "refused": not powered,
    }


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _row_from_mapping(value: dict[str, Any]) -> dict[str, Any]:
    rung = _first(value, "rung", "scale", "tier", "size")
    if rung is None:
        raise ValueError("gate input row has no rung")
    b = _number(_first(value, "b", "benefit", "incomplete_to_complete"))
    c = _number(_first(value, "c", "cost", "complete_to_incomplete"))
    supplied_net = _number(_first(value, "net", "paired_net", "delta"))
    if b is None or c is None:
        raise ValueError(f"rung {rung!r} requires b and c")
    computed_net = b - c
    ratio = _number(
        _first(value, "ratio", "token_ratio", "token_inflation", "tokens_ratio")
    )
    off = _number(_first(value, "tokens_off", "off", "median_tok_off", "median_off"))
    on = _number(_first(value, "tokens_on", "on", "median_tok_on", "median_on"))
    if ratio is None and off not in (None, 0) and on is not None:
        ratio = on / off
    row = {
        "rung": str(rung),
        "b": int(b) if b.is_integer() else b,
        "c": int(c) if c.is_integer() else c,
        "net": int(computed_net) if computed_net.is_integer() else computed_net,
        "reported_net": (
            int(supplied_net) if supplied_net is not None and supplied_net.is_integer() else supplied_net
        ),
        "net_consistent": supplied_net is None or math.isclose(supplied_net, computed_net),
        "ratio": ratio,
        "tokens_off": off,
        "tokens_on": on,
    }
    return row


def parse_gate_inputs(path: str | Path) -> list[dict[str, Any]]:
    """Parse JSON rows or the F45-style whitespace/table input file."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith(("[", "{")):
        value = json.loads(text)
        if isinstance(value, dict):
            value = _first(value, "rungs", "rows", "inputs")
        if not isinstance(value, list):
            raise ValueError("gate inputs JSON must be a list of rung objects")
        return [_row_from_mapping(row) for row in value if isinstance(row, dict)]

    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw_line)
        if not line:
            continue
        lower = line.lower()
        if "rung" in lower and ("b" in lower or "net" in lower):
            continue
        matches = {match.group("key").lower(): match.group("value") for match in KEY_VALUE_RE.finditer(line)}
        if matches:
            try:
                rows.append(_row_from_mapping(matches))
            except ValueError as exc:
                raise ValueError(f"{source}:{line_number}: {exc}") from exc
            continue

        tokens = [token for token in re.split(r"[|,\t]+|\s+", line.strip()) if token]
        if len(tokens) < 5:
            continue
        numeric = [token for token in tokens[1:] if _number(token) is not None]
        if len(numeric) < 4 or _number(tokens[0]) is None:
            continue
        values: dict[str, Any] = {
            "rung": tokens[0],
            "b": numeric[0],
            "c": numeric[1],
            "net": numeric[2],
        }
        if len(numeric) >= 6:
            values.update({"tokens_off": numeric[-3], "tokens_on": numeric[-2], "ratio": numeric[-1]})
        else:
            values["ratio"] = numeric[3]
        try:
            rows.append(_row_from_mapping(values))
        except ValueError as exc:
            raise ValueError(f"{source}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"no rung rows found in {source}")
    return rows


def _normal_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _threshold(value: Any, metric: str, bar_name: str) -> tuple[float | None, float | None]:
    """Return (minimum, maximum) from a scalar, pair, or nested metric object."""
    if isinstance(value, dict):
        minimum = _number(_first(value, "min", "minimum", "lower", "at_least"))
        maximum = _number(_first(value, "max", "maximum", "upper", "at_most"))
        return minimum, maximum
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return _number(value[0]), _number(value[1])
    number = _number(value)
    if number is None:
        return None, None
    if metric == "net":
        return (number, None) if bar_name == "PASS" else (None, number)
    return (None, number) if bar_name == "PASS" else (number, None)


def _bar_spec(bar: Any, name: str) -> dict[str, Any]:
    if isinstance(bar, list):
        bar = {"clauses": bar}
    if not isinstance(bar, dict):
        return {}
    spec: dict[str, Any] = {"clauses": bar.get("clauses", [])}
    for key, value in bar.items():
        norm = _normal_key(key)
        if norm in {"net", "pairednet", "netband", "netrange"}:
            spec["net_min"], spec["net_max"] = _threshold(value, "net", name)
        elif norm in {
            "ratio",
            "tokenratio",
            "tokeninflation",
            "tokensratio",
            "tokencost",
            "costratio",
            "tokenmultiplier",
        }:
            spec["ratio_min"], spec["ratio_max"] = _threshold(value, "ratio", name)
            if isinstance(value, dict):
                scope = _first(value, "scope", "rung_scope", "clause_scope")
                if scope is not None:
                    spec["ratio_scope"] = scope
        elif "net" in norm and norm.endswith(("min", "minimum", "lower", "atleast")):
            spec["net_min"] = _number(value)
        elif "net" in norm and norm.endswith(("max", "maximum", "upper", "atmost")):
            spec["net_max"] = _number(value)
        elif (
            "ratio" in norm or "token" in norm or "cost" in norm
        ) and norm.endswith(("min", "minimum", "lower", "atleast", "above", "kill")):
            spec["ratio_min"] = _number(value)
        elif (
            "ratio" in norm or "token" in norm or "cost" in norm
        ) and norm.endswith(("max", "maximum", "upper", "atmost", "below")):
            spec["ratio_max"] = _number(value)
        elif norm in {"rungs", "toprungs", "rungscope", "scope"}:
            spec[norm] = value
        elif norm in {"ratioscope", "tokenscope", "tokenclausescope", "netScope".lower()}:
            spec["ratio_scope" if "ratio" in norm or "token" in norm else "net_scope"] = value

    clauses = spec.get("clauses")
    if isinstance(clauses, list):
        for clause in clauses:
            if not isinstance(clause, dict):
                continue
            metric = str(_first(clause, "metric", "name", "clause") or "").lower()
            if "net" in metric:
                low, high = _threshold(clause, "net", name)
                spec["net_min"] = low if low is not None else spec.get("net_min")
                spec["net_max"] = high if high is not None else spec.get("net_max")
            if "ratio" in metric or "token" in metric:
                low, high = _threshold(clause, "ratio", name)
                spec["ratio_min"] = low if low is not None else spec.get("ratio_min")
                spec["ratio_max"] = high if high is not None else spec.get("ratio_max")
                scope = _first(clause, "scope", "rung_scope", "clause_scope")
                if scope is not None:
                    spec["ratio_scope"] = scope
    return spec


def _bars(gate: dict[str, Any]) -> dict[str, Any]:
    value = _first(gate, "bars", "bands", "verdict_bands")
    if isinstance(value, dict):
        return {str(key).upper(): clause for key, clause in value.items()}
    return {
        name: _first(gate, name.lower(), name, f"{name.lower()}_bar")
        for name in ("PASS", "GRAY", "KILL")
        if _first(gate, name.lower(), name, f"{name.lower()}_bar") is not None
    }


def _sort_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(rows)
    try:
        return sorted(values, key=lambda row: float(str(row["rung"]).replace(",", "")))
    except ValueError:
        return values


def _selected_rows(rows: list[dict[str, Any]], spec: dict[str, Any], default_top: bool = True) -> list[dict[str, Any]]:
    value = spec.get("rungs", spec.get("toprungs", spec.get("rungscope")))
    ordered = _sort_rows(rows)
    if value is None:
        return ordered[-2:] if default_top and len(ordered) > 2 else ordered
    if isinstance(value, (list, tuple)):
        wanted = {str(item) for item in value}
        return [row for row in ordered if str(row["rung"]) in wanted]
    text = str(value).lower()
    if "top" in text:
        match = re.search(r"\d+", text)
        count = int(match.group(0)) if match else 2
        return ordered[-count:]
    if text in {"all", "any", "every"}:
        return ordered
    return [row for row in ordered if str(row["rung"]) == str(value)]


def _condition_result(
    name: str,
    metric: str,
    rows: list[dict[str, Any]],
    minimum: float | None,
    maximum: float | None,
    scope: str = "all",
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    for row in rows:
        observed = row.get(metric)
        met = observed is not None and (minimum is None or observed >= minimum) and (maximum is None or observed <= maximum)
        if observed is None:
            expression = f"{metric} unavailable"
        elif minimum is not None and maximum is not None:
            expression = f"{minimum:g} <= {metric}={observed:g} <= {maximum:g}"
        elif minimum is not None:
            expression = f"{metric}={observed:g} >= {minimum:g}"
        elif maximum is not None:
            expression = f"{metric}={observed:g} <= {maximum:g}"
        else:
            expression = f"{metric} observed"
        details.append({"rung": row["rung"], "observed": observed, "met": met, "arithmetic": expression})
    values = [detail["met"] for detail in details]
    met = any(values) if "any" in str(scope).lower() else bool(values) and all(values)
    return {"name": name, "metric": metric, "scope": scope, "met": met, "rungs": details}


def _evaluate_bar(
    name: str,
    bar: Any,
    rows: list[dict[str, Any]],
    pass_spec: dict[str, Any],
) -> dict[str, Any]:
    spec = _bar_spec(bar, name)
    target = _selected_rows(rows, spec, default_top=True)
    results: list[dict[str, Any]] = []
    if spec.get("net_min") is not None or spec.get("net_max") is not None:
        results.append(_condition_result(
            f"{name}.net", "net", target, spec.get("net_min"), spec.get("net_max"), str(spec.get("net_scope", "all"))
        ))
    if spec.get("ratio_min") is not None or spec.get("ratio_max") is not None:
        scope = str(spec.get("ratio_scope", "all"))
        ratio_rows = target
        if name == "KILL" and "pass" in scope.lower():
            pass_min = pass_spec.get("net_min")
            ratio_rows = [row for row in rows if pass_min is not None and row.get("net") is not None and row["net"] >= pass_min]
        results.append(_condition_result(
            f"{name}.token_ratio", "ratio", ratio_rows, spec.get("ratio_min"), spec.get("ratio_max"), "any" if name == "KILL" else scope
        ))
    # PASS and GRAY require all registered clauses. KILL is an OR bar.
    if name == "KILL":
        met = any(item["met"] for item in results)
    else:
        met = bool(results) and all(item["met"] for item in results)
    return {"name": name, "met": met, "clauses": results, "spec": spec, "target_rungs": [row["rung"] for row in target]}


def read_verdict(
    name: str,
    inputs_path: str | Path,
    registry_path: str | Path = DEFAULT_REGISTRY,
) -> dict[str, Any]:
    """Read registered PASS/GRAY/KILL bands and expose every arithmetic clause."""
    registration = verify_registration(name, registry_path)
    if not registration["ok"]:
        raise ValueError(f"registered gate source drifted: {name}")
    gate = load_gate(registration["path"])
    rows = parse_gate_inputs(inputs_path)
    bars = _bars(gate)
    if not bars:
        raise ValueError("gate registration has no PASS/GRAY/KILL bars")
    pass_result = _evaluate_bar("PASS", bars.get("PASS", {}), rows, {})
    pass_spec = pass_result["spec"]
    # Re-evaluate PASS with its own spec available to any passing-rung KILL clause.
    pass_result = _evaluate_bar("PASS", bars.get("PASS", {}), rows, pass_spec)
    gray_result = _evaluate_bar("GRAY", bars.get("GRAY", {}), rows, pass_spec)
    kill_result = _evaluate_bar("KILL", bars.get("KILL", {}), rows, pass_spec)

    conflicts: list[str] = []
    arithmetic_mismatch = [row["rung"] for row in rows if not row["net_consistent"]]
    if arithmetic_mismatch:
        conflicts.append("input.net_arithmetic")
    for band_result in (pass_result, gray_result, kill_result):
        for clause in band_result["clauses"]:
            if any(detail["observed"] is None for detail in clause["rungs"]):
                conflicts.append(f"{clause['name']}:missing")

    kill_ratio_clause = next((item for item in kill_result["clauses"] if item["metric"] == "ratio"), None)
    kill_spec = kill_result["spec"]
    if gray_result["met"] and kill_ratio_clause and kill_spec.get("ratio_min") is not None:
        scope = str(kill_spec.get("ratio_scope", ""))
        ratios_above = any(
            row.get("ratio") is not None and row["ratio"] > kill_spec["ratio_min"] for row in rows
        )
        passing_rows = [
            row for row in rows
            if pass_spec.get("net_min") is not None and row.get("net") is not None and row["net"] >= pass_spec["net_min"]
        ]
        if ratios_above and "pass" in scope.lower() and not passing_rows:
            conflicts.extend(["GRAY.net", "KILL.token_ratio", "KILL.token_scope:no_passing_rung"])
        elif ratios_above and kill_result["met"]:
            conflicts.extend(["GRAY.net", "KILL.token_ratio"])
    if pass_result["met"] and kill_result["met"]:
        conflicts.extend(["PASS", "KILL"])

    if conflicts:
        verdict = "AMBIGUOUS"
    elif pass_result["met"]:
        verdict = "PASS"
    elif kill_result["met"]:
        verdict = "KILL"
    elif gray_result["met"]:
        verdict = "GRAY"
    else:
        verdict = "AMBIGUOUS"
        conflicts.append("no_registered_band_match")

    arithmetic_rows = []
    for row in rows:
        arithmetic_rows.append({
            **row,
            "net_arithmetic": f"{row['b']} - {row['c']} = {row['net']}",
        })
    return {
        "version": 1,
        "name": str(name),
        "registration_sha256": registration["frozen_sha256"],
        "inputs": str(Path(inputs_path)),
        "rungs": arithmetic_rows,
        "clauses": {
            "PASS": pass_result,
            "GRAY": gray_result,
            "KILL": kill_result,
        },
        "conflicts": sorted(set(conflicts)),
        "verdict": verdict,
    }


# Small function aliases keep the module convenient to import in analysis code.
register = register_gate
power = power_gate
read = read_verdict
verify = verify_registration


def _summary(result: dict[str, Any]) -> str:
    verdict = result.get("verdict", result.get("status", "UNKNOWN"))
    name = result.get("name", "")
    if result.get("conflicts"):
        return f"{name}: {verdict} conflicts={','.join(result['conflicts'])}"
    return f"{name}: {verdict}"


def _emit(result: dict[str, Any], json_output: bool = True) -> None:
    print(_summary(result), file=sys.stderr)
    if json_output:
        print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register, power, and strictly read paired gates.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="emit the bounded JSON report")
    parser.add_argument("--registry", type=Path, default=None, help="append-only registry JSONL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register", help="freeze a gate registration")
    register.add_argument("gate", type=Path)
    register.add_argument("--registry", type=Path, default=argparse.SUPPRESS)
    register.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS)

    power = subparsers.add_parser("power", help="write a power memo or refuse an underpowered pool")
    power.add_argument("name")
    power.add_argument("--registry", type=Path, default=argparse.SUPPRESS)
    power.add_argument("--b-pool", type=int, required=False)
    power.add_argument("--floor", type=int, default=20)
    power.add_argument("--pass-net", type=int, default=8)
    power.add_argument("-o", "--output", type=Path)
    power.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS)

    read = subparsers.add_parser("read", help="read a registered gate strictly")
    read.add_argument("name")
    read.add_argument("--inputs", type=Path, required=True)
    read.add_argument("--registry", type=Path, default=argparse.SUPPRESS)
    read.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS)

    verify = subparsers.add_parser("verify", help="verify a frozen registration sha")
    verify.add_argument("name")
    verify.add_argument("--registry", type=Path, default=argparse.SUPPRESS)
    verify.add_argument("--json", dest="json_output", action="store_true", default=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = getattr(args, "registry", None) or DEFAULT_REGISTRY
    json_output = bool(getattr(args, "json_output", False) or True)
    try:
        if args.command == "register":
            result = register_gate(args.gate, registry)
            result = {**result, "status": "PASS"}
            _emit(result, json_output)
            return 0
        if args.command == "power":
            result = power_gate(
                args.name,
                registry,
                b_pool=args.b_pool,
                floor=args.floor,
                pass_net=args.pass_net,
            )
            if args.output:
                args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                result = {**result, "output": str(args.output)}
            _emit(result, json_output)
            return 0 if not result["refused"] else 3
        if args.command == "read":
            result = read_verdict(args.name, args.inputs, registry)
            _emit(result, json_output)
            return {"PASS": 0, "GRAY": 0, "KILL": 1, "AMBIGUOUS": 2}[result["verdict"]]
        result = verify_registration(args.name, registry)
        _emit(result, json_output)
        return 0 if result["ok"] else 1
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
