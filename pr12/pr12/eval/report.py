"""Спільне для run.py і compare.py: читання файла прогону, статус
прикладу за k спробами, зведення за видами, друк таблиць.

Готовий модуль; змінювати не потрібно.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

KINDS = ("normal", "edge", "negative", "adversarial")


def load_run(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fmt(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def tokens_of(output: dict) -> int | None:
    """Токени спроби: вхід + вихід, якщо `usage` є."""
    usage = output.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if prompt is None and completion is None:
        return usage.get("total_tokens")
    return int(prompt or 0) + int(completion or 0)


def by_case(results: list[dict]) -> dict[str, list[dict]]:
    """Результати, згруповані за прикладом, у порядку спроб."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["id"]].append(r)
    for rows in groups.values():
        rows.sort(key=lambda r: r["attempt"])
    return groups


def case_status(rows: list[dict]) -> str:
    """`pass` — усі спроби пройшли (pass^k); `flaky` — лише деякі (pass@k,
    але не pass^k); `fail` — жодна."""
    passed = [r["passed"] for r in rows]
    if not passed:
        return "none"
    if all(passed):
        return "pass"
    if any(passed):
        return "flaky"
    return "fail"


def summarize(run: dict) -> dict:
    """Зведення за видами й разом: прикладів, pass^k, pass@k, спроб,
    токенів на спробу, вартість, тривалість."""
    kinds = {c["id"]: c["kind"] for c in run["cases"]}
    groups = by_case(run["results"])
    summary: dict = {}
    for kind in list(KINDS) + ["разом"]:
        ids = [cid for cid in groups if kind == "разом" or kinds.get(cid) == kind]
        if not ids:
            continue
        rows = [r for cid in ids for r in groups[cid]]
        statuses = Counter(case_status(groups[cid]) for cid in ids)
        tokens = [t for t in (tokens_of(r["output"]) for r in rows) if t is not None]
        costs = [r["output"].get("cost") for r in rows if isinstance(r["output"].get("cost"), (int, float))]
        elapsed = [r["output"].get("elapsed") for r in rows if isinstance(r["output"].get("elapsed"), (int, float))]
        summary[kind] = {
            "cases": len(ids),
            "pass_all": statuses["pass"],
            "pass_any": statuses["pass"] + statuses["flaky"],
            "flaky": statuses["flaky"],
            "attempts": len(rows),
            "attempts_passed": sum(1 for r in rows if r["passed"]),
            "errors": sum(1 for r in rows if r["output"].get("status") != "ok"),
            "tokens_avg": round(sum(tokens) / len(tokens)) if tokens else None,
            "cost_total": round(sum(costs), 6) if costs else None,
            "elapsed_p50": percentile(elapsed, 0.5),
        }
    return summary


def print_summary(summary: dict, k: int) -> None:
    print(f"\n{'вид':<13}{'прикл.':>7}{'pass^k':>8}{'pass@k':>8}{'спроб':>12}{'збоїв':>7}"
          f"{'ток./спр.':>11}{'вартість, $':>13}{'час p50, с':>12}")
    for kind, s in summary.items():
        attempts = f"{s['attempts_passed']}/{s['attempts']}"
        print(f"{kind:<13}{s['cases']:>7}{s['pass_all']:>8}{s['pass_any']:>8}{attempts:>12}"
              f"{s['errors']:>7}{fmt(s['tokens_avg']):>11}{fmt(s['cost_total'], 6):>13}"
              f"{fmt(s['elapsed_p50']):>12}")
    print(f"(k = {k}: pass^k — усі k спроб пройшли; pass@k — хоч одна)")


def failed_rows(run: dict) -> list[tuple[dict, list[dict]]]:
    """Спроби, що не пройшли, разом із провальними перевірками."""
    rows = []
    for r in run["results"]:
        if r["passed"]:
            continue
        rows.append((r, [c for c in r["checks"] if not c["passed"]]))
    return rows


def print_failed(run: dict, limit: int | None = None) -> None:
    kinds = {c["id"]: c["kind"] for c in run["cases"]}
    rows = failed_rows(run)
    if not rows:
        print("\nусі спроби пройшли")
        return
    print(f"\nНЕ ПРОЙШЛИ ({len(rows)} спроб):")
    for r, checks in rows[:limit]:
        reasons = "; ".join(f"{c['name']}: {c['detail']}" if c["detail"] else c["name"] for c in checks)
        print(f"  {r['id']:<6} {kinds.get(r['id'], ''):<12} спроба {r['attempt']}  {reasons[:160]}")
    if limit and len(rows) > limit:
        print(f"  … і ще {len(rows) - limit}")
    counter = Counter(c["name"] for _, checks in rows for c in checks)
    print("перевірки, що провалювалися найчастіше: "
          + ", ".join(f"{name} ×{n}" for name, n in counter.most_common(6)))
