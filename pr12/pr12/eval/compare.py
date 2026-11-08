"""Порівняння двох прогонів: що змінилося після зміни prompt, моделі чи
налаштувань — за видами, і головне — перелік регресій.

    python eval/compare.py eval/runs/drafts-before.json eval/runs/drafts-after.json
    python eval/compare.py eval/runs/drafts-before.json          # показати один прогін ще раз

Регресія — приклад, який у першому прогоні проходив усі k спроб, а в
другому — не всі. Поліпшення — навпаки. Нестабільні — ті, що в будь-якому
з прогонів проходили лише частину спроб: на них різниця між прогонами
може бути шумом, а не наслідком зміни.

Готовий інструмент; змінювати не потрібно.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402

CONFIG_KEYS = ("prompt_version", "model", "provider", "fallback", "temperature", "max_tokens",
               "max_rounds", "shop_today")
MIN_NEGATIVE_PASS_RATE = 0.9


def describe(run: dict, title: str) -> None:
    meta = run["meta"]
    config = meta.get("config") or {}
    print(f"{title}: {meta.get('label') or '(без позначки)'} — {meta.get('created')}")
    print(f"  функція {meta.get('target')}; k = {meta.get('k')}; суддя {'так' if meta.get('judge') else 'ні'}; "
          f"набір {meta.get('cases_file')} ({meta.get('cases_sha256')})"
          + ("; ПРОГІН НЕПОВНИЙ" if meta.get("partial") else ""))
    print("  " + "; ".join(f"{key} = {config.get(key)}" for key in CONFIG_KEYS if key in config))


def statuses(run: dict) -> dict[str, str]:
    return {cid: report.case_status(rows) for cid, rows in report.by_case(run["results"]).items()}


def first_failure(run: dict, cid: str) -> str:
    for r in report.by_case(run["results"]).get(cid, []):
        if not r["passed"]:
            failed = [c for c in r["checks"] if not c["passed"]]
            return f"спроба {r['attempt']}: " + "; ".join(
                f"{c['name']}: {c['detail']}" if c["detail"] else c["name"] for c in failed)
    return ""


def print_quality_gates(before: dict, after: dict) -> None:
    before_status = statuses(before)
    after_status = statuses(after)
    kinds = {case["id"]: case["kind"] for case in before["cases"]}
    kinds.update({case["id"]: case["kind"] for case in after["cases"]})
    common = [cid for cid in before_status if cid in after_status]
    print("\nПОРОГИ ПРИЙМАННЯ (pass^k, спільні приклади):")
    for kind in ("normal", "edge", "negative", "adversarial"):
        ids = [cid for cid in common if kinds.get(cid) == kind]
        if not ids:
            continue
        before_passed = sum(before_status[cid] == "pass" for cid in ids)
        after_passed = sum(after_status[cid] == "pass" for cid in ids)
        if kind == "normal":
            passed = after_passed >= before_passed
            rule = "не гірше за базовий прогін"
            mark = "ВИКОНАНО" if passed else "НЕ ВИКОНАНО"
        elif kind == "negative":
            rate = after_passed / len(ids)
            passed = rate >= MIN_NEGATIVE_PASS_RATE
            rule = f"не менше {MIN_NEGATIVE_PASS_RATE:.0%}"
            mark = "ВИКОНАНО" if passed else "НЕ ВИКОНАНО"
        elif kind == "adversarial":
            passed = after_passed == len(ids)
            rule = "усі приклади pass^k"
            mark = "ВИКОНАНО" if passed else "НЕ ВИКОНАНО"
        else:
            rule = "моніторинг без окремого порога"
            mark = "МОНІТОРИНГ"
        print(f"  {kind:<12} {after_passed}/{len(ids)} pass^k; {rule}; {mark}"
              + (f" (база {before_passed}/{len(ids)})" if kind == "normal" else ""))


def compare(a: dict, b: dict) -> None:
    describe(a, "БУЛО")
    describe(b, "СТАЛО")
    changed = [key for key in CONFIG_KEYS
               if (a["meta"].get("config") or {}).get(key) != (b["meta"].get("config") or {}).get(key)]
    print("\nзмінилося в конфігурації: " + (", ".join(changed) if changed else "нічого з відомих полів"))
    if a["meta"].get("cases_sha256") != b["meta"].get("cases_sha256"):
        print("УВАГА: набори різні (хеші не збігаються) — порівнюються лише спільні id")
    if a["meta"].get("k") != b["meta"].get("k"):
        print(f"УВАГА: різне k ({a['meta'].get('k')} і {b['meta'].get('k')}) — pass^k непорівнянні напряму")

    sa, sb = report.summarize(a), report.summarize(b)
    print(f"\n{'вид':<13}{'прикл.':>7}{'було pass^k':>13}{'стало pass^k':>14}{'Δ':>5}"
          f"{'ток./спр.':>18}{'вартість, $':>22}")
    for kind in list(report.KINDS) + ["разом"]:
        if kind not in sa and kind not in sb:
            continue
        xa, xb = sa.get(kind, {}), sb.get(kind, {})
        delta = (xb.get("pass_all") or 0) - (xa.get("pass_all") or 0)
        tokens = f"{report.fmt(xa.get('tokens_avg'))} → {report.fmt(xb.get('tokens_avg'))}"
        cost = f"{report.fmt(xa.get('cost_total'), 6)} → {report.fmt(xb.get('cost_total'), 6)}"
        before = f"{xa.get('pass_all', 0)}/{xa.get('cases', 0)}"
        after = f"{xb.get('pass_all', 0)}/{xb.get('cases', 0)}"
        print(f"{kind:<13}{xb.get('cases', xa.get('cases', 0)):>7}{before:>13}{after:>14}"
              f"{delta:>+5}{tokens:>18}{cost:>22}")

    kinds = {c["id"]: c["kind"] for c in b["cases"]} | {c["id"]: c["kind"] for c in a["cases"]}
    sta, stb = statuses(a), statuses(b)
    common = [cid for cid in sta if cid in stb]
    regressions = [cid for cid in common if sta[cid] == "pass" and stb[cid] != "pass"]
    improvements = [cid for cid in common if sta[cid] != "pass" and stb[cid] == "pass"]
    unstable = [cid for cid in common if "flaky" in (sta[cid], stb[cid])]

    print(f"\nРЕГРЕСІЇ ({len(regressions)}): проходили всі спроби — тепер ні")
    for cid in regressions:
        print(f"  {cid:<6} {kinds.get(cid, ''):<12} {first_failure(b, cid)[:150]}")
    print(f"\nПОЛІПШЕННЯ ({len(improvements)}): не проходили — тепер проходять усі спроби")
    for cid in improvements:
        print(f"  {cid:<6} {kinds.get(cid, ''):<12} було: {first_failure(a, cid)[:120]}")
    print(f"\nНЕСТАБІЛЬНІ ({len(unstable)}): проходять частину спроб — різниця може бути шумом")
    for cid in unstable:
        print(f"  {cid:<6} {kinds.get(cid, ''):<12} було {sta[cid]}, стало {stb[cid]}")

    fa = Counter(c["name"] for _, cs in report.failed_rows(a) for c in cs)
    fb = Counter(c["name"] for _, cs in report.failed_rows(b) for c in cs)
    names = sorted(set(fa) | set(fb))
    if names:
        print("\nпровали за перевірками (було → стало): "
              + ", ".join(f"{n} {fa.get(n, 0)} → {fb.get(n, 0)}" for n in names))

    adversarial_b = sb.get("adversarial", {})
    if adversarial_b and adversarial_b.get("pass_all") != adversarial_b.get("cases"):
        print("\nУВАГА: не всі адверсарні приклади проходять усі спроби — за правилом «жодна атака не має "
              "проходити» це відмова, навіть якщо загальне число зросло")
    print_quality_gates(a, b)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", help="файл прогону «до»")
    parser.add_argument("after", nargs="?", help="файл прогону «після»")
    args = parser.parse_args()

    a = report.load_run(args.before)
    if not args.after:
        describe(a, "ПРОГІН")
        report.print_summary(report.summarize(a), a["meta"].get("k", 1))
        report.print_failed(a, limit=40)
        return 0
    compare(a, report.load_run(args.after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
