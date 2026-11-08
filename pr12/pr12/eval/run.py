"""Прогін набору прикладів через AI-функцію й перевірки.

Запуск із папки pr12 (застосунки ПР11 і ПР8 запускати не потрібно —
адаптери імпортують їх зі своїх папок):

    python eval/run.py --target drafts --cases eval/cases/drafts.json --k 3 --label "v1, базовий"
    python eval/run.py --target agent  --cases eval/cases/agent.json
    python eval/run.py --target drafts --cases eval/cases/drafts.json --kinds adversarial --no-judge
    python eval/run.py --target drafts --cases eval/cases/drafts.json --ids d01,a02 --k 1

Кожен приклад виконується `k` разів (стабільність: pass^k проти pass@k),
після кожної спроби — усі перевірки з прикладу. Результат — файл у
`eval/runs/` з метаданими (функція, конфігурація, версія prompt, модель,
набір і його хеш, k, позначка), результатом кожної спроби, вердиктами
перевірок і зведенням за видами. Два такі файли порівнює
`eval/compare.py`.

Це готовий інструмент: він лише ходить в адаптер і перевірки. Приклади,
перевірки, суддя й висновки — ваші.
"""

import argparse
import datetime
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
PR_DIR = HERE.parent

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

# .env цієї роботи — першим: його змінні мають перевагу над .env застосунків,
# які адаптери завантажать пізніше (так можна, наприклад, увімкнути
# LLM_PROVIDER=mock лише для прогону).
load_dotenv(PR_DIR / ".env")
sys.path.insert(0, str(HERE))

import checks  # noqa: E402
import report  # noqa: E402
import targets  # noqa: E402


def load_cases(path: Path) -> list[dict]:
    """Прочитати набір і перевірити його форму: унікальні id, відомі види,
    є вхід і перевірки."""
    try:
        cases = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"набір {path} не читається: {exc}") from exc
    if not isinstance(cases, list):
        raise SystemExit(f"набір {path} має бути JSON-списком")
    problems = []
    seen = set()
    for i, case in enumerate(cases):
        if not isinstance(case, dict):
            problems.append(f"приклад #{i + 1} має бути JSON-об'єктом")
            continue
        cid = case.get("id")
        if not isinstance(cid, str) or not cid.strip():
            problems.append(f"приклад #{i + 1} без id")
            continue
        if cid in seen:
            problems.append(f"{cid}: id повторюється")
        seen.add(cid)
        if case.get("kind") not in report.KINDS:
            problems.append(f"{cid}: вид {case.get('kind')!r}, має бути один із {', '.join(report.KINDS)}")
        if not isinstance(case.get("input"), dict):
            problems.append(f"{cid}: немає input")
        specs = case.get("checks")
        if not isinstance(specs, list) or not specs:
            problems.append(f"{cid}: немає жодної перевірки — такий приклад проходить завжди")
            continue
        for spec in specs:
            if not isinstance(spec, dict):
                problems.append(f"{cid}: перевірка має бути JSON-об'єктом")
                continue
            if spec.get("check") not in checks.REGISTRY:
                problems.append(f"{cid}: невідома перевірка {spec.get('check')!r}")
    if problems:
        raise SystemExit("набір не в порядку:\n  " + "\n  ".join(problems))
    return cases


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def slim(output: dict) -> dict:
    """Що з результату адаптера зберігати: усе, крім системної інструкції
    (вона велика й однакова для всіх прикладів)."""
    return {k: v for k, v in output.items() if k != "system_prompt"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=sorted(targets.TARGETS), help="яку функцію оцінюємо")
    parser.add_argument("--cases", required=True, help="файл набору прикладів (JSON)")
    parser.add_argument("--k", type=int, default=1, help="скільки разів виконати кожен приклад")
    parser.add_argument("--ids", help="лише ці приклади, через кому")
    parser.add_argument("--kinds", help="лише ці види, через кому")
    parser.add_argument("--no-judge", action="store_true", help="пропустити перевірки судді")
    parser.add_argument("--label", default="", help="позначка прогону: що це за конфігурація")
    parser.add_argument("--out", help="куди записати результат; типово eval/runs/<функція>-<час>.json")
    args = parser.parse_args()

    if args.k < 1:
        parser.error("--k має бути не менше 1")
    cases_path = Path(args.cases)
    cases = load_cases(cases_path)
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    if args.kinds:
        wanted_kinds = {s.strip() for s in args.kinds.split(",")}
        cases = [c for c in cases if c["kind"] in wanted_kinds]
    if not cases:
        raise SystemExit("після відбору прикладів не лишилося")

    target = targets.load(args.target)
    try:
        config = target.setup()
    except NotImplementedError as exc:
        raise SystemExit(f"функція ще заготовка: {exc}") from exc

    skip = ("judge",) if args.no_judge else ()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else HERE / "runs" / f"{args.target}-{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"функція: {args.target}; модель: {config.get('model')}; провайдер: {config.get('provider')}; "
          f"версія prompt: {config.get('prompt_version')}; k = {args.k}; прикладів: {len(cases)}"
          + (f"; позначка: {args.label}" if args.label else ""))
    print(f"набір: {cases_path} ({file_sha256(cases_path)})\n")

    results: list[dict] = []
    started_all = time.perf_counter()
    try:
        for case in cases:
            for attempt in range(1, args.k + 1):
                target.before_case(case)
                try:
                    output = target.run(case)
                except NotImplementedError as exc:
                    raise SystemExit(f"функція ще заготовка: {exc}") from exc
                except Exception as exc:  # noqa: BLE001 — необроблений виняток функції = провал спроби
                    output = {"text": None, "status": "error", "calls": [],
                              "error": {"kind": "exception", "message": f"{type(exc).__name__}: {exc}"}}
                verdicts = checks.run_checks(case, output, skip=skip)
                passed = all(v.passed for v in verdicts)
                results.append({
                    "id": case["id"], "kind": case["kind"], "attempt": attempt,
                    "output": slim(output),
                    "checks": [{"name": v.name, "passed": v.passed, "detail": v.detail, "data": v.data}
                               for v in verdicts],
                    "passed": passed,
                })
                mark = "ок" if passed else "НІ"
                tokens = report.tokens_of(output)
                why = "" if passed else "; ".join(f"{v.name}: {v.detail}" for v in verdicts if not v.passed)
                print(f"{case['id']:<6} {case['kind']:<12} {attempt}/{args.k}  {mark}  "
                      f"{report.fmt(output.get('elapsed'), 1):>5} с  {report.fmt(tokens):>5} ток  {why[:110]}")
    except KeyboardInterrupt:
        print("\nперервано; зберігаю те, що є")

    run = {
        "meta": {
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "label": args.label,
            "target": args.target,
            "config": config,
            "cases_file": str(cases_path),
            "cases_sha256": file_sha256(cases_path),
            "k": args.k,
            "judge": not args.no_judge,
            "filters": {"ids": args.ids, "kinds": args.kinds},
            "python": platform.python_version(),
            "duration_s": round(time.perf_counter() - started_all, 1),
            "partial": len(results) < len(cases) * args.k,
        },
        "cases": [{"id": c["id"], "kind": c["kind"], "about": c.get("about", "")} for c in cases],
        "results": results,
    }
    run["summary"] = report.summarize(run)
    report.print_summary(run["summary"], args.k)
    report.print_failed(run, limit=30)

    out_path.write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nзаписано {out_path}  ({run['meta']['duration_s']} с)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
