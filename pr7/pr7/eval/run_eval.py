"""Run and compare the invoice samples against samples/expected.json."""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
EVAL = ROOT / "eval"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CRITICAL_FIELDS = {"supplier.iban", "supplier.code", "total"}
APOSTROPHES = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u02bc": "'", "\u0060": "'",
})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="baseline", help="Name used for this run's result folder")
    parser.add_argument("--sample", help="Run one sample path, e.g. clean/rahunok-02.png")
    parser.add_argument("--max-side", type=int, help="Override IMAGE_MAX_SIDE for this run")
    parser.add_argument("--compare-to", help="Compare this run with an earlier run-id")
    parser.add_argument("--force", action="store_true", help="Call the model again for cached samples")
    parser.add_argument(
        "--request-delay",
        type=float,
        default=6,
        help="Seconds between model requests (default: 6, to stay within free-tier rate limits)",
    )
    return parser.parse_args()


def flatten(value: object, prefix: str = "") -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            result.update(flatten(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            result.update(flatten(child, path))
    else:
        result[prefix] = value
    return result


def normalized(value: object, path: str) -> object:
    if value is None:
        return None
    if path == "supplier.iban":
        return re.sub(r"\s+", "", str(value)).upper()
    if path.rsplit(".", 1)[-1] in {"price", "amount", "total", "vat", "total_without_vat"}:
        text = str(value).replace("\u00a0", "").replace(" ", "").replace(",", ".")
        try:
            return Decimal(text).quantize(Decimal("0.01"))
        except InvalidOperation:
            return value
    if isinstance(value, str):
        return " ".join(value.translate(APOSTROPHES).split()).casefold()
    return value


def compare_fields(expected: dict, actual: dict) -> dict[str, dict]:
    expected_flat = flatten(expected)
    actual_flat = flatten(actual)
    comparisons: dict[str, dict] = {}
    for path in sorted(expected_flat.keys() | actual_flat.keys()):
        want = expected_flat.get(path)
        got = actual_flat.get(path)
        if path not in expected_flat:
            status = "fabrication"
        elif want is None:
            status = "correct" if path in actual_flat and got is None else "fabrication"
        elif path not in actual_flat or got is None:
            status = "omission"
        elif normalized(want, path) == normalized(got, path):
            status = "correct"
        else:
            status = "error"
        comparisons[path] = {"expected": want, "extracted": got, "status": status}
    return comparisons


def issue_catches(path: str, issues: list[dict]) -> bool:
    for issue in issues:
        issue_path = issue.get("field", "")
        if path == issue_path or path.startswith(issue_path + ".") or issue_path.startswith(path + "."):
            return True
        path_parts = path.split(".")
        issue_parts = issue_path.split(".")
        if (len(path_parts) >= 2 and len(issue_parts) >= 2
                and path_parts[0] == issue_parts[0] == "items"
                and path_parts[1] == issue_parts[1]
                and issue.get("rule", "").startswith("arithmetic_")):
            return True
    return False


def measure(records: list[dict]) -> dict:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_variant[record["variant"]].append(record)

    def summarize(group: list[dict]) -> dict:
        statuses: Counter = Counter()
        critical: Counter = Counter()
        caught = 0
        silent = 0
        matched_decisions = 0
        prompt_tokens = 0
        response_tokens = 0
        elapsed = 0.0
        matrix: Counter = Counter()
        completed = 0
        for record in group:
            if "error" in record:
                continue
            completed += 1
            actual_decision = record["decision"]
            expected_decision = record["expected_decision"]
            matched_decisions += actual_decision == expected_decision
            matrix[f"{expected_decision}->{actual_decision}"] += 1
            prompt_tokens += record.get("usage", {}).get("prompt_tokens") or 0
            response_tokens += record.get("usage", {}).get("completion_tokens") or 0
            elapsed += record.get("elapsed", {}).get("total") or 0
            for path, comparison in record["fields"].items():
                status = comparison["status"]
                statuses[status] += 1
                if path in CRITICAL_FIELDS:
                    critical[status] += 1
                if status in {"error", "omission", "fabrication"}:
                    was_caught = comparison.get("caught_by_rules", False)
                    caught += was_caught
                    silent += actual_decision == "auto" and not was_caught
        total_fields = sum(statuses.values())
        total_critical = sum(critical.values())
        return {
            "files": len(group),
            "completed": completed,
            "failed": len(group) - completed,
            "correct": statuses["correct"],
            "errors": statuses["error"],
            "omissions": statuses["omission"],
            "fabrications": statuses["fabrication"],
            "caught_by_rules": caught,
            "silent_errors": silent,
            "decision_matches": matched_decisions,
            "critical_correct": critical["correct"],
            "critical_fields": total_critical,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": response_tokens,
            "elapsed_seconds": round(elapsed, 2),
            "field_accuracy": round(statuses["correct"] / total_fields, 4) if total_fields else None,
            "critical_accuracy": round(critical["correct"] / total_critical, 4) if total_critical else None,
            "decision_matrix": dict(matrix),
        }

    return {
        "variants": {variant: summarize(group) for variant, group in sorted(by_variant.items())},
        "overall": summarize(records),
    }


def write_findings(run_id: str, compare_id: str | None) -> None:
    run_dir = EVAL / "results" / run_id
    current = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    current_records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    lines = [
        "# Висновки ПР7",
        "",
        f"Прогін `{run_id}`; результати й сирі відповіді — у `results/{run_id}/`.",
        f"Успішно оброблено: {current['overall']['completed']}; "
        f"помилок запиту: {current['overall']['failed']}.",
        "Нормалізація: пробіли в IBAN прибрані, регістр IBAN не враховується; "
        "суми порівнюються як Decimal до копійок; пробіли й різновиди апострофа "
        "в тексті уніфіковано. Латинські й кириличні літери навмисно не ототожнюються.",
        "Зображення не копіюються до звіту; локальні файли результатів містять сирі "
        "відповіді з реквізитами й мають залишатися захищеними.",
        "",
        "| Варіант | Файлів | Правильно | Помилок | Пропусків | Вигадок | Спіймано | Тихих помилок | Рішення збіглося | Токени запиту |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, stat in current["variants"].items():
        lines.append(
            f"| {variant} | {stat['files']} | {stat['correct']} | {stat['errors']} | "
            f"{stat['omissions']} | {stat['fabrications']} | {stat['caught_by_rules']} | "
            f"{stat['silent_errors']} | {stat['decision_matches']}/{stat['completed']} | "
            f"{stat['prompt_tokens']} |"
        )
    lines.extend([
        "",
        "### Критичні поля",
        "",
        "| Варіант | Правильно | Усього | Частка |",
        "|---|---:|---:|---:|",
    ])
    for variant, stat in current["variants"].items():
        accuracy = stat["critical_accuracy"]
        lines.append(
            f"| {variant} | {stat['critical_correct']} | {stat['critical_fields']} | "
            f"{accuracy if accuracy is not None else 'немає даних'} |"
        )
    overall = current["overall"]
    lines.extend([
        "",
        f"Критичні поля: правильно {overall['critical_correct']} із {overall['critical_fields']} "
        f"({overall['critical_accuracy'] if overall['critical_accuracy'] is not None else 'немає даних'}).",
        f"Час моделі разом: {overall['elapsed_seconds']} с; токени відповіді: "
        f"{overall['completion_tokens']}.",
        "",
        "## Матриця рішень",
        "",
        "| Очікуване → фактичне | Кількість |",
        "|---|---:|",
    ])
    for decision, count in sorted(overall["decision_matrix"].items()):
        lines.append(f"| {decision} | {count} |")
    lines.extend([
        "",
        f"Зайва ручна перевірка (очікувалося auto, отримано review): "
        f"{overall['decision_matrix'].get('auto->review', 0)}; "
        f"проблему пропущено (очікувалося review, отримано auto): "
        f"{overall['decision_matrix'].get('review->auto', 0)}.",
    ])

    by_path = {record["path"]: record for record in current_records}

    def field_status(path: str, field: str) -> str:
        record = by_path.get(path)
        if not record or "error" in record:
            return "не виконано"
        comparison = record.get("fields", {}).get(field)
        if not comparison:
            return "поле відсутнє"
        caught = issue_catches(field, record.get("issues", []))
        return f"{comparison['status']}; {'спіймано' if caught else 'не спіймано'}"

    lines.extend([
        "",
        "## Відповіді на контрольні питання",
        "",
        f"- Найнижча точність за варіантами: {min(current['variants'], key=lambda k: current['variants'][k]['field_accuracy'] if current['variants'][k]['field_accuracy'] is not None else 1)}. "
        "Зіставте розподіл помилок, пропусків і вигадок у таблиці, щоб визначити, "
        "чи модель помилялася впевнено, чи утримувалася від відповіді.",
        f"- Обрізаний `rahunok-03`: суми — {field_status('degraded/rahunok-03-crop.jpg', 'total')}; "
        f"строк дії — {field_status('degraded/rahunok-03-crop.jpg', 'valid_until')}.",
        f"- `rahunok-04`, де немає строку дії: {field_status('clean/rahunok-04.png', 'valid_until')}.",
        f"- Помилкова сума в `rahunok-05`: {field_status('clean/rahunok-05.png', 'items.1.amount')}.",
        f"- Переставлені цифри IBAN у `rahunok-06`: {field_status('clean/rahunok-06.png', 'supplier.iban')}.",
        f"- Текст-інструкція в `rahunok-08`: рішення {by_path.get('clean/rahunok-08.png', {}).get('decision', 'не виконано')}; "
        "у цій архітектурі рішення визначає код перевірок, а не модель.",
        f"- Тихих помилок у `auto`: {overall['silent_errors']}. Додаткові правила та "
        "хибні тривоги оцінюйте за конкретними полями й результатами чистих зразків.",
        "",
    ])
    if compare_id:
        prior_path = EVAL / "results" / compare_id / "summary.json"
        if prior_path.exists():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            before, after = prior["overall"], current["overall"]
            lines.extend([
                f"## Зміна: `{compare_id}` → `{run_id}`",
                "",
                f"- Правильно: {before['correct']} → {after['correct']} полів.",
                f"- Тихі помилки: {before['silent_errors']} → {after['silent_errors']}.",
                f"- Токени запиту: {before['prompt_tokens']} → {after['prompt_tokens']}.",
                f"- Час: {before['elapsed_seconds']} → {after['elapsed_seconds']} с.",
                "",
            ])
        else:
            lines.extend(["", f"Порівняння `{compare_id}` ще не знайдено; запустіть цей прогін повторно з `--compare-to`.", ""])
    lines.extend([
        "## Межа для повторного знімка",
        "",
        "Робоче правило: не передавати в автоматичний реєстр документ, якщо критичне "
        "поле не прочитане, перевірка IBAN/довідника не пройдена або модельне "
        "вилучення не відповідає контрольній арифметиці. Просити перезняти варто, "
        "коли потрібні реквізити фізично не видно (обрізання/розмиття); точний "
        "поріг якості зображення слід встановити за цими вимірами, а не припускати.",
        "",
    ])
    (EVAL / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise SystemExit("--run-id може містити лише латинські літери, цифри, крапку, _ та -.")
    if args.request_delay < 0:
        raise SystemExit("--request-delay не може бути відʼємним.")
    if args.max_side is not None:
        if args.max_side < 1:
            raise SystemExit("--max-side має бути додатним числом.")
        os.environ["IMAGE_MAX_SIDE"] = str(args.max_side)

    sys.path.insert(0, str(ROOT))
    from app.extraction import process
    from app.images import ImageError
    from app.llm import LLMError

    expected = json.loads((SAMPLES / "expected.json").read_text(encoding="utf-8"))
    local_expected = SAMPLES / "expected.own.local.json"
    if local_expected.exists():
        expected.update(json.loads(local_expected.read_text(encoding="utf-8")))
    paths = sorted(
        path for folder in SAMPLES.iterdir() if folder.is_dir()
        for path in folder.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    own_paths = [path for path in paths if path.is_relative_to(SAMPLES / "own")]
    if len(own_paths) < 2:
        raise SystemExit(
            "Додайте щонайменше два власні зображення в samples/own/: "
            "один варіант degrade.py та одне фото рахунку, надрукованого або відкритого на екрані."
        )
    items = []
    for image_path in paths:
        relative = image_path.relative_to(SAMPLES).as_posix()
        if relative not in expected:
            raise SystemExit(f"Для {relative} немає очікуваних значень у samples/expected.json.")
        items.append((image_path, relative, expected[relative]))
    if args.sample:
        items = [item for item in items if item[1] == args.sample]
        if not items:
            raise SystemExit(f"Зразок {args.sample!r} не знайдено або для нього немає очікуваних значень.")
    if not items:
        raise SystemExit("У samples/ немає зображень для оцінювання.")

    run_dir = EVAL / "results" / args.run_id
    records_dir = run_dir / "files"
    records_dir.mkdir(parents=True, exist_ok=True)
    records = []
    last_request = None
    for image_path, relative, truth in items:
        record_path = records_dir / Path(relative).with_suffix(".json")
        if record_path.exists() and not args.force:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if "error" in record:
                print(f"[retry] {relative}")
                record = None
            else:
                print(f"[cache] {relative}")
        else:
            record = None

        if record is None:
            print(f"[run] {relative}")
            if last_request is not None:
                time.sleep(max(0, args.request_delay - (time.monotonic() - last_request)))
            try:
                last_request = time.monotonic()
                result = process(image_path.read_bytes())
                issues = [issue.__dict__ for issue in result.issues]
                actual_checks = {issue["rule"] for issue in issues}
                expected_checks = set(truth.get("expected_checks", []))
                fields = compare_fields(truth["fields"], result.document or {})
                for path, comparison in fields.items():
                    if comparison["status"] in {"error", "omission", "fabrication"}:
                        comparison["caught_by_rules"] = issue_catches(path, issues)
                record = {
                    "path": relative,
                    "variant": "own" if relative.startswith("own/") else truth.get("variant", "unknown"),
                    "expected_decision": truth["expected_decision"],
                    "decision": result.decision,
                    "expected_checks": sorted(expected_checks),
                    "actual_checks": sorted(actual_checks),
                    "checks_match": expected_checks <= actual_checks,
                    "missing_expected_checks": sorted(expected_checks - actual_checks),
                    "unexpected_checks": sorted(actual_checks - expected_checks),
                    "issues": issues,
                    "fields": fields,
                    "image": result.image,
                    "elapsed": result.elapsed,
                    "usage": result.usage or {},
                    "model": result.model,
                    "raw_response": result.raw_response,
                }
            except (ImageError, LLMError) as exc:
                expected_checks = sorted(truth.get("expected_checks", []))
                record = {
                    "path": relative,
                    "variant": "own" if relative.startswith("own/") else truth.get("variant", "unknown"),
                    "expected_decision": truth["expected_decision"],
                    "decision": None,
                    "expected_checks": expected_checks,
                    "actual_checks": [],
                    "checks_match": None,
                    "missing_expected_checks": expected_checks,
                    "unexpected_checks": [],
                    "issues": [],
                    "fields": {},
                    "raw_response": exc.raw_response,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        records.append(record)

    summary = measure(records)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print(f"Збережено {len(records)} записів: {run_dir}")
    print(f"Завершено успішно: {summary['overall']['completed']}/{len(records)}")
    if summary["overall"]["failed"]:
        print(f"Помилок обробки: {summary['overall']['failed']}; повторний запуск спробує їх ще раз.")
    if args.compare_to and not summary["overall"]["failed"]:
        write_findings(args.run_id, args.compare_to)
        print(f"Висновки оновлено: {EVAL / 'findings.md'}")
    elif args.compare_to:
        print("Висновки не оновлено: спершу потрібен повний успішний прогін.")
    return 1 if summary["overall"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
