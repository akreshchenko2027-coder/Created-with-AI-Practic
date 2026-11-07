"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:

    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow

Читання файлу, відбір за датою і допоміжні функції готові. Розділи
звіту — заготовки: поля, за якими вони працюють, — ваші, бо назви
подій і полів ви обрали, коли писали записи в шлюзі й фонових задачах.
`summary` працює на будь-якому журналі й показує, що в ньому взагалі є.

Орієнтир для кожного розділу — питання, на яке він відповідає; воно в
докстрінгу. Відповідь — числа й кілька рядків журналу, а не переказ.
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines. Рядки, які не розбираються, рахуються й
    пропускаються; `since` — дата або час у форматі ISO, від якого брати
    записи (порівнюється з полем `ts` як рядок)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для
    порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    """Згрупувати записи за значенням поля."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі: скільки записів, за який час, яких подій і
    рівнів скільки. Працює на будь-якому журналі."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    """Які виклики найповільніші й що таке «повільно» в цих числах?

    Розподіл тривалості звертань до провайдера (медіана, p95, максимум),
    а не середнє; десятка найдовших із ідентифікаторами, провайдером,
    моделлю, спробою; де пройшов час — у черзі, до першого фрагмента чи
    в генерації; розріз за провайдером і годиною.
    """
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    durations = [float(r["duration_ms"]) for r in attempts if _number(r.get("duration_ms"))]
    if not durations:
        print("звертань до провайдера з тривалістю немає")
        return
    print(f"спроб: {len(durations)}; медіана {_fmt(percentile(durations, .5), 1)} мс; "
          f"p95 {_fmt(percentile(durations, .95), 1)} мс; максимум {max(durations):.1f} мс")
    for label, field in (("черга", "queue_ms"), ("до першого фрагмента", "first_token_ms")):
        values = [float(r[field]) for r in attempts if _number(r.get(field))]
        if values:
            print(f"{label}: медіана {_fmt(percentile(values, .5), 1)} мс; "
                  f"p95 {_fmt(percentile(values, .95), 1)} мс")
    print("за провайдером:")
    for provider, rows in sorted(group(attempts, "provider").items(), key=lambda item: str(item[0])):
        values = [float(r["duration_ms"]) for r in rows if _number(r.get("duration_ms"))]
        if values:
            print(f"  {provider}: {len(values)}; медіана {_fmt(percentile(values, .5), 1)} мс; "
                  f"p95 {_fmt(percentile(values, .95), 1)} мс")
    print("найдовші:")
    for row in sorted(attempts, key=lambda r: _number(r.get("duration_ms")) or 0, reverse=True)[:10]:
        print(f"  {row.get('duration_ms')} мс; draft={row.get('draft_id')}; "
              f"attempt={row.get('attempt')}; provider={row.get('provider')}; model={row.get('model')}; "
              f"queue={row.get('queue_ms')} мс; first_token={row.get('first_token_ms')} мс")
    hours = Counter(str(r.get("ts", ""))[:13] for r in attempts if r.get("ts"))
    if hours:
        print("за годиною:", dict(sorted(hours.items())))


def failed(records: list[dict]) -> None:
    """Що й чому не вдалося?

    Невдалі спроби за видом збою; остаточні збої чернеток; частка
    чернеток із невдалою першою спробою, які все ж завершилися успіхом
    (чи окупилися повтори); вартість спроб, які нічого не дали.
    """
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    failures = [r for r in attempts if r.get("status") in {"failed", "rejected"}]
    by_kind = Counter(r.get("kind") or "unknown" for r in failures)
    print(f"невдалі/відхилені спроби: {len(failures)}")
    print("за видом:", dict(by_kind) if by_kind else "немає")
    drafts = [r for r in records if r.get("event") == "draft.failed"]
    print(f"чернеток завершилися помилкою: {len(drafts)}")
    for row in drafts:
        print(f"  draft={row.get('draft_id')}; kind={row.get('kind')}; "
              f"attempts={row.get('attempts')}; cost={_fmt(row.get('cost'), 8)} $")
    by_draft: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        if row.get("draft_id"):
            by_draft[str(row["draft_id"])].append(row)
    completed = {str(r.get("draft_id")) for r in records if r.get("event") == "draft.completed"}
    recovered = [
        rows for draft_id, rows in by_draft.items()
        if draft_id in completed and any(r.get("status") != "ok" for r in rows)
    ]
    print(f"чернеток із невдалою спробою, завершених повтором: {len(recovered)}")
    wasted = sum(float(r["cost"]) for r in failures if _number(r.get("cost")))
    unknown = sum(1 for r in failures if r.get("cost") is None)
    print(f"вартість невдалих спроб: {wasted:.8f} $; без вартості: {unknown}")


def repeated(records: list[dict]) -> None:
    """Що ми робили двічі?

    Той самий ключ кешу кілька разів за короткий час (кандидати на кеш
    і питання «чому»); чернетки з кількома спробами (повтори шлюзу);
    клієнти з серіями замовлень за хвилину.
    """
    attempts = [r for r in records if r.get("event") == "llm.attempt" and r.get("cache_key")]
    keys: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        keys[str(row["cache_key"])].append(row)
    repeated_keys = [(key, rows) for key, rows in keys.items() if len(rows) > 1]
    print(f"ключів кешу, за якими провайдер викликався повторно: {len(repeated_keys)}")
    for key, rows in sorted(repeated_keys, key=lambda item: len(item[1]), reverse=True)[:10]:
        drafts = sorted({str(r.get("draft_id")) for r in rows if r.get("draft_id")})
        print(f"  {len(rows)} спроб; key={key[:12]}; drafts={', '.join(drafts)}")
    by_draft: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        if row.get("draft_id"):
            by_draft[str(row["draft_id"])].append(row)
    retried = [(draft_id, rows) for draft_id, rows in by_draft.items() if len(rows) > 1]
    print(f"чернеток із кількома спробами: {len(retried)}")
    for draft_id, rows in sorted(retried, key=lambda item: len(item[1]), reverse=True)[:10]:
        print(f"  draft={draft_id}; спроб={len(rows)}; "
              f"статуси={','.join(str(r.get('status')) for r in rows)}")
    per_minute: dict[tuple[str, str], int] = Counter()
    for row in records:
        if row.get("event") != "draft.created" or not row.get("client") or not row.get("ts"):
            continue
        per_minute[(str(row["client"]), str(row["ts"])[:16])] += 1
    bursts = [(key, count) for key, count in per_minute.items() if count > 1]
    print(f"серій замовлень клієнта за хвилину: {len(bursts)}")
    for (client, minute), count in sorted(bursts, key=lambda item: item[1], reverse=True)[:10]:
        print(f"  client={client}; хвилина={minute}; замовлень={count}")


def spend(records: list[dict]) -> None:
    """Скільки це коштувало?

    Вартість і токени за день, за модель, за клієнта; середня вартість
    чернетки; частка вартості, яка припала на спроби зі збоєм і на
    запасний варіант; скільки записів без вартості або з оцінкою.
    """
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    costs_known = [float(r["cost"]) for r in attempts if _number(r.get("cost"))]
    estimated = sum(bool(r.get("cost_estimated")) for r in attempts)
    missing = sum(r.get("cost") is None for r in attempts)
    print(f"спроб: {len(attempts)}; відома вартість: {sum(costs_known):.8f} $; "
          f"оцінок: {estimated}; без вартості: {missing}")
    for label, key in (("за днем", "ts"), ("за моделлю", "model"), ("за клієнтом", "client")):
        totals: dict[str, float] = defaultdict(float)
        for row in attempts:
            amount = _number(row.get("cost"))
            if amount is not None:
                value = str(row.get(key) or "невідомо")
                totals[value[:10] if key == "ts" else value] += amount
        print(f"{label}:")
        for name, amount in sorted(totals.items()):
            print(f"  {name}: {amount:.8f} $")
    failed_cost = sum(
        float(r["cost"]) for r in attempts
        if r.get("status") != "ok" and _number(r.get("cost")) is not None
    )
    fallback_cost = sum(
        float(r["cost"]) for r in attempts
        if r.get("fallback") and _number(r.get("cost")) is not None
    )
    drafts = [r for r in records if r.get("event") == "draft.completed" and _number(r.get("cost")) is not None]
    average = sum(float(r["cost"]) for r in drafts) / len(drafts) if drafts else None
    print(f"невдалі спроби: {failed_cost:.8f} $; запасний варіант: {fallback_cost:.8f} $; "
          f"середня вартість завершеної чернетки: {_fmt(average, 8)} $")


def cache_report(records: list[dict]) -> None:
    """Що дав кеш?

    Влучання, промахи, обходи (`fresh`) і частка влучань; скільки
    звертань до провайдера й грошей заощаджено; які звернення
    повторювалися найчастіше.
    """
    hits = [r for r in records if r.get("event") == "cache.hit"]
    misses = [r for r in records if r.get("event") == "cache.miss"]
    bypassed = [r for r in records if r.get("event") == "cache.bypass"]
    lookups = len(hits) + len(misses)
    rate = len(hits) / lookups * 100 if lookups else 0
    saved_cost = sum(float(r["cost_saved"]) for r in hits if _number(r.get("cost_saved")) is not None)
    saved_tokens = sum(int(r.get("tokens_saved") or 0) for r in hits if _number(r.get("tokens_saved")) is not None)
    provider_calls = sum(
        1 for r in records if r.get("event") == "llm.attempt" and r.get("status") != "rejected"
    )
    print(f"влучань: {len(hits)}; промахів: {len(misses)}; обходів fresh: {len(bypassed)}; "
          f"частка влучань: {rate:.1f}%")
    print(f"звертань до провайдера: {provider_calls}; збережено за оцінкою {saved_cost:.8f} $ "
          f"і {saved_tokens} токенів")
    frequencies = Counter(
        str(r.get("cache_key")) for r in records
        if r.get("event") in {"cache.hit", "cache.miss", "cache.bypass"} and r.get("cache_key")
    )
    repeated_keys = [(key, count) for key, count in frequencies.items() if count > 1]
    print("найчастіші ключі:")
    for key, count in sorted(repeated_keys, key=lambda item: item[1], reverse=True)[:10]:
        print(f"  {count} звернень; key={key[:12]}")


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _fmt(value, digits: int = 2) -> str:
    number = _number(value)
    return "—" if number is None else f"{number:.{digits}f}"


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        SECTIONS[name](records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
