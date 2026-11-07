"""Прогін замовлень через API застосунку — щоб порівняти роботу з кешем і
без нього на одному й тому самому потоці.

Запуск із папки pr11 при запущеному застосунку:

    python eval/replay.py --orders 20 --seed 1 --out eval/replay-nocache.json
    python eval/replay.py --orders 20 --seed 1 --out eval/replay-cache.json
    python eval/replay.py --compare eval/replay-nocache.json eval/replay-cache.json

Той самий `--seed` дає ту саму послідовність звернень із повторами:
приблизно половина замовлень повторює одне з попередніх. Скрипт
замовляє чернетки по черзі (або по `--concurrency` одночасно), чекає
завершення кожної, збирає знімки `GET /api/drafts/{id}` і підсумовує.
На 429 чекає `Retry-After` і повторює. Ліміт на клієнта в `.env` на час
прогону підніміть, інакше двадцять замовлень розтягнуться на хвилини.

Це готовий інструмент, а не частина завдання: він лише ходить у ваш
API. Поля знімка, яких у вашій реалізації немає, показуються порожніми.
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TERMINAL = {"done", "failed", "cancelled"}


def api(base: str, method: str, path: str, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
            return response.status, (json.loads(payload) if payload else {}), dict(response.headers)
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        try:
            parsed = json.loads(payload) if payload else {}
        except ValueError:
            parsed = {"raw": payload.decode("utf-8", "replace")[:200]}
        return exc.code, parsed, dict(exc.headers)


def make_sequence(ids: list[str], orders: int, seed: int, repeat: float) -> list[str]:
    """Послідовність звернень із повторами, однакова для однакового seed."""
    rng = random.Random(seed)
    sequence: list[str] = []
    for _ in range(orders):
        if sequence and rng.random() < repeat:
            sequence.append(rng.choice(sequence[-6:]))
        else:
            sequence.append(rng.choice(ids))
    return sequence


def order(base: str, request_id: str, client: str, fresh: bool, timeout: float) -> dict:
    """Замовити чернетку, дочекатися кінцевого статусу, повернути знімок."""
    started = time.perf_counter()
    while True:
        status, body, headers = api(base, "POST", "/api/drafts",
                                    {"request_id": request_id, "client": client, "fresh": fresh})
        if status == 429:
            wait = float(headers.get("Retry-After") or 5)
            print(f"  429 для {request_id}: чекаю {wait:.0f} с", file=sys.stderr)
            time.sleep(wait)
            continue
        if status >= 300:
            return {"request_id": request_id, "status": f"http {status}", "error": body,
                    "wall": time.perf_counter() - started}
        break

    draft_id = body.get("draft_id")
    snapshot = body
    deadline = time.perf_counter() + timeout
    while snapshot.get("status") not in TERMINAL and time.perf_counter() < deadline:
        time.sleep(0.25)
        status, snapshot, _ = api(base, "GET", f"/api/drafts/{draft_id}")
        if status >= 300:
            snapshot = {"status": f"http {status}", "error": snapshot}
            break
    snapshot["wall"] = time.perf_counter() - started
    snapshot.setdefault("request_id", request_id)
    return snapshot


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def summarize(rows: list[dict]) -> dict:
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
    done = [r for r in rows if r.get("status") == "done"]
    cached = [r for r in done if r.get("cached")]
    walls = [r["wall"] for r in rows if "wall" in r]
    firsts = [r["elapsed"]["first_token"] for r in rows
              if isinstance(r.get("elapsed"), dict) and isinstance(r["elapsed"].get("first_token"), (int, float))]
    costs = [r["cost"] for r in rows if isinstance(r.get("cost"), (int, float))]
    tokens_in = sum((r.get("usage") or {}).get("prompt_tokens", 0) or 0 for r in rows if isinstance(r.get("usage"), dict))
    tokens_out = sum((r.get("usage") or {}).get("completion_tokens", 0) or 0 for r in rows if isinstance(r.get("usage"), dict))
    attempts = sum(r.get("attempts") or 0 for r in rows if not r.get("cached"))
    return {
        "orders": len(rows),
        "statuses": statuses,
        "cached_hits": len(cached),
        "provider_attempts": attempts,
        "fallbacks": sum(1 for r in rows if r.get("fallback")),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_total": round(sum(costs), 6) if costs else None,
        "cost_known_for": len(costs),
        "wall_p50": percentile(walls, 0.5),
        "wall_p95": percentile(walls, 0.95),
        "first_token_p50": percentile(firsts, 0.5),
        "first_token_p95": percentile(firsts, 0.95),
    }


def fmt(value, digits=2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def print_rows(rows: list[dict]) -> None:
    print(f"{'#':>3} {'зверн.':<7} {'чернетка':<9} {'статус':<10} {'пров.':<7} {'спроб':>5} "
          f"{'кеш':<4} {'$':>9} {'перший, с':>10} {'разом, с':>9}")
    for i, r in enumerate(rows, 1):
        elapsed = r.get("elapsed") if isinstance(r.get("elapsed"), dict) else {}
        print(f"{i:>3} {r.get('request_id', ''):<7} {str(r.get('draft_id', '')):<9} {str(r.get('status')):<10} "
              f"{str(r.get('provider') or ''):<7} {fmt(r.get('attempts')):>5} "
              f"{'так' if r.get('cached') else '':<4} {fmt(r.get('cost'), 6):>9} "
              f"{fmt(elapsed.get('first_token')):>10} {fmt(r.get('wall')):>9}")


def print_summary(summary: dict, label: str = "") -> None:
    if label:
        print(f"\n== {label}")
    print(f"замовлень {summary['orders']}; статуси {summary['statuses']}; "
          f"з кешу {summary['cached_hits']}; звертань до провайдера (сума спроб) {summary['provider_attempts']}; "
          f"запасний варіант {summary['fallbacks']}")
    print(f"токенів: вхід {summary['tokens_in']}, вихід {summary['tokens_out']}; "
          f"вартість разом {fmt(summary['cost_total'], 6)} $ (відома для {summary['cost_known_for']} замовлень)")
    print(f"тривалість, с: медіана {fmt(summary['wall_p50'])}, p95 {fmt(summary['wall_p95'])}; "
          f"до першого фрагмента: медіана {fmt(summary['first_token_p50'])}, p95 {fmt(summary['first_token_p95'])}")


def compare(paths: list[str]) -> None:
    runs = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            runs.append((path, json.load(f)))
    keys = ["orders", "cached_hits", "provider_attempts", "fallbacks", "tokens_in", "tokens_out",
            "cost_total", "wall_p50", "wall_p95", "first_token_p50", "first_token_p95"]
    width = max(len(p) for p, _ in runs)
    print(f"{'':<18}" + "".join(f"{p:>{width + 2}}" for p, _ in runs))
    for key in keys:
        print(f"{key:<18}" + "".join(f"{fmt(run['summary'].get(key), 6 if key == 'cost_total' else 2):>{width + 2}}"
                                     for _, run in runs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--orders", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=float, default=0.5, help="частка замовлень, що повторюють попередні")
    parser.add_argument("--client", default="replay")
    parser.add_argument("--fresh", action="store_true", help="усі замовлення з fresh=true (оминути кеш)")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120, help="скільки секунд чекати на одну чернетку")
    parser.add_argument("--out", help="куди записати знімки й підсумок (JSON)")
    parser.add_argument("--compare", nargs="+", metavar="FILE", help="порівняти збережені прогони")
    args = parser.parse_args()

    if args.compare:
        compare(args.compare)
        return 0

    status, samples, _ = api(args.base, "GET", "/api/requests")
    if status != 200 or not samples:
        print(f"не вдалося отримати зразки: {status} {samples}", file=sys.stderr)
        return 1
    ids = [s["id"] for s in samples]
    sequence = make_sequence(ids, args.orders, args.seed, args.repeat)
    print(f"послідовність ({args.orders}, seed {args.seed}): {' '.join(sequence)}\n")

    status, health, _ = api(args.base, "GET", "/api/health")
    started = time.perf_counter()
    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda rid: order(args.base, rid, args.client, args.fresh, args.timeout), sequence))
    else:
        rows = [order(args.base, rid, args.client, args.fresh, args.timeout) for rid in sequence]
    total = time.perf_counter() - started

    print_rows(rows)
    summary = summarize(rows)
    summary["wall_total"] = total
    print_summary(summary)
    print(f"увесь прогін: {total:.1f} с")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "health": health if status == 200 else None,
                       "sequence": sequence, "rows": rows, "summary": summary}, f, ensure_ascii=False, indent=2)
        print(f"записано {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
