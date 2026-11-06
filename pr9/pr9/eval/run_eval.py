import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from app import assistant, workflow
from shop import service


def load_requests(path: Path) -> list[dict]:
    requests = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(requests, list) or not requests:
        raise ValueError("Набір звернень має бути непорожнім JSON-масивом.")
    identifiers = {item["id"] for item in requests}
    if len(identifiers) != len(requests):
        raise ValueError("Ідентифікатори звернень мають бути унікальними.")
    for item in requests:
        if item.get("after") and item["after"] not in identifiers:
            raise ValueError(f"{item['id']}: невідоме попереднє звернення {item['after']}.")
    return requests


def order_requests(requests: list[dict]) -> list[dict]:
    children = {}
    for item in requests:
        if item.get("after"):
            children.setdefault(item["after"], []).append(item)
    ordered = []
    seen = set()

    def add(item: dict) -> None:
        if item["id"] in seen:
            return
        seen.add(item["id"])
        ordered.append(item)
        for child in children.get(item["id"], []):
            add(child)

    for request in requests:
        if not request.get("after"):
            add(request)
    if len(ordered) != len(requests):
        raise ValueError("Не вдалося впорядкувати звернення з полем after.")
    return ordered


def shop_snapshot(customer_id: str) -> dict:
    failure_rate = os.environ.get("SHOP_FAILURE_RATE", "0")
    os.environ["SHOP_FAILURE_RATE"] = "0"
    try:
        returns = [
            {
                key: item[key]
                for key in ("return_id", "order_id", "sku", "quantity", "reason", "status")
            }
            for item in service.list_returns()
        ]
        orders = service.list_orders(customer_id)
        return {
            "returns": returns,
            "orders": [
                {"order_id": item["order_id"], "status": item["status_label"]}
                for item in orders
            ],
        }
    finally:
        os.environ["SHOP_FAILURE_RATE"] = failure_rate


def tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    total = usage.get("total_tokens")
    if total is not None:
        return total
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    return prompt + completion


def save_results(path: Path, results: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def execute(request: dict, mode: str, repeat: int, reject_id: str) -> dict:
    before = shop_snapshot(request["customer"])
    started = time.perf_counter()
    error = None
    calls = 0
    used_tokens = None
    reply = None
    path = []
    status = "error"
    actual_route = None
    stopped = None
    try:
        if mode == "workflow":
            run = workflow.start(request["text"], request["customer"])
            if run.status == "awaiting_approval":
                approve = not (request["id"] == reject_id and repeat == 1)
                run = workflow.resume(
                    run.run_id,
                    approve,
                    "eval: перевірка відхилення" if not approve else "eval: підтвердження",
                )
            status = run.status
            actual_route = run.route
            stopped = run.stopped
            reply = run.reply
            path = [
                {"name": step.name, "kind": step.kind, "status": step.status}
                for step in run.steps
            ]
            calls = run.usage.get("model_calls", 0)
            used_tokens = tokens(run.usage)
        else:
            result = assistant.answer(request["text"], request["customer"])
            status = "answer" if result.stopped == "answer" else result.stopped
            stopped = result.stopped
            reply = result.text
            path = [
                {
                    "name": call.name,
                    "status": call.status,
                    "arguments": call.arguments,
                }
                for call in result.calls
            ]
            calls = result.rounds
            used_tokens = tokens(result.usage)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    after = shop_snapshot(request["customer"])
    return {
        "request_id": request["id"],
        "kind": request["kind"],
        "mode": mode,
        "repeat": repeat,
        "route": actual_route,
        "expected_route": request.get("expect_route"),
        "status": status,
        "stopped": stopped,
        "path": path,
        "shop_before": before,
        "shop_after": after,
        "reply": reply,
        "model_calls": calls,
        "tokens": used_tokens,
        "elapsed_seconds": round(elapsed, 3),
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Порівняння workflow й агента на наборі звернень ПР9.")
    parser.add_argument("--requests", type=Path, default=HERE / "eval" / "requests.json")
    parser.add_argument("--output", type=Path, default=HERE / "eval" / "results.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids", help="Список ID через кому для вибіркового прогону.")
    parser.add_argument("--mode", choices=("workflow", "agent", "both"), default="both")
    parser.add_argument("--failure-rate", type=float)
    parser.add_argument("--reject-id", default="r04")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats має бути не меншим за 1")
    if args.failure_rate is not None:
        if not 0 <= args.failure_rate <= 1:
            parser.error("--failure-rate має бути від 0 до 1")
        os.environ["SHOP_FAILURE_RATE"] = str(args.failure_rate)

    requests = order_requests(load_requests(args.requests))
    if args.ids:
        selected = {value.strip() for value in args.ids.split(",") if value.strip()}
        unknown = selected - {item["id"] for item in requests}
        if unknown:
            parser.error(f"Невідомі ID: {', '.join(sorted(unknown))}")
        required = set(selected)
        by_id = {item["id"]: item for item in requests}
        for request_id in tuple(required):
            parent = by_id[request_id].get("after")
            while parent:
                required.add(parent)
                parent = by_id[parent].get("after")
        requests = [item for item in requests if item["id"] in required]
    if args.limit is not None:
        requests = requests[:max(0, args.limit)]
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    modes = ("workflow", "agent") if args.mode == "both" else (args.mode,)
    for mode in modes:
        for repeat in range(1, args.repeats + 1):
            for request in requests:
                if not request.get("after"):
                    service.reset()
                result = execute(request, mode, repeat, args.reject_id)
                results.append(result)
                print(
                    f"{mode} {request['id']} #{repeat}: {result['status']}"
                    + (f" — {result['error']}" if result["error"] else ""),
                    flush=True,
                )
                save_results(args.output, results)

    save_results(args.output, results)
    print(f"Збережено {len(results)} результатів у {args.output.resolve()}")
    return 0 if all(not item["error"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
