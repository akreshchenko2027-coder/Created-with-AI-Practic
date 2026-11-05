import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.json")
    parser.add_argument("--output", type=Path, default=ROOT / "eval" / "results.json")
    parser.add_argument("--max-rounds", type=int)
    parser.add_argument("--failure-rate", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_rounds is not None:
        if args.max_rounds < 1:
            print("--max-rounds має бути не менше 1.", file=sys.stderr)
            return 2
        os.environ["TOOL_MAX_ROUNDS"] = str(args.max_rounds)
    if args.failure_rate is not None:
        if not 0 <= args.failure_rate <= 1:
            print("--failure-rate має бути від 0 до 1.", file=sys.stderr)
            return 2
        os.environ["SHOP_FAILURE_RATE"] = str(args.failure_rate)

    sys.path.insert(0, str(ROOT))
    from app import assistant, llm
    from shop import service

    try:
        llm.get_client()
    except llm.LLMError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    questions_path = args.questions if args.questions.is_absolute() else ROOT / args.questions
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    results = []

    for item in questions:
        service.reset()
        started = time.perf_counter()
        try:
            result = assistant.answer(item["question"], item["customer"])
            record = {
                **item,
                "answer": result.text,
                "calls": [call.__dict__ for call in result.calls],
                "rounds": result.rounds,
                "stopped": result.stopped,
                "model": result.model,
                "elapsed": result.elapsed,
                "usage": result.usage,
                "run_elapsed": time.perf_counter() - started,
            }
        except llm.LLMError as exc:
            record = {
                **item,
                "error": str(exc),
                "run_elapsed": time.perf_counter() - started,
            }
        results.append(record)
        state = record.get("stopped", "error")
        print(f"{item['id']}: {state}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Збережено {len(results)} результатів у {output_path}")
    return 0 if all("error" not in item for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
