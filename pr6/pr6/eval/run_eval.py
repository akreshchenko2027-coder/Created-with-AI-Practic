"""Прогін набору питань і збереження контексту для ручного розбору."""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import index, keyword, rag, retrieval
from app.llm import LLMError


def load_questions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("питання")
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"У файлі {path} немає непорожнього масиву «питання»")
    for number, item in enumerate(questions, start=1):
        required = {"питання", "очікувані_документи", "очікується_відповідь"}
        if not required.issubset(item):
            raise ValueError(f"У питанні {number} бракує полів {sorted(required - item.keys())}")
    return questions


def write_results(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("eval/questions.json")
        if Path("eval/questions.json").is_file()
        else Path("eval/questions.example.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="продовжити з результату, збереженого у --output",
    )
    parser.add_argument(
        "--pause-between",
        type=float,
        default=0,
        help="пауза в секундах між запитами до моделі (для дотримання лімітів API)",
    )
    parser.add_argument(
        "--context-chunks",
        type=int,
        help="змінити лише кількість фрагментів у контексті для експерименту",
    )
    args = parser.parse_args()
    if args.context_chunks is not None:
        if args.context_chunks < 1:
            parser.error("--context-chunks має бути додатним числом")
        retrieval.CONTEXT_CHUNKS = args.context_chunks
    if args.pause_between < 0:
        parser.error("--pause-between не може бути відʼємним")

    questions = load_questions(args.questions)
    search_index = index.load()
    keyword_index = keyword.build(search_index.chunks)
    records: list[dict] = []
    completed: dict[int, dict] = {}
    if args.resume:
        if not args.output.is_file():
            parser.error(f"немає файлу для продовження: {args.output}")
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        previous_metrics = previous.get("metrics", {})
        if previous_metrics.get("context_chunks") != retrieval.CONTEXT_CHUNKS:
            parser.error("кількість фрагментів у збереженому прогоні не збігається")
        for record in previous.get("results", []):
            number = record.get("number")
            if (
                not isinstance(number, int)
                or number < 1
                or number > len(questions)
                or record.get("question") != questions[number - 1]["питання"]
            ):
                parser.error("набір питань у збереженому прогоні не збігається")
            completed[number] = record

    for number, item in enumerate(questions, start=1):
        if number in completed:
            records.append(completed[number])
            print(f"[{number:02}] вже виконано; пропускаю")
            continue
        try:
            result = rag.answer(
                item["питання"],
                search_index,
                keyword_index,
                filters=item.get("фільтри"),
            )
        except LLMError as exc:
            write_results(
                args.output,
                {
                    "metrics": {
                        "completed": len(records),
                        "questions": len(questions),
                        "context_chunks": retrieval.CONTEXT_CHUNKS,
                    },
                    "results": records,
                },
            )
            print(
                f"Зупинка на питанні {number}; проміжні результати збережено "
                f"у {args.output}. Помилка моделі: {exc}",
                file=sys.stderr,
            )
            return 1
        retrieved = [
            {
                "source": hit.chunk.source,
                "score": hit.score,
                "metadata": hit.chunk.metadata,
                "text": hit.chunk.text,
            }
            for hit in result.retrieved
        ]
        expected_documents = item["очікувані_документи"]
        found_documents = {hit["source"] for hit in retrieved}
        record = {
            "number": number,
            "kind": item.get("вид"),
            "question": item["питання"],
            "expected_documents": expected_documents,
            "retrieved_documents": sorted(found_documents),
            "expected_documents_in_context": set(expected_documents).issubset(found_documents),
            "expected_answer": item.get("очікувана_відповідь"),
            "expected_found": item["очікується_відповідь"],
            "found": result.found,
            "decision_correct": result.found == item["очікується_відповідь"],
            "answer": result.text,
            "sources": [
                {"ref": source.ref, "source": source.chunk.source, "score": source.score}
                for source in result.sources
            ],
            "retrieved": retrieved,
            "model": result.model,
            "elapsed": result.elapsed,
            "usage": result.usage,
            "review": {
                "answer_contains_expected_facts": None,
                "answer_grounded_in_context": None,
                "citations_support_claims": None,
                "failure_classification": None,
                "notes": None,
            },
        }
        records.append(record)
        write_results(
            args.output,
            {
                "metrics": {
                    "completed": len(records),
                    "questions": len(questions),
                    "context_chunks": retrieval.CONTEXT_CHUNKS,
                },
                "results": records,
            },
        )
        coverage = "так" if record["expected_documents_in_context"] else "ні"
        decision = "так" if record["decision_correct"] else "ні"
        print(f"[{number:02}] контекст: {coverage}; рішення: {decision}; {result.text}")
        if args.pause_between and number < len(questions):
            time.sleep(args.pause_between)

    answered_questions = sum(bool(item["очікується_відповідь"]) for item in questions)
    retrieval_records = [
        record for record in records if record["expected_documents"]
    ]
    metrics = {
        "questions": len(questions),
        "expected_document_coverage": (
            sum(record["expected_documents_in_context"] for record in retrieval_records)
            / len(retrieval_records)
            if retrieval_records else None
        ),
        "answer_decision_accuracy": sum(record["decision_correct"] for record in records) / len(records),
        "false_answers": sum(
            record["found"] and not record["expected_found"] for record in records
        ),
        "unnecessary_refusals": sum(
            not record["found"] and record["expected_found"] for record in records
        ),
        "answerable_questions": answered_questions,
        "context_chunks": retrieval.CONTEXT_CHUNKS,
    }
    output = {"metrics": metrics, "results": records}
    write_results(args.output, output)
    print(f"\nМетрики: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"Детальні результати збережено: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
