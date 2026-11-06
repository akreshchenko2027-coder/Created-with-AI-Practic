import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))
if MAX_ROUNDS < 1:
    raise ValueError("TOOL_MAX_ROUNDS must be at least 1")


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def _arguments_text(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def answer(question: str, customer_id: str) -> Answer:
    messages = llm.build_messages(question)
    available_tools = tools.specs()
    context = tools.Context(customer_id=customer_id)
    calls: list[ToolTrace] = []
    elapsed = {"model": 0.0, "tools": 0.0}
    usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    has_usage = False
    model_name = None
    mutation_completed = False

    for round_number in range(1, MAX_ROUNDS + 1):
        response = llm.chat(messages, available_tools, tool_choice="auto")
        elapsed["model"] += response.get("elapsed") or 0.0
        model_name = response.get("model") or model_name
        usage = response.get("usage")
        if usage:
            has_usage = True
            usage_totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
            usage_totals["completion_tokens"] += usage.get("completion_tokens") or 0
            usage_totals["total_tokens"] += usage.get("total_tokens") or 0

        message = response.get("message")
        if not isinstance(message, dict):
            raise llm.LLMError("Сервіс моделі повернув некоректну відповідь.")
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            text = message.get("content")
            if not isinstance(text, str) or not text.strip():
                raise llm.LLMError("Модель не повернула текстової відповіді.")
            return Answer(
                text=text.strip(),
                calls=calls,
                rounds=round_number,
                model=model_name,
                elapsed=elapsed,
                usage=usage_totals if has_usage else None,
            )

        write_calls = [
            item
            for item in tool_calls
            if (item.get("function") or {}).get("name") == "create_return"
        ]
        reject_writes = len(write_calls) > 1

        for item in tool_calls:
            call_id = item.get("id")
            function = item.get("function") or {}
            name = function.get("name") or ""
            raw_arguments = _arguments_text(function.get("arguments", ""))
            if not call_id:
                raise llm.LLMError("Модель повернула виклик інструмента без ідентифікатора.")

            started = time.perf_counter()
            if reject_writes and name == "create_return":
                result = tools.ToolResult(
                    status="rejected",
                    content={
                        "error": "За одне звертання можна оформити лише одну заявку на повернення.",
                        "code": "multiple_mutations",
                    },
                    reason="За одне звертання можна оформити лише одну заявку на повернення.",
                )
            elif mutation_completed and name == "create_return":
                result = tools.ToolResult(
                    status="rejected",
                    content={
                        "error": "Заявку на повернення вже створено для цього запиту.",
                        "code": "duplicate_mutation",
                    },
                    reason="Заявку на повернення вже створено для цього запиту.",
                )
            else:
                result = tools.call(name, raw_arguments, context)
                if name == "create_return" and result.status == "ok":
                    mutation_completed = True
            tool_elapsed = time.perf_counter() - started
            elapsed["tools"] += tool_elapsed
            calls.append(
                ToolTrace(
                    round=round_number,
                    name=name or "(невідомий інструмент)",
                    arguments=raw_arguments,
                    status=result.status,
                    reason=result.reason,
                    result=result.content,
                    elapsed=tool_elapsed,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result.content, ensure_ascii=False, default=str),
                }
            )

        if round_number == MAX_ROUNDS:
            last_results = [call.result for call in calls if call.round == round_number]
            unavailable = any(
                isinstance(result, dict) and result.get("code") == "unavailable"
                for result in last_results
            )
            text = (
                "Сервіс магазину тимчасово недоступний. Спробуйте повторити запит пізніше."
                if unavailable
                else "Не вдалося завершити запит за відведену кількість звертань. Спробуйте уточнити його."
            )
            return Answer(
                text=text,
                calls=calls,
                rounds=round_number,
                stopped="limit",
                model=model_name,
                elapsed=elapsed,
                usage=usage_totals if has_usage else None,
            )

    raise RuntimeError("The assistant loop exited unexpectedly")
