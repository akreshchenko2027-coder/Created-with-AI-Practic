"""Перевірка середовища для ПР12.

Запуск із папки pr12:
    python check_env.py

Скрипт перевіряє версію Python, пакети, налаштування в `.env`, чи
знайдено застосунки ПР11 і ПР8 і чи немає в них заготовок, чи читаються
набори прикладів і чи всі перевірки в них відомі, які перевірки й
суддя ще заготовки, і — якщо суддю налаштовано — робить пробний виклик
судді з `response_format` за схемою вердикту: чи приймає провайдер
схему і чи повертає модель валідний JSON.

Це діагностика перед роботою, а не зразок для наслідування.
"""

import importlib
import inspect
import json
import os
import sys
from pathlib import Path

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "jsonschema")
HERE = Path(__file__).resolve().parent
CASE_FILES = ("eval/cases/drafts.json", "eval/cases/agent.json", "eval/cases/drafts.heldout.json")
EXAMPLE_SUFFIX = ".example.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def note(what: str, hint: str) -> None:
    """Зауваження, яке не є помилкою середовища."""
    print(f"[ .. ] {what} — {hint}")


def check_python() -> bool:
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    report(ok, f"Python {'.'.join(map(str, actual))}",
           "" if ok else f"потрібен Python {'.'.join(map(str, MIN_PYTHON))} або новіший")
    return ok


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            report(False, f"пакет {name}", "не встановлено: pip install -r requirements.txt")
            ok = False
        else:
            report(True, f"пакет {name}", "")
    return ok


def check_settings() -> bool:
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False
    ok = True
    if not (HERE / ".env").exists():
        report(False, "файл .env", "немає: скопіюйте .env.example у .env")
        ok = False
    load_dotenv(HERE / ".env")
    for name in ("DRAFTS_APP_DIR", "AGENT_APP_DIR"):
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", "не задано; шукатиму сусідню папку репозиторію")
    if os.getenv("LLM_PROVIDER") == "mock":
        note("LLM_PROVIDER=mock у pr12/.env", "обидва застосунки відповідатимуть підставним провайдером — "
                                              "це перевірка конвеєра, не оцінювання")
    return ok


def _app_dir(name: str, default: str) -> Path:
    value = os.getenv(name)
    if value:
        directory = Path(value).expanduser().resolve()
    else:
        directory = (HERE.parents[1] / default).resolve()
    nested = directory / default
    if not (directory / "app").exists() and (nested / "app").exists():
        directory = nested
    return directory


def check_app(label: str, env_name: str, default: str, files: tuple[str, ...]) -> bool:
    """Застосунок знайдено, і в перелічених файлах немає заготовок."""
    directory = _app_dir(env_name, default)
    if not directory.exists():
        report(False, f"{label}: папка {directory}", f"не знайдено; задайте {env_name} у .env")
        return False
    missing = [f for f in files if not (directory / f).exists()]
    if missing:
        report(False, f"{label}: {directory}", "немає файлів: " + ", ".join(missing))
        return False
    stubs = [f for f in files
             if "NotImplementedError" in (directory / f).read_text(encoding="utf-8", errors="replace")]
    if stubs:
        report(False, f"{label}: {directory}", "ще заготовки (функцію не оцінити): " + ", ".join(stubs))
        return False
    report(True, f"{label}: {directory}", "знайдено, заготовок немає")
    return True


def check_cases() -> bool:
    """Набори читаються, id унікальні, види відомі, перевірки існують."""
    sys.path.insert(0, str(HERE / "eval"))
    try:
        import checks
        import report as rpt
    except Exception as exc:  # noqa: BLE001
        report(False, "модулі eval/", f"не імпортуються: {exc}")
        return False
    ok = True
    for rel in CASE_FILES:
        path = HERE / rel
        if not path.exists():
            example = path.with_name(path.name.replace(".json", EXAMPLE_SUFFIX))
            note(f"набір {rel}", f"немає; скопіюйте {example.name} у {path.name}" if example.exists()
                 else "немає")
            continue
        try:
            cases = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            report(False, f"набір {rel}", f"не JSON: {exc}")
            ok = False
            continue
        ids = [c.get("id") for c in cases]
        problems = []
        if len(set(ids)) != len(ids):
            problems.append("id повторюються")
        bad_kinds = sorted({str(c.get("kind")) for c in cases if c.get("kind") not in rpt.KINDS})
        if bad_kinds:
            problems.append("невідомі види: " + ", ".join(bad_kinds))
        unknown = sorted({spec.get("check") for c in cases for spec in c.get("checks") or []
                          if spec.get("check") not in checks.REGISTRY})
        if unknown:
            problems.append("невідомі перевірки: " + ", ".join(map(str, unknown)))
        empty = [c.get("id") for c in cases if not c.get("checks")]
        if empty:
            problems.append("без перевірок: " + ", ".join(map(str, empty)))
        kinds = {k: sum(1 for c in cases if c.get("kind") == k) for k in rpt.KINDS}
        summary = ", ".join(f"{k} {n}" for k, n in kinds.items())
        if problems:
            report(False, f"набір {rel}: {len(cases)} прикладів", "; ".join(problems))
            ok = False
        else:
            report(True, f"набір {rel}: {len(cases)} прикладів", summary)
    return ok


def check_stubs() -> None:
    """Які перевірки й функції судді ще заготовки. На початку роботи так і
    має бути."""
    try:
        import checks
        import judge
    except Exception as exc:  # noqa: BLE001
        note("заготовки", f"не вдалося імпортувати eval/: {exc}")
        return
    pending = [name for name, fn in checks.REGISTRY.items()
               if "NotImplementedError" in inspect.getsource(fn)]
    if pending:
        note("перевірки-заготовки", ", ".join(pending) + " — приклади з ними не проходитимуть")
    else:
        report(True, "перевірки", "усі мають реалізацію")
    judge_pending = [name for name in ("build_messages", "evaluate")
                     if "NotImplementedError" in inspect.getsource(getattr(judge, name))]
    if judge_pending:
        note("суддя-заготовка", ", ".join(judge_pending) + " — перевірка judge не проходитиме")
    else:
        report(True, "суддя", "реалізовано")


def check_judge_call() -> bool:
    """Пробний виклик судді зі схемою вердикту."""
    try:
        import judge
    except Exception as exc:  # noqa: BLE001
        report(False, "пробний виклик судді", f"не вдалося імпортувати judge.py: {exc}")
        return False
    settings = judge.settings()
    if not settings["configured"]:
        note("суддя", "JUDGE_MODEL або ключ не задано — пробний виклик пропущено; "
                      "для сценаріїв із суддею потрібні")
        return True
    generator_model = os.getenv("LLM_MODEL", "")
    if generator_model and generator_model == settings["model"]:
        report(False, "модель-суддя", "має відрізнятися від моделі-генератора")
        return False
    try:
        import openai
    except ImportError:
        report(False, "пробний виклик судді", "немає пакета openai")
        return False

    client = openai.OpenAI(base_url=judge.JUDGE_BASE_URL, api_key=judge.JUDGE_API_KEY,
                           timeout=judge.JUDGE_TIMEOUT, max_retries=0)
    criteria = ["tone"]
    messages = [
        {"role": "system", "content": "Ти оцінюєш відповідь служби підтримки за критеріями. Для кожного "
                                      "критерію поверни passed true/false і коротку причину. Відповідай "
                                      "лише JSON за схемою: {\"criteria\": [{\"id\", \"passed\", "
                                      "\"reason\"}], \"overall\": bool}."
                                      + ("\n/no_think" if "qwen3" in judge.JUDGE_MODEL.lower() else "")},
        {"role": "user", "content": "Критерії: tone — текст ввічливий і діловий.\n\nВідповідь: «Добрий "
                                    "день, Ірино! Заявку R-5001 оформлено. З повагою, служба підтримки.»"},
    ]
    schema_format = {"type": "json_schema",
                     "json_schema": {"name": "verdict", "schema": judge.verdict_schema()}}

    def call(response_format: dict):
        request = {
            "model": judge.JUDGE_MODEL,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 300,
            "response_format": response_format,
        }
        if "qwen3" in judge.JUDGE_MODEL.lower():
            request["reasoning_effort"] = "none"
        completion = client.chat.completions.create(**request)
        return completion.choices[0].message.content or ""

    accepted = "json_object" if "qwen3" in judge.JUDGE_MODEL.lower() else "json_schema"
    try:
        try:
            response_format = {"type": "json_object"} if accepted == "json_object" else schema_format
            raw = call(response_format)
        except openai.BadRequestError:
            accepted = "json_object"
            raw = call({"type": "json_object"})
    except openai.AuthenticationError:
        report(False, "пробний виклик судді", "401 — ключ невірний або протермінований")
        return False
    except openai.RateLimitError:
        report(False, "пробний виклик судді", "429 — ліміт провайдера; спробуйте пізніше")
        return False
    except openai.APITimeoutError:
        report(False, "пробний виклик судді", f"таймаут після {judge.JUDGE_TIMEOUT:.0f} с")
        return False
    except openai.APIStatusError as exc:
        report(False, "пробний виклик судді", f"статус {exc.status_code}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний виклик судді", f"збій ({type(exc).__name__}): {exc}")
        return False

    try:
        verdict = judge.parse_verdict(raw, criteria)
    except ValueError as exc:
        if accepted != "json_schema":
            report(False, f"пробний виклик судді ({judge.JUDGE_MODEL}, {accepted})",
                   f"відповідь не за схемою: {exc}; сира відповідь: {raw[:120]!r}")
            return False
        accepted = "json_object"
        try:
            raw = call({"type": "json_object"})
            verdict = judge.parse_verdict(raw, criteria)
        except Exception as retry_exc:  # noqa: BLE001 — діагностика повторного формату
            report(False, f"пробний виклик судді ({judge.JUDGE_MODEL}, json_object)",
                   f"повтор не пройшов: {type(retry_exc).__name__}: {retry_exc}")
            return False
    report(True, f"пробний виклик судді ({judge.JUDGE_MODEL})",
           f"response_format {accepted}; вердикт за схемою: tone={verdict.criteria[0].passed}")
    if accepted == "json_object":
        note("response_format", "провайдер не прийняв json_schema — у evaluate використовуйте "
                                "json_object і схему в тексті інструкції")
    return True


def main() -> int:
    print("Перевірка середовища для ПР12\n")
    results = [check_python(), check_packages(), check_settings()]
    results.append(check_app("застосунок ПР11 (чернетка)", "DRAFTS_APP_DIR", "pr11",
                             ("app/drafting.py", "app/gateway.py", "app/providers/mock.py")))
    results.append(check_app("помічник ПР8 (інструменти)", "AGENT_APP_DIR", "pr8",
                             ("app/assistant.py", "app/tools.py", "shop/service.py")))
    results.append(check_cases())
    check_stubs()
    results.append(check_judge_call())
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
