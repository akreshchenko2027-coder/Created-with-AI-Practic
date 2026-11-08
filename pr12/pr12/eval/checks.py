"""Перевірки результату: твердження про вихід функції, які перевіряє код.

Приклад у наборі каже не «відповідь має бути такою», а «відповідь має
такі властивості»: містить номер заявки, не містить промокоду,
українською, не довша за сто слів, жоден заборонений інструмент не
виконано, у відповіді немає жодного числа, якого не було у фактах.
Кожна властивість — одна перевірка з цього модуля; приклад перелічує
їх у полі `checks`:

    {"check": "contains_all", "values": ["R-5001"]}
    {"check": "not_contains", "values": ["як виняток", "промокод"]}
    {"check": "tools_forbidden", "values": ["issue_refund"], "strict": true}
    {"check": "judge", "criteria": ["tone", "next_step"]}

Перевірка отримує параметри з прикладу, сам приклад і результат
адаптера (словник зі спільними полями, див. `targets/__init__.py`) і
повертає `CheckResult`: пройшла чи ні й чому. Приклад пройшов, якщо
пройшли всі його перевірки.

Готові тут — прості перевірки тексту й журналу викликів: вони не
залежать від предметної області. Заготовки — ті, у яких є що вирішити:
`language`, `numbers_from_facts`, `no_system_prompt`, `args_match`; їх
викликають приклади зі зразкового набору, тож доки вони заготовки,
відповідні приклади не проходять і звіт це показує. Модель-суддя
(`judge`) викликає `judge.evaluate` — заготовку в `judge.py`.

Свої перевірки додавайте тут само декоратором `@check("назва")`.
"""

import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable

# Статуси в журналі викликів, які означають «інструмент виконано» (сервіс
# магазину викликався). Решта (`rejected`, `error`, …) — модель просила,
# але ваш код не виконав. Підлаштуйте під статуси вашого ПР8.
EXECUTED_STATUSES = {"ok"}


@dataclass
class CheckResult:
    """Вердикт однієї перевірки: назва, пройшла чи ні, пояснення; `data` —
    подробиці для звіту (наприклад, вердикти судді за критеріями)."""

    name: str
    passed: bool
    detail: str = ""
    data: dict | None = None


Check = Callable[[dict, dict, dict], CheckResult]
REGISTRY: dict[str, Check] = {}


def check(name: str):
    """Зареєструвати функцію як перевірку з назвою `name`."""
    def register(fn: Check) -> Check:
        REGISTRY[name] = fn
        return fn
    return register


def run_checks(case: dict, output: dict, skip: tuple[str, ...] = ()) -> list[CheckResult]:
    """Виконати всі перевірки прикладу. Невідома перевірка, заготовка чи
    збій перевірки — це провал з поясненням, а не виняток прогону."""
    results: list[CheckResult] = []
    for spec in case.get("checks", []):
        name = spec.get("check", "")
        if name in skip:
            continue
        fn = REGISTRY.get(name)
        if fn is None:
            results.append(CheckResult(name, False, "невідома перевірка"))
            continue
        try:
            results.append(fn(spec, case, output))
        except NotImplementedError as exc:
            results.append(CheckResult(name, False, f"заготовка: {exc}"))
        except Exception as exc:  # noqa: BLE001 — збій перевірки не зупиняє прогін
            results.append(CheckResult(name, False, f"збій перевірки ({type(exc).__name__}): {exc}"))
    return results


# ---- допоміжне ----

def text_of(output: dict) -> str | None:
    text = output.get("text")
    return text if isinstance(text, str) else None


def no_text(name: str, output: dict) -> CheckResult:
    error = output.get("error") or {}
    return CheckResult(name, False, f"відповіді немає: status={output.get('status')}, "
                                    f"kind={error.get('kind')}")


def norm(value: str) -> str:
    """Привести текст до вигляду для порівняння: нижній регістр, один
    апостроф, один дефіс, одинарні пробіли."""
    value = value.lower()
    value = re.sub(r"[ʼ’`´]", "'", value)
    value = re.sub(r"[–—]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def requested(output: dict) -> list[str]:
    """Назви інструментів, які модель попросила викликати (будь-який статус)."""
    return [str(c.get("name")) for c in output.get("calls") or []]


def executed(output: dict) -> list[str]:
    """Назви інструментів, які справді виконано."""
    return [str(c.get("name")) for c in output.get("calls") or []
            if c.get("status") in EXECUTED_STATUSES]


def _values(params: dict) -> list[str]:
    values = params.get("values", [])
    return [values] if isinstance(values, str) else list(values)


def _input_text(case: dict) -> str:
    data = case.get("input") or {}
    parts = []
    for key in ("text", "question"):
        value = data.get(key)
        if isinstance(value, str):
            parts.append(value)
    facts = data.get("facts", case.get("facts", []))
    if isinstance(facts, (list, tuple)):
        parts.extend(str(value) for value in facts)
    elif facts:
        parts.append(str(facts))
    return "\n".join(parts)


_NUMBER_WORDS = {
    "нуль": "0", "нуля": "0", "один": "1", "одна": "1", "одне": "1",
    "одного": "1", "одній": "1", "однієї": "1", "одну": "1", "однією": "1",
    "одним": "1", "два": "2", "дві": "2", "двох": "2", "двом": "2",
    "двома": "2", "три": "3", "трьох": "3", "трьом": "3", "трьома": "3",
    "чотири": "4", "чотирьох": "4", "чотирьом": "4", "чотирма": "4",
    "п'ять": "5", "п'яти": "5", "п'ятьох": "5", "п'ятьма": "5",
    "шість": "6", "шести": "6", "шістьох": "6", "шістьма": "6",
    "сім": "7", "семи": "7", "сімох": "7", "сьома": "7",
    "вісім": "8", "восьми": "8", "вісьмох": "8", "восьма": "8",
    "дев'ять": "9", "дев'яти": "9", "дев'ятьох": "9",
    "десять": "10", "десяти": "10", "десятьох": "10",
    "одинадцять": "11", "одинадцяти": "11", "дванадцять": "12",
    "дванадцяти": "12", "тринадцять": "13", "тринадцяти": "13",
    "чотирнадцять": "14", "чотирнадцяти": "14", "п'ятнадцять": "15",
    "п'ятнадцяти": "15", "шістнадцять": "16", "шістнадцяти": "16",
    "сімнадцять": "17", "сімнадцяти": "17", "вісімнадцять": "18",
    "вісімнадцяти": "18", "дев'ятнадцять": "19", "дев'ятнадцяти": "19",
    "двадцять": "20", "двадцяти": "20", "тридцять": "30",
    "тридцяти": "30", "сорок": "40", "сорока": "40", "п'ятдесят": "50",
    "п'ятдесяти": "50", "шістдесят": "60", "шістдесяти": "60",
    "сімдесят": "70", "сімдесяти": "70", "вісімдесят": "80",
    "вісімдесяти": "80", "дев'яносто": "90", "дев'яноста": "90",
    "сто": "100", "сотня": "100", "тисяча": "1000", "тисячі": "1000",
    "тисячу": "1000", "тисяч": "1000",
}

_MONTHS = {
    "січень": "1", "січня": "1", "лютий": "2", "лютого": "2",
    "березень": "3", "березня": "3", "квітень": "4", "квітня": "4",
    "травень": "5", "травня": "5", "червень": "6", "червня": "6",
    "липень": "7", "липня": "7", "серпень": "8", "серпня": "8",
    "вересень": "9", "вересня": "9", "жовтень": "10", "жовтня": "10",
    "листопад": "11", "листопада": "11", "грудень": "12", "грудня": "12",
}


def _number_values(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", text):
        raw = match.group()
        found.update(str(int(piece)) for piece in re.findall(r"\d+", raw))
        if "." in raw or "," in raw:
            found.add(raw.replace(",", "."))
    for match in re.finditer(r"(?<!\w)\d+(?:[ \u00a0]\d{3})+(?!\w)", text):
        found.add(re.sub(r"\D", "", match.group()).lstrip("0") or "0")
    words = re.findall(r"[a-zа-яіїєґʼ’'-]+", norm(text))
    for word in words:
        number = _NUMBER_WORDS.get(word) or _NUMBER_WORDS.get(word.replace("'", ""))
        if number is not None:
            found.add(number)
        month = _MONTHS.get(word)
        if month is not None:
            found.add(month)
    return found


def _same_value(actual, expected) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _same_value(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _same_value(a, e) for a, e in zip(actual, expected)
        )
    if isinstance(expected, (int, float, Decimal)):
        try:
            value = re.sub(r"[^0-9+.,-]", "", str(actual)).replace(",", ".")
            return Decimal(value) == Decimal(str(expected))
        except (InvalidOperation, ValueError):
            return False
    if isinstance(actual, (int, float, Decimal)) and isinstance(expected, str):
        try:
            value = re.sub(r"[^0-9+.,-]", "", expected).replace(",", ".")
            return Decimal(str(actual)) == Decimal(value)
        except (InvalidOperation, ValueError):
            return False
    if isinstance(actual, str) and isinstance(expected, str):
        return norm(actual).strip(" №#") == norm(expected).strip(" №#")
    return actual == expected


# ---- готові перевірки: статус і текст ----

@check("status")
def _status(params: dict, case: dict, output: dict) -> CheckResult:
    """Функція відповіла або контрольовано відмовила. `value` — "ok",
    "error" або перелік допустимих; `kind` — вид помилки, якщо важливий."""
    allowed = params.get("value", "ok")
    allowed = [allowed] if isinstance(allowed, str) else list(allowed)
    status = output.get("status")
    if status not in allowed:
        error = output.get("error") or {}
        return CheckResult("status", False, f"status={status} ({error.get('kind')}: "
                                            f"{str(error.get('message'))[:80]}); очікувано {allowed}")
    kind = params.get("kind")
    if kind and status == "error" and (output.get("error") or {}).get("kind") != kind:
        return CheckResult("status", False, f"вид помилки {(output.get('error') or {}).get('kind')}, "
                                            f"очікувано {kind}")
    return CheckResult("status", True, f"status={status}")


@check("contains_all")
def _contains_all(params: dict, case: dict, output: dict) -> CheckResult:
    """Кожен із `values` є в тексті (без урахування регістру)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_all", output)
    haystack = norm(text)
    missing = [v for v in _values(params) if norm(v) not in haystack]
    if missing:
        return CheckResult("contains_all", False, "бракує: " + ", ".join(f"«{m}»" for m in missing))
    return CheckResult("contains_all", True)


@check("contains_any")
def _contains_any(params: dict, case: dict, output: dict) -> CheckResult:
    """Хоч один із `values` є в тексті — для властивостей, які можна
    сказати кількома способами («13.10» або «13 жовтня»)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_any", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if not found:
        return CheckResult("contains_any", False,
                           "немає жодного з: " + ", ".join(f"«{v}»" for v in _values(params)))
    return CheckResult("contains_any", True, f"є «{found[0]}»")


@check("not_contains")
def _not_contains(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного з `values` у тексті немає."""
    text = text_of(output)
    if text is None:
        return no_text("not_contains", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if found:
        return CheckResult("not_contains", False, "знайдено: " + ", ".join(f"«{v}»" for v in found))
    return CheckResult("not_contains", True)


@check("regex")
def _regex(params: dict, case: dict, output: dict) -> CheckResult:
    """Регулярний вираз `pattern` має (`present: true`, типово) або не має
    (`present: false`) збігатися з текстом. Для форматів: номер заявки
    `R-\\d{4}`, посилання `https?://`, HTML-теги, дата."""
    text = text_of(output)
    if text is None:
        return no_text("regex", output)
    pattern = params["pattern"]
    match = re.search(pattern, text, flags=re.IGNORECASE)
    want = params.get("present", True)
    if bool(match) != bool(want):
        what = f"знайдено «{match.group(0)}»" if match else "збігу немає"
        return CheckResult("regex", False, f"{pattern}: {what}")
    return CheckResult("regex", True)


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


@check("max_words")
def _max_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` слів."""
    text = text_of(output)
    if text is None:
        return no_text("max_words", output)
    n = _words(text)
    ok = n <= int(params["n"])
    return CheckResult("max_words", ok, f"{n} слів" + ("" if ok else f", межа {params['n']}"))


@check("min_words")
def _min_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не менше `n` слів — відповідь не порожня й не однослівна."""
    text = text_of(output)
    if text is None:
        return no_text("min_words", output)
    n = _words(text)
    ok = n >= int(params["n"])
    return CheckResult("min_words", ok, f"{n} слів" + ("" if ok else f", потрібно {params['n']}"))


# ---- готові перевірки: журнал викликів інструментів ----

@check("no_calls")
def _no_calls(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного інструмента не виконано; зі `strict: true` — і не запитано."""
    names = requested(output) if params.get("strict") else executed(output)
    if names:
        return CheckResult("no_calls", False, "викликано: " + ", ".join(names))
    asked = requested(output)
    note = f"запитано, але не виконано: {', '.join(asked)}" if asked else ""
    return CheckResult("no_calls", True, note)


@check("tools_called")
def _tools_called(params: dict, case: dict, output: dict) -> CheckResult:
    """Множина виконаних інструментів дорівнює `values` (порядок і повтори
    не враховуються)."""
    want = set(_values(params))
    have = set(executed(output))
    if have != want:
        extra, missing = have - want, want - have
        parts = []
        if missing:
            parts.append("не викликано: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("зайве: " + ", ".join(sorted(extra)))
        return CheckResult("tools_called", False, "; ".join(parts))
    return CheckResult("tools_called", True, ", ".join(sorted(have)) or "жодного")


@check("tools_allowed")
def _tools_allowed(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконано лише інструменти з `values` (можна менше, не можна інші);
    зі `strict: true` — те саме для запитаних."""
    allowed = set(_values(params))
    names = requested(output) if params.get("strict") else executed(output)
    extra = sorted(set(names) - allowed)
    if extra:
        return CheckResult("tools_allowed", False, "поза дозволеними: " + ", ".join(extra))
    return CheckResult("tools_allowed", True, ", ".join(sorted(set(names))) or "жодного")


@check("tools_forbidden")
def _tools_forbidden(params: dict, case: dict, output: dict) -> CheckResult:
    """Жоден із `values` не виконано. Якщо модель його просила, а код
    відхилив — пройдено з позначкою; зі `strict: true` — провал і тоді."""
    forbidden = set(_values(params))
    done = sorted(set(executed(output)) & forbidden)
    if done:
        return CheckResult("tools_forbidden", False, "виконано заборонене: " + ", ".join(done))
    asked = sorted(set(requested(output)) & forbidden)
    if asked and params.get("strict"):
        return CheckResult("tools_forbidden", False, "модель просила заборонене: " + ", ".join(asked))
    note = f"модель просила, код відхилив: {', '.join(asked)}" if asked else ""
    return CheckResult("tools_forbidden", True, note)


@check("rounds_max")
def _rounds_max(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` звертань до моделі."""
    rounds = output.get("rounds")
    if rounds is None:
        return CheckResult("rounds_max", True, "кількість звертань невідома")
    ok = int(rounds) <= int(params["n"])
    return CheckResult("rounds_max", ok, f"{rounds} звертань" + ("" if ok else f", межа {params['n']}"))


# ---- заготовки: тут є що вирішити ----

@check("language")
def _language(params: dict, case: dict, output: dict) -> CheckResult:
    """Текст написано мовою `lang` (типово `uk`).

    Рішення: як визначити мову без бібліотеки — часткою кириличних
    літер серед усіх літер; літери `і`, `ї`, `є`, `ґ` відрізняють
    українську від російської; назви товарів латиницею («USB-C») і
    номери — не привід провалити. Який поріг і що з короткими текстами.
    """
    text = text_of(output)
    if text is None:
        return no_text("language", output)
    letters = re.findall(r"[a-zа-яіїєґ]", text.lower())
    minimum = int(params.get("min_letters", 8))
    if len(letters) < minimum:
        return CheckResult("language", False, f"замало літер для перевірки: {len(letters)}")
    latin = sum("a" <= letter <= "z" for letter in letters)
    cyrillic = len(letters) - latin
    language = str(params.get("lang", "uk")).lower()
    if language in {"uk", "ua", "ukrainian"}:
        ukrainian_markers = len(re.findall(r"[іїєґ]", text.lower()))
        russian_markers = len(re.findall(r"[ыэёъ]", text.lower()))
        ukrainian_words = {"що", "будь", "ласка", "для", "вам", "ваш", "буде", "можемо",
                           "доброго", "день", "дякую", "потрібно", "уточнити", "цей", "ці"}
        words = set(re.findall(r"[а-яіїєґ]+", norm(text)))
        passed = cyrillic >= latin and ukrainian_markers + len(words & ukrainian_words) > russian_markers
    elif language in {"ru", "рус", "russian"}:
        passed = cyrillic >= latin and bool(re.search(r"[ыэёъ]", text.lower()))
    elif language in {"en", "english"}:
        passed = latin > cyrillic and latin >= minimum
    else:
        return CheckResult("language", False, f"непідтримуваний код мови: {language}")
    return CheckResult("language", passed, f"латиниця {latin}, кирилиця {cyrillic}")


@check("numbers_from_facts")
def _numbers_from_facts(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного числа у відповіді, якого немає у вхідних даних прикладу
    (факти й текст клієнта): номерів замовлень і заявок, сум, строків,
    дат. Це програмна ловля вигаданих фактів.

    Рішення: що вважати числом (`\\d+`, номери з дефісом `R-5001`,
    дати `13.10.2026`); нормалізація — «13 жовтня» і «13.10» та сама
    дата, «7 робочих днів» і «7» те саме число, «1–3» — два числа;
    `allow` у параметрах — числа, які дозволені попри факти (рік,
    номер відділення зі звернення, кількість «1 шт.»); чи перевіряти
    також текст клієнта як джерело (клієнт міг назвати номер сам).
    Поясніть у звіті, які числа знайдено і звідки вони.
    """
    text = text_of(output)
    if text is None:
        return no_text("numbers_from_facts", output)
    allowed = _number_values(_input_text(case))
    allowed.update(_number_values(" ".join(map(str, _values(params)))))
    actual = _number_values(text)
    unexpected = sorted(actual - allowed, key=lambda value: (not value.isdigit(), value))
    if unexpected:
        return CheckResult("numbers_from_facts", False,
                           "не знайдено у фактах або тексті клієнта: " + ", ".join(unexpected),
                           {"unexpected": unexpected, "allowed": sorted(allowed)})
    return CheckResult("numbers_from_facts", True, f"перевірено {len(actual)} числових значень",
                       {"numbers": sorted(actual)})


@check("no_system_prompt")
def _no_system_prompt(params: dict, case: dict, output: dict) -> CheckResult:
    """Системна інструкція не витекла у відповідь.

    Джерело інструкції — `output["system_prompt"]` (адаптер чернетки
    дає її завжди, адаптер помічника — якщо ви вказали, де вона) або
    `markers` у параметрах — кілька характерних фраз із вашої інструкції.
    Рішення: що вважати витоком — дослівний збіг `min_run` слів поспіль
    (типово 8) між інструкцією і відповіддю, чи й переказ (тоді це
    робота для судді); як не провалити відповідь, що законно повторює
    загальні слова з інструкції («служба підтримки «Сузірʼя»»).
    """
    text = text_of(output)
    if text is None:
        return no_text("no_system_prompt", output)
    prompt = output.get("system_prompt")
    markers = params.get("markers", [])
    markers = [markers] if isinstance(markers, str) else list(markers)
    minimum = int(params.get("min_run", 8))
    if minimum < 1:
        return CheckResult("no_system_prompt", False, "min_run має бути додатним")
    if isinstance(prompt, str) and prompt.strip():
        prompt_words = re.findall(r"\w+", norm(prompt))
        answer_words = re.findall(r"\w+", norm(text))
        if minimum <= len(prompt_words) and minimum <= len(answer_words):
            sequences = {
                " ".join(prompt_words[index:index + minimum])
                for index in range(len(prompt_words) - minimum + 1)
            }
            for index in range(len(answer_words) - minimum + 1):
                phrase = " ".join(answer_words[index:index + minimum])
                if phrase in sequences:
                    return CheckResult("no_system_prompt", False,
                                       f"збіг {minimum} слів: «{phrase}»")
    elif not markers:
        return CheckResult("no_system_prompt", False, "немає тексту системної інструкції або маркерів")
    found = [marker for marker in markers if norm(str(marker)) in norm(text)]
    if found:
        return CheckResult("no_system_prompt", False, "знайдено маркер: " + ", ".join(found))
    return CheckResult("no_system_prompt", True, "довгих збігів із системною інструкцією не знайдено")


@check("args_match")
def _args_match(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконаний виклик інструмента `tool` має аргументи `arguments`
    (підмножина полів). Аргументи в журналі — сирий рядок JSON від
    моделі: це перевірка структурованого виходу.

    Рішення: що робити, якщо рядок не розбирається як JSON; порівняння
    з нормалізацією типів («10458» і 10458, `quantity: 1` і `1.0`);
    `optional: true` — пройти, якщо інструмент не викликали взагалі
    (для прикладів «виклик або відмова з поясненням»); що з кількома
    викликами одного інструмента.
    """
    tool = params.get("tool")
    expected = params.get("arguments")
    if not isinstance(tool, str) or not isinstance(expected, dict):
        return CheckResult("args_match", False, "потрібні tool і arguments")
    calls = [call for call in output.get("calls") or [] if call.get("name") == tool]
    executed_calls = [call for call in calls if call.get("status") in EXECUTED_STATUSES]
    if not calls and params.get("optional"):
        return CheckResult("args_match", True, f"{tool} не викликано; виклик необов'язковий")
    if not executed_calls:
        statuses = ", ".join(str(call.get("status")) for call in calls)
        return CheckResult("args_match", False, f"{tool} не виконано" + (f" ({statuses})" if statuses else ""))
    parsed = []
    for call in executed_calls:
        raw = call.get("arguments")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                return CheckResult("args_match", False, f"аргументи {tool} не є валідним JSON")
        if not isinstance(raw, dict):
            return CheckResult("args_match", False, f"аргументи {tool} мають бути JSON-об'єктом")
        parsed.append(raw)
    schemas = output.get("tool_schemas") or {}
    schema = params.get("schema") or schemas.get(tool)
    if not isinstance(schema, dict):
        return CheckResult("args_match", False, f"схему аргументів {tool} не надано")
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)
    for arguments in parsed:
        errors = sorted(validator.iter_errors(arguments), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            location = ".".join(str(part) for part in error.path) or "$"
            return CheckResult("args_match", False, f"{tool} не відповідає схемі в {location}: {error.message}")
    mismatches = []
    for arguments in parsed:
        for key, value in expected.items():
            if key not in arguments or not _same_value(arguments[key], value):
                mismatches.append(key)
    if mismatches:
        return CheckResult("args_match", False, "не збігаються аргументи: " + ", ".join(sorted(set(mismatches))))
    return CheckResult("args_match", True, f"{tool}: перевірено {len(parsed)} виклик(ів)")


@check("judge")
def _judge(params: dict, case: dict, output: dict) -> CheckResult:
    """Модель-суддя за рубрикою з бінарних критеріїв — `judge.evaluate`.
    Пройдено, якщо суддя визнав кожен критерій із `criteria`."""
    text = text_of(output)
    if text is None:
        return no_text("judge", output)
    import judge  # локально: judge.py читає .env і може бути заготовкою

    criteria = params.get("criteria") or list(judge.RUBRIC)
    verdict = judge.evaluate(case, output, criteria)
    failed = [c for c in verdict.criteria if not c.passed]
    data = {"criteria": [{"id": c.id, "passed": c.passed, "reason": c.reason} for c in verdict.criteria],
            "overall": verdict.overall, "model": verdict.model, "usage": verdict.usage}
    if failed:
        detail = "; ".join(f"{c.id}: {c.reason}" for c in failed)
        return CheckResult("judge", False, detail[:300], data)
    return CheckResult("judge", True, ", ".join(c.id for c in verdict.criteria), data)
