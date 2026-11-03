"""Контракт відповіді моделі: що саме вона має повернути і як це
перевіряється.

Окремий модуль, як у ПР4, і з тієї самої причини: схема — домовленість
між моделлю та рештою застосунку, і вона не залежить від провайдера.
`app/rag.py` працює лише з тим, що пройшло перевірку тут.

Що потрібно решті застосунку від відповіді (розділ 2 практичної роботи):

* текст відповіді для клієнта;
* на які фрагменти контексту відповідь спирається — номери з позначок,
  які ви поставили в `retrieval.build_context`;
* чи знайдено відповідь у наданих фрагментах, чи їх для відповіді не
  вистачило.

Як назвати поля, які з них обовʼязкові, чи потрібні ще якісь — ваше
рішення. Як і те, чим описати схему: JSON Schema вручну або
pydantic-модель. Реалізацію з ПР4 можна взяти за основу.

Сторінка каркаса бере текст відповіді з поля `answer` результату
`rag.answer`, а не з відповіді моделі напряму — тож назви полів у схемі
ні до чого не привʼязані, крім вашого коду.
"""


import json


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі.

    Ця сама схема передається моделі як опис очікуваного результату
    (через `response_format` або текстом в інструкції) і використовується
    для перевірки того, що повернулося.
    """
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "found": {"type": "boolean"},
            "citations": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
                "uniqueItems": True,
            },
        },
        "required": ["answer", "found", "citations"],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    Якщо текст не є JSON або не проходить схему — підняти помилку з
    поясненням, що саме не так.

    Схема перевіряє форму. Чи існують названі моделлю номери джерел серед
    тих, що їй показали, і чи не «знайшла» вона відповідь, не пославшись
    ні на що, — перевірка змісту, і її місце в `app/rag.py`.
    """
    try:
        result = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Модель повернула невалідний JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("Відповідь моделі має бути JSON-об'єктом")
    if set(result) != {"answer", "found", "citations"}:
        raise ValueError("Відповідь має містити лише поля answer, found і citations")
    if not isinstance(result["answer"], str) or not result["answer"].strip():
        raise ValueError("Поле answer у відповіді моделі порожнє")
    if not isinstance(result["found"], bool):
        raise ValueError("Поле found має бути логічним значенням")
    citations = result["citations"]
    if (
        not isinstance(citations, list)
        or any(type(ref) is not int or ref < 1 for ref in citations)
        or len(citations) != len(set(citations))
    ):
        raise ValueError("Поле citations має бути списком унікальних додатних цілих чисел")
    if result["found"] != bool(result["citations"]):
        raise ValueError("Поле found має відповідати наявності citations")
    return result
