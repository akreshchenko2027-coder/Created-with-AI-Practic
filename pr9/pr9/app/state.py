"""Стан звернення: що зберігається між кроками й між запусками
застосунку.

Окремий модуль, бо ним користуються і workflow (`app/workflow.py`), який
стан змінює, і сховище (`app/store.py`), яке його зберігає, і
веб-рівень, який його показує. Поведінки тут немає — лише дані.

Склад полів — ваше рішення: додайте те, чого бракує для продовження
після паузи й для пояснення, що сталося; приберіть те, що вам не
знадобилося. Якщо зміните склад, перевірте `store.py` і сторінку.
"""

from dataclasses import dataclass, field


@dataclass
class Step:
    """Запис журналу: один виконаний (або відхилений, або пропущений)
    крок.

    `n` — порядковий номер у зверненні (з 1). `name` — назва кроку,
    наприклад `parse`, `get_order`, `approval`, `reply`. `kind` — тип:
    `model`, `tool`, `code` або `human`. `status` — що сталося,
    наприклад `ok`, `rejected`, `error`, `waiting`. `detail` — пояснення
    для людини. `input` і `output` — що крок отримав і що повернув
    (для кроку-інструмента — аргументи і те, що повернуто). `elapsed` —
    секунди, `usage` — токени, якщо крок звертався до моделі. `at` —
    коли записано, ISO 8601.
    """

    n: int
    name: str
    kind: str
    status: str
    detail: str | None = None
    input: dict | list | str | None = None
    output: dict | list | str | None = None
    elapsed: float | None = None
    usage: dict | None = None
    at: str | None = None


@dataclass
class Run:
    """Стан одного звернення — те, що зберігається і з чим працює
    веб-рівень.

    `customer_id` — клієнт із сеансу, а не з тексту звернення. `status` —
    де звернення зараз, наприклад `running`, `awaiting_approval`,
    `needs_info`, `handed_off`, `done`, `rejected`, `stopped`, `failed`;
    набір і значення — ваші, сторінка показує будь-який. `route` — вид
    звернення після розбору. `data` — те, що кроки передають один
    одному. `pending` — дія, що чекає підтвердження: що буде зроблено,
    з якими аргументами й на якій підставі; сторінка показує поля
    `summary`, `tool`, `arguments`, `grounds`, якщо вони є, і решту як є.
    `decision` — рішення оператора. `reply` — відповідь клієнтові.
    `stopped` — причина зупинки, якщо звернення зупинено обмеженням чи
    збоєм. `usage` і `elapsed` — сумарно за всі кроки, наприклад
    `{"model": 3.1, "tools": 0.02}`.

    Склад полів — ваше рішення. Якщо зміните його, змініть і `store.py`,
    і (за потреби) сторінку.
    """

    run_id: str
    customer_id: str
    text: str
    status: str = "running"
    route: str | None = None
    steps: list[Step] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    pending: dict | None = None
    decision: dict | None = None
    reply: str | None = None
    stopped: str | None = None
    usage: dict = field(default_factory=dict)
    elapsed: dict = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
