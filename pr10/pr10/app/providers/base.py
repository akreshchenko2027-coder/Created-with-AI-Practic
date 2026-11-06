"""Контракт провайдера: єдине, що решта застосунку знає про моделі.

Усе, що стосується конкретного API — назва SDK, формат повідомлень,
коди помилок, вигляд фрагментів у streaming — лишається в реалізаціях
(`openai_compat.py`, `mock.py`). Шлюз (`app/gateway.py`) працює лише
з тим, що описано тут, і тому може підставити будь-якого провайдера:
справжнього, підставного чи запасного.

Тут лише дані й сигнатури; поведінки немає. Склад полів — відправна
точка: додайте те, чого бракує вашим реалізаціям, але так, щоб шлюз і
далі не залежав від конкретного провайдера.
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class Request:
    """Що просимо згенерувати.

    `messages` — історія у звичному форматі ролей. `model` — назва
    моделі або `None`, тоді провайдер бере свою за замовчуванням.
    `metadata` — те, що не йде моделі, але потрібне нам: наприклад,
    `request_id` звернення, щоб підставний провайдер знав, яку чернетку
    віддати, а журнал — до чого належить виклик.
    """

    messages: list[dict]
    max_tokens: int
    temperature: float
    model: str | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """Один фрагмент відповіді в режимі streaming.

    `text` — приріст тексту (може бути порожнім). `usage` і
    `finish_reason` деякі провайдери віддають лише в останньому
    фрагменті — тоді вони заповнені тільки там.
    """

    text: str
    usage: dict | None = None
    finish_reason: str | None = None


@dataclass
class Result:
    """Підсумок одного вдалого звертання.

    `provider` — назва провайдера, `model` — модель, яка насправді
    відповіла (у запасному варіанті вона може відрізнятися від
    замовленої). `usage` — токени, якщо провайдер їх повернув.
    """

    text: str
    provider: str
    model: str
    usage: dict | None = None
    finish_reason: str | None = None
    elapsed: float | None = None


class ProviderError(Exception):
    """Збій провайдера, зведений до виду, зрозумілого шлюзу.

    `kind` — вид збою: `timeout` (не дочекалися), `rate_limit` (429),
    `unavailable` (5xx, обрив зʼєднання), `auth` (невірний або
    протермінований ключ), `bad_request` (провайдер не прийняв запит),
    `cancelled` (перервано на наше прохання), `unknown`. `retryable` —
    чи має сенс повторити ту саму спробу. `retry_after` — скільки
    секунд просив зачекати провайдер, якщо просив. `message` — текст,
    який можна показати людині; `detail` — технічні подробиці для
    журналу, не для сторінки.

    Який виняток SDK у який `kind` перетворюється — рішення реалізації
    провайдера. Чи повторювати `kind` із `retryable=True` — рішення
    шлюзу: він знає про спроби, бюджет часу й запасний варіант, а
    провайдер — ні.
    """

    def __init__(self, kind: str, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.detail = detail


class Provider(Protocol):
    """Те, що вміє будь-який провайдер.

    `complete` повертає відповідь цілком. `stream` віддає фрагменти в
    міру надходження; споживач може перестати читати достроково — тоді
    реалізація має закрити зʼєднання, а не дочитувати до кінця.
    Обидва методи або повертають результат, або підіймають
    `ProviderError` — ніяких винятків SDK назовні.
    """

    name: str

    def complete(self, request: Request) -> Result: ...

    def stream(self, request: Request) -> Iterator[Chunk]: ...
