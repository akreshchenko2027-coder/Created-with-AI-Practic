"""Контракт провайдера: єдине, що решта застосунку знає про моделі.

Усе, що стосується конкретного API — назва SDK, формат повідомлень,
коди помилок, вигляд фрагментів у streaming — лишається в реалізаціях
(`openai_compat.py`, `mock.py`). Шлюз (`app/gateway.py`) працює лише
з тим, що описано тут, і тому може підставити будь-якого провайдера:
справжнього, підставного чи запасного.

Відносно ПР10 до `Result` додано три поля: `cost` і `cost_estimated`
(вартість звертання, яку рахує шлюз через `app/costs.py`) та `cached`
(результат узято з кешу, провайдер не викликався). Якщо ви змінювали
цей файл у ПР10, перенесіть свої зміни сюди, а не навпаки.

Тут лише дані й сигнатури; поведінки немає.
"""

from dataclasses import dataclass, field
from typing import Iterator, Protocol


@dataclass
class Request:
    """Що просимо згенерувати.

    `messages` — історія у звичному форматі ролей. `model` — назва
    моделі або `None`, тоді провайдер бере свою за замовчуванням.
    `metadata` — те, що не йде моделі, але потрібне нам: `request_id`
    звернення, версія prompt, прапорець `fresh`, ідентифікатори
    чернетки й HTTP-запиту — для підставного провайдера, кешу й журналу.
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
    `cost` — вартість у доларах за `app/costs.py`, `cost_estimated` —
    чи це оцінка без `usage`. `cached` — результат з кешу.
    """

    text: str
    provider: str
    model: str
    usage: dict | None = None
    finish_reason: str | None = None
    elapsed: float | None = None
    cost: float | None = None
    cost_estimated: bool = False
    cached: bool = False
    cache_saved_cost: float | None = None
    cache_saved_tokens: int = 0


class ProviderError(Exception):
    """Збій провайдера, зведений до виду, зрозумілого шлюзу.

    `kind` — вид збою: `timeout` (не дочекалися), `rate_limit` (429),
    `unavailable` (5xx, обрив зʼєднання), `auth` (невірний або
    протермінований ключ), `bad_request` (провайдер не прийняв запит),
    `cancelled` (перервано на наше прохання), `budget` (денний ліміт
    витрат вичерпано — підіймає шлюз, не провайдер), `unknown`.
    `retryable` — чи має сенс повторити ту саму спробу. `retry_after` —
    скільки секунд просив зачекати провайдер, якщо просив. `message` —
    текст, який можна показати людині; `detail` — технічні подробиці
    для журналу, не для сторінки.

    Який виняток SDK у який `kind` перетворюється — рішення реалізації
    провайдера. Чи повторювати `kind` із `retryable=True` — рішення
    шлюзу: він знає про спроби, бюджет часу й запасний варіант, а
    провайдер — ні.
    """

    def __init__(self, kind: str, message: str, *, retryable: bool = False,
                 retry_after: float | None = None, detail: str | None = None,
                 cost: float | None = None, cost_estimated: bool = False,
                 usage: dict | None = None, attempts: int = 0,
                 provider: str | None = None, model: str | None = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.detail = detail
        self.cost = cost
        self.cost_estimated = cost_estimated
        self.usage = usage
        self.attempts = attempts
        self.provider = provider
        self.model = model


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
