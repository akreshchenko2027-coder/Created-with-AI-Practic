"""Зразки звернень для сторінки: текст клієнта і факти, які встановив
оператор.

Читається з `samples/requests.json` один раз при старті. Поле
`reply_example` сторінці не віддається — воно для підставного
провайдера, а не для підказки операторові.
"""

import json
from pathlib import Path

SAMPLES_FILE = Path(__file__).parent.parent / "samples" / "requests.json"

_samples: list[dict] = json.loads(SAMPLES_FILE.read_text(encoding="utf-8"))
_by_id: dict[str, dict] = {s["id"]: s for s in _samples}


def all_samples() -> list[dict]:
    return _samples


def get(request_id: str) -> dict | None:
    return _by_id.get(request_id)


def public(sample: dict) -> dict:
    """Те, що бачить сторінка: без прикладу відповіді."""
    return {k: v for k, v in sample.items() if k != "reply_example"}
