"""Підготовка зображення до надсилання моделі.

Файл, який надіслав користувач, — недовірені байти: це може бути не
зображення, зображення завбільшки 40 мегапікселів, фото, повернуте на
бік (орієнтація записана в EXIF, а не в пікселях), або знімок 300 × 200,
на якому цифр не розібрати. Модуль вирішує, що з цього можна віддати
моделі і в якому вигляді. Про модель він не знає: повертає підготовлені
байти й відомості про них, а як вкласти їх у запит — справа `app/llm.py`.

Функція `prepare` виконує перевірку, орієнтацію, зменшення й перекодування.
Ліміти й формат — рішення з розділу 2 практичної роботи:

* що вважати неприйнятним файлом і яку помилку повернути: не
  зображення, порожній файл, завеликий файл, завеликий у пікселях;
* чи враховувати орієнтацію з EXIF (фото з телефона);
* до якого розміру зменшувати і чи зменшувати взагалі — ціна зображення
  в токенах росте з роздільністю, а дрібний шрифт зі зменшенням
  зливається;
* у якому форматі надсилати: PNG без втрат чи JPEG, який менший;
* чи відсіювати завідомо непридатні знімки ще до моделі (надто малі,
  надто розмиті) — і що тоді сказати користувачеві.
"""

import io
import os
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError
from dotenv import load_dotenv

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))
IMAGE_MAX_PIXELS = 40_000_000


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання.

    `data` і `mime` — те, що піде моделі. `original` і `sent` — розміри
    до й після підготовки, наприклад `{"width": 1240, "height": 1754,
    "bytes": 179639}`: сторінка показує їх поруч із токенами, щоб було
    видно, скільки коштує роздільність.
    """

    data: bytes
    mime: str
    original: dict
    sent: dict


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі.

    Непридатний файл — `ImageError` з поясненням, що не так.
    """
    if not content:
        raise ImageError("Файл порожній.")
    if len(content) > UPLOAD_MAX_BYTES:
        raise ImageError(f"Файл завеликий: максимум {UPLOAD_MAX_BYTES // (1024 * 1024)} МБ.")
    if IMAGE_MAX_SIDE < 1:
        raise ValueError("IMAGE_MAX_SIDE має бути додатним числом.")
    if IMAGE_MAX_PIXELS < 1:
        raise ValueError("IMAGE_MAX_PIXELS має бути додатним числом.")

    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > IMAGE_MAX_PIXELS:
                raise ImageError(
                    f"Зображення має забагато пікселів (ліміт {IMAGE_MAX_PIXELS:,})."
                )
            image = ImageOps.exif_transpose(source)
            image.load()
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageError("Файл не є коректним зображенням або його неможливо прочитати.") from exc

    original = {
        "width": image.width,
        "height": image.height,
        "bytes": len(content),
    }
    image = image.convert("RGB")
    image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    data = output.getvalue()
    return PreparedImage(
        data=data,
        mime="image/jpeg",
        original=original,
        sent={"width": image.width, "height": image.height, "bytes": len(data)},
    )
