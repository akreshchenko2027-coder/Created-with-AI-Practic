"""Погіршити зображення документа — для власних зразків до перевірки.

Готові погіршені варіанти лежать у `samples/degraded/`. Цей скрипт — щоб
зробити свої: інший ступінь того самого, комбінацію кількох вад, або
погіршити власне фото. Це інструмент, а не частина застосунку.

Приклади (з папки pr7):

    python degrade.py samples/clean/rahunok-02.png --blur 1.5
    python degrade.py samples/clean/rahunok-04.png --scale 0.25 --jpeg 20
    python degrade.py samples/clean/rahunok-06.png --rotate 90
    python degrade.py samples/clean/rahunok-02.png --contrast 0.3 --noise 25 -o samples/own/r02-bad.jpg

Без `-o` результат потрапляє в `samples/own/` з іменем, що перелічує
застосовані вади. Папка `samples/own/` до репозиторію не потрапляє.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

OWN = Path(__file__).parent / "samples" / "own"


def main() -> int:
    p = argparse.ArgumentParser(description="Погіршити зображення документа.")
    p.add_argument("source", type=Path, help="вихідне зображення")
    p.add_argument("-o", "--output", type=Path, help="куди зберегти (за замовчуванням samples/own/)")
    p.add_argument("--blur", type=float, help="розмиття, радіус у пікселях (1–4)")
    p.add_argument("--scale", type=float, help="масштаб, частка від оригіналу (0.2–0.8)")
    p.add_argument("--contrast", type=float, help="контраст, 1 — без змін (0.2–0.8 — вицвілий друк)")
    p.add_argument("--rotate", type=float, help="поворот, градуси проти годинникової стрілки")
    p.add_argument("--crop-bottom", type=float, help="відрізати знизу частку висоти (0.1–0.6)")
    p.add_argument("--noise", type=float, help="шум, сила (10–40)")
    p.add_argument("--jpeg", type=int, help="зберегти як JPEG із цією якістю (5–60)")
    args = p.parse_args()

    img = Image.open(args.source).convert("RGB")
    applied = []
    if args.crop_bottom:
        img = img.crop((0, 0, img.width, int(img.height * (1 - args.crop_bottom))))
        applied.append(f"crop{args.crop_bottom:g}")
    if args.rotate:
        img = img.rotate(args.rotate, resample=Image.BICUBIC, expand=True, fillcolor="white")
        applied.append(f"rot{args.rotate:g}")
    if args.contrast:
        img = ImageEnhance.Contrast(img).enhance(args.contrast)
        applied.append(f"contrast{args.contrast:g}")
    if args.blur:
        img = img.filter(ImageFilter.GaussianBlur(args.blur))
        applied.append(f"blur{args.blur:g}")
    if args.noise:
        noise = Image.effect_noise(img.size, args.noise).convert("RGB")
        img = Image.blend(img, noise, 0.15)
        applied.append(f"noise{args.noise:g}")
    if args.scale:
        img = img.resize((max(1, int(img.width * args.scale)), max(1, int(img.height * args.scale))),
                         Image.BILINEAR)
        applied.append(f"scale{args.scale:g}")
    if not applied and not args.jpeg:
        print("Не задано жодної вади — див. python degrade.py --help")
        return 1

    if args.output:
        out = args.output
    else:
        suffix = ".jpg" if args.jpeg else ".png"
        tags = "-".join(applied + ([f"jpeg{args.jpeg}"] if args.jpeg else []))
        out = OWN / f"{args.source.stem}-{tags}{suffix}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.jpeg or out.suffix.lower() in (".jpg", ".jpeg"):
        img.save(out, "JPEG", quality=args.jpeg or 85)
    else:
        img.save(out)
    print(f"{out}  {img.width}x{img.height}  {out.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
