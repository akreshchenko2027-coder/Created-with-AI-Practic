"""Калібрування судді: наскільки його вердикти збігаються з вашими.

1. Зробити шаблон розмітки з прогону, у якому працював суддя:

       python eval/agreement.py --template eval/runs/drafts-....json > eval/labels.json

   У шаблоні — текст кожної оціненої відповіді і критерії з `null`
   замість вердикту. Проставте `true`/`false` самі, **не дивлячись** на
   вердикти судді (вони в файлі прогону, не в шаблоні).

2. Порахувати згоду за критеріями:

       python eval/agreement.py eval/runs/drafts-....json eval/labels.json

   Для кожного критерію: скільки оцінено, частка згоди, і окремо —
   скільки разів суддя пропустив те, що ви провалили (небезпечніша
   помилка), і навпаки. Низька згода — привід переписати формулювання
   критерію або віддати його коду чи людині, а не підкрутити суддю.

Готовий інструмент; змінювати не потрібно.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report  # noqa: E402


def judged(run: dict):
    """(ключ, текст, вердикти судді за критеріями) для кожної спроби з суддею."""
    for r in run["results"]:
        for c in r["checks"]:
            if c["name"] == "judge" and c.get("data") and c["data"].get("criteria"):
                key = f"{r['id']}#{r['attempt']}"
                verdicts = {item["id"]: item["passed"] for item in c["data"]["criteria"]}
                yield key, r["output"].get("text"), verdicts


def template(run: dict) -> dict:
    rows = {}
    for key, text, verdicts in judged(run):
        rows[key] = {"text": text, "criteria": {cid: None for cid in verdicts}}
    return rows


def agreement(run: dict, labels: dict) -> None:
    stats = defaultdict(lambda: {"n": 0, "agree": 0, "judge_pass_human_fail": 0, "judge_fail_human_pass": 0})
    for key, _, verdicts in judged(run):
        human = (labels.get(key) or {}).get("criteria") or {}
        for cid, judge_passed in verdicts.items():
            label = human.get(cid)
            if not isinstance(label, bool):
                continue
            s = stats[cid]
            s["n"] += 1
            if label == judge_passed:
                s["agree"] += 1
            elif judge_passed:
                s["judge_pass_human_fail"] += 1
            else:
                s["judge_fail_human_pass"] += 1
    if not stats:
        print("розмічених вердиктів немає: заповніть true/false у файлі розмітки")
        return
    print(f"{'критерій':<22}{'оцінено':>8}{'згода':>8}{'суддя так / ви ні':>20}{'суддя ні / ви так':>20}")
    total_n = total_agree = 0
    for cid, s in sorted(stats.items()):
        share = f"{s['agree'] / s['n']:.0%}" if s["n"] else "—"
        print(f"{cid:<22}{s['n']:>8}{share:>8}{s['judge_pass_human_fail']:>20}{s['judge_fail_human_pass']:>20}")
        total_n += s["n"]
        total_agree += s["agree"]
    print(f"{'разом':<22}{total_n:>8}{(total_agree / total_n if total_n else 0):>8.0%}")
    print("\n«суддя так / ви ні» — суддя пропустив те, що ви провалили: ці критерії переформулювати "
          "або віддати коду чи людині")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="файл прогону з вердиктами судді")
    parser.add_argument("labels", nargs="?", help="файл вашої розмітки (з --template)")
    parser.add_argument("--template", action="store_true", help="надрукувати шаблон розмітки")
    args = parser.parse_args()

    run = report.load_run(args.run)
    if args.template:
        print(json.dumps(template(run), ensure_ascii=False, indent=1))
        return 0
    if not args.labels:
        parser.error("вкажіть файл розмітки або --template")
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)
    agreement(run, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
