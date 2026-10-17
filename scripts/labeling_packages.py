"""Split prepared frames into per-person labeling packages for X-AnyLabeling, whole videos per person.

Each package holds images with draft labels as X-AnyLabeling JSON (unchecked) and classes.txt.
Share packages only directly inside the team (Tailscale, AirDrop, USB); never through cloud storage.
"""

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ("propagate_static.py", "apply_drafts.py")

INSTRUCTIONS = """Разметка: {name}
Ролики: {videos}
Кадров: {frames}

ЗАПУСК
  Mac:     двойной клик по start_mac.command
           (если macOS не даёт открыть: правый клик -> Открыть -> Открыть)
  Windows: двойной клик по start_windows.bat
  Первый запуск сам установит X-AnyLabeling 4.1.3 (несколько минут). Нужен Python 3.11+:
  https://www.python.org/downloads/ (на Windows отметьте «Add python.exe to PATH»).

КАК РАЗМЕЧАТЬ
  1. Первый кадр каждого ролика: всё, включая door, safe, laptop, counter, currency_detector. Отметьте кадр (см. п. 3).
     Закройте программу и дважды кликните copy_static_mac.command (Windows: copy_static_windows.bat).
     Дверь, сейф, ноутбук, счётчик, детектор валют скопируются на остальные кадры ролика (подпись copied). Откройте программу снова.
  2. Остальные кадры: удалите ложные рамки, дорисуйте пропущенные, проверьте скопированные.
     Cmd+J — режим редактирования; выделять рамки удобнее в списке Shapes справа;
     удалить — Fn+Delete (Windows: Delete); R — нарисовать рамку; Cmd+Z — отменить.
  3. Кадр готов -> Cmd+Option+K (Windows: Ctrl+Alt+K) или Edit -> Mark as Checked. В заголовке [Checked].
     Галочки слева в списке файлов означают только «есть файл разметки» — не смотрите на них.
     Без отметки Checked кадр не попадёт в обучение. Пустой кадр тоже отмечайте.
     Cmd+Shift+D — следующий неотмеченный кадр. A/D — назад/вперёд.
  ОСТОРОЖНО: Cmd+Delete удаляет файл разметки, Cmd+Shift+Delete удаляет сам кадр.
  Скриншоты кадров никуда не отправляйте.

ПРАВИЛА
  Классы: {classes}
  cash — одна рамка на пачку или веер купюр, лежащих вместе; отдельная купюра — отдельная рамка.
  id_card — только когда документ виден; в закрытой руке не размечать.
  personal_item — куртка, сумка, кружка/стакан на рабочем месте кассира (не одежда на человеке).
  door, safe, laptop — сам объект целиком, независимо от того, открыт он или нет.
  currency_detector — прибор проверки подлинности купюр (детектор валют), отдельный от счётчика денег (counter).
  Объект закрыт больше чем наполовину — не размечать. Ложные черновые рамки удалять.

СДАЧА (можно частями, например каждые 30–50 кадров)
  Заархивируйте папку images целиком или только *.json и передайте Диару напрямую:
  AirDrop, Tailscale, флешка. Каждая новая сдача заменяет прошлую, ничего не потеряется.

ЧЕРНОВИКИ ОТ МОДЕЛИ-ПОМОЩНИКА (когда пришлют архив)
  Закройте программу, выполните из этой папки:  python3 apply_drafts.py <архив>.zip   (Windows: py ...)
  Отмеченные кадры и кадры с вашими собственными рамками не меняются; старые версии сохраняются в backup_*.
  После этого снова запустите copy_static_* — дверь, сейф, ноутбук и счётчик вернутся с вашего кадра.
"""


def assign(frames_per_video: dict[str, int], people: list[str]) -> dict[str, list[str]]:
    """Whole videos per person, largest first to the least loaded person: balanced frame counts."""
    load = {person: 0 for person in people}
    plan: dict[str, list[str]] = {person: [] for person in people}
    for video, count in sorted(frames_per_video.items(), key=lambda item: (-item[1], item[0])):
        person = min(people, key=lambda name: (load[name], people.index(name)))
        plan[person].append(video)
        load[person] += count
    return plan


def draft_json(image: Path, label: Path, names: list[str]) -> dict:
    height, width = cv2.imread(str(image)).shape[:2]
    shapes = []
    if label.is_file():
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            index, x, y, w, h = line.split()
            x1, y1 = (float(x) - float(w) / 2) * width, (float(y) - float(h) / 2) * height
            x2, y2 = (float(x) + float(w) / 2) * width, (float(y) + float(h) / 2) * height
            shapes.append(
                {
                    "label": names[int(index)],
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "group_id": None,
                    "description": "draft",
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )
    return {
        "version": "4.1.3",
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image.name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("people", nargs="+", help="short names, e.g. diyar aru temir dana")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "training")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "labeling")
    parser.add_argument("--force", action="store_true", help="rebuild packages that already exist")
    args = parser.parse_args()
    if len(set(args.people)) != len(args.people):
        raise SystemExit("Names must be unique")
    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists; rerun with --force only if nobody has started labeling")
    started = [p for p in args.out.glob("*/images/*.json") if json.loads(p.read_text(encoding="utf-8")).get("checked")]
    if started:
        raise SystemExit(f"{len(started)} frames are already checked in {args.out}; refusing to rebuild over work")

    names = (args.dataset / "classes.txt").read_text().split()
    rows = list(csv.DictReader((args.dataset / "manifest.csv").open()))
    frames = defaultdict(list)
    for row in rows:
        frames[row["video"]].append(row)
    plan = assign({video: len(items) for video, items in frames.items()}, args.people)

    shutil.rmtree(args.out, ignore_errors=True)
    for person, videos in plan.items():
        package = args.out / person
        images = package / "images"
        images.mkdir(parents=True)
        count = 0
        for video in videos:
            for row in frames[video]:
                source = args.dataset / "images" / row["split"] / row["frame"]
                shutil.copy2(source, images / row["frame"])
                label = args.dataset / "labels" / row["split"] / f"{source.stem}.txt"
                data = draft_json(source, label, names)
                (images / f"{source.stem}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                count += 1
        (package / "classes.txt").write_text("\n".join(names) + "\n")
        (package / "START.txt").write_text(
            INSTRUCTIONS.format(
                name=person, videos=", ".join(sorted(videos)), frames=count, classes=", ".join(names)
            ),
            encoding="utf-8",
        )
        for tool in TOOLS:  # standard-library helpers, so teammates need no repository checkout
            shutil.copy2(ROOT / "scripts" / tool, package / tool)
        for launcher in (ROOT / "scripts" / "labeling_kit").iterdir():  # double-click start, keeps exec bit
            shutil.copy2(launcher, package / launcher.name)
        archive = shutil.make_archive(str(args.out / person), "zip", root_dir=args.out, base_dir=person)
        print(f"{person:10} {count:5} frames, {len(videos):2} videos -> {archive}")
    (args.out / "assignment.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print("After labeling: scripts/merge_labels.py <returned zip or folder> ...")


if __name__ == "__main__":
    main()
