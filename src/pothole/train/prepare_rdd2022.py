"""Convert RDD2022 (Pascal-VOC XML) to a YOLO dataset.

RDD2022 layout (per country):  <root>/<Country>/train/images/*.jpg
                               <root>/<Country>/train/annotations/xmls/*.xml
(The official test split has no labels, so we carve val out of train.)

Classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, D40 pothole.

    python -m src.pothole.train.prepare_rdd2022 --root RDD2022 --out data/pothole/yolo \
        --countries India Japan --classes D40
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

ALL_CLASSES = ["D00", "D10", "D20", "D40"]


def voc_to_yolo(xml_path: Path, class_ids: dict[str, int]) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = float(size.findtext("width")), float(size.findtext("height"))
    if w <= 0 or h <= 0:
        return []
    lines = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in class_ids:
            continue
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0.0, min(x1, w)), max(0.0, min(x2, w))
        y1, y2 = max(0.0, min(y1, h)), max(0.0, min(y2, h))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        lines.append(
            f"{class_ids[name]} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}"
        )
    return lines


def prepare(root: Path, out: Path, countries: list[str], classes: list[str],
            val_frac: float = 0.15, keep_empty: float = 0.1, seed: int = 0) -> dict:
    class_ids = {c: i for i, c in enumerate(classes)}
    rng = random.Random(seed)
    items = []  # (image, label_lines)
    for country in countries:
        xml_dir = root / country / "train" / "annotations" / "xmls"
        img_dir = root / country / "train" / "images"
        for xml in sorted(xml_dir.glob("*.xml")):
            img = next((img_dir / (xml.stem + e) for e in (".jpg", ".jpeg", ".png")
                        if (img_dir / (xml.stem + e)).exists()), None)
            if img is None:
                continue
            lines = voc_to_yolo(xml, class_ids)
            if not lines and rng.random() > keep_empty:  # keep a few negatives, drop most empties
                continue
            items.append((country, img, lines))
    rng.shuffle(items)
    n_val = int(len(items) * val_frac)
    stats = {"train": 0, "val": 0, "boxes": 0}
    for i, (country, img, lines) in enumerate(items):
        split = "val" if i < n_val else "train"
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        name = f"{country}_{img.name}"
        shutil.copy2(img, out / "images" / split / name)
        (out / "labels" / split / (Path(name).stem + ".txt")).write_text("\n".join(lines))
        stats[split] += 1
        stats["boxes"] += len(lines)
    names = "\n".join(f"  {i}: {c}" for c, i in class_ids.items())
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n"
    )
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/pothole/yolo"))
    ap.add_argument("--countries", nargs="+", default=["India"])
    ap.add_argument("--classes", nargs="+", default=["D40"], choices=ALL_CLASSES)
    ap.add_argument("--val-frac", type=float, default=0.15)
    a = ap.parse_args()
    print(prepare(a.root, a.out, a.countries, a.classes, a.val_frac))
