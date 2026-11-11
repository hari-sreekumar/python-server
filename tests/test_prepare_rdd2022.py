from pathlib import Path

from src.pothole.train.prepare_rdd2022 import prepare, voc_to_yolo

XML = """<annotation><size><width>600</width><height>400</height><depth>3</depth></size>
<object><name>D40</name><bndbox><xmin>100</xmin><ymin>100</ymin><xmax>300</xmax><ymax>200</ymax></bndbox></object>
<object><name>D00</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
<object><name>D40</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>11</xmax><ymax>11</ymax></bndbox></object>
</annotation>"""


def test_voc_to_yolo_filters_class_and_tiny_boxes(tmp_path: Path):
    x = tmp_path / "a.xml"
    x.write_text(XML)
    assert voc_to_yolo(x, {"D40": 0}) == ["0 0.333333 0.375000 0.333333 0.250000"]


def test_prepare_builds_dataset(tmp_path: Path):
    for c in ("India",):
        (tmp_path / c / "train" / "annotations" / "xmls").mkdir(parents=True)
        (tmp_path / c / "train" / "images").mkdir(parents=True)
        for i in range(10):
            (tmp_path / c / "train" / "annotations" / "xmls" / f"i{i}.xml").write_text(XML)
            (tmp_path / c / "train" / "images" / f"i{i}.jpg").write_bytes(b"x")
    out = tmp_path / "out"
    st = prepare(tmp_path, out, ["India"], ["D40"], val_frac=0.2)
    assert st["train"] + st["val"] == 10 and st["val"] == 2 and st["boxes"] == 10
    assert "0: D40" in (out / "data.yaml").read_text()
    assert len(list((out / "labels" / "train").glob("*.txt"))) == 8
