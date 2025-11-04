import sys
import gzip
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from embeddings_extraction.data import build_item_dataframe, prepare_items


def _write_meta_gz(path: Path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(str(r) + "\n")

def test_build_item_dataframe_and_prepare_items_with_datamaps(tmp_path):
    meta_path = tmp_path / "meta.json.gz"
    rows = [
        {"asin": "A1", "title": "T1", "brand": "B1", "categories": [["Cat", "Sub"]], "price": "10"},
        {"asin": "A2", "title": "T2", "brand": "B2", "categories": ["Root"], "price": "20"},
    ]
    _write_meta_gz(meta_path, rows)

    df = build_item_dataframe(str(meta_path))
    assert isinstance(df, pd.DataFrame)
    assert set(["asin", "title", "brand", "categories", "price"]).issubset(df.columns)
    assert df.loc[0, "categories"] in ("Cat > Sub", "Root")

    datamaps_path = tmp_path / "datamaps.json"
    with open(datamaps_path, "w") as f:
        json.dump({"item2id": {"A1": 1, "A2": 2}}, f)

    template = "Title: {title}; Brand: {brand}; Categories: {categories}; Price: {price};"
    ids, texts, df_prepared = prepare_items(df, str(datamaps_path), template)

    assert ids == [0, 1]
    assert len(texts) == 2
    assert all(t.startswith("Title: ") and t.endswith(";") for t in texts)
    assert len(df_prepared) == 2


def test_prepare_items_without_datamaps(tmp_path):
    meta_path = tmp_path / "meta.json.gz"
    _write_meta_gz(meta_path, [{"asin": "X", "title": "TT", "brand": "BB", "categories": [], "price": ""}])
    df = build_item_dataframe(str(meta_path))
    ids, texts, _ = prepare_items(df, datamaps_path=None, text_template="Title: {title}; Brand: {brand}; Categories: {categories}; Price: {price};")
    assert ids == ["X"]
    assert len(texts) == 1
    assert "TT" in texts[0]
