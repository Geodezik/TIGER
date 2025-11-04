import gzip
import json
import os
import pandas as pd


def parse_gz_json(path: str):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            yield eval(line)

def build_item_dataframe(meta_path: str):
    rows = []
    for meta in parse_gz_json(meta_path):
        asin = meta.get("asin")
        title = meta.get("title", "")
        brand = meta.get("brand", "Unknown")
        cats = meta.get("categories", [])
        if isinstance(cats, list) and len(cats) > 0:
            first = cats[0]
            if isinstance(first, list):
                cat_str = " > ".join(map(str, first))
            else:
                cat_str = str(first)
        else:
            cat_str = ""
        price = meta.get("price", "")
        rows.append({
            "asin": asin,
            "title": title,
            "brand": brand,
            "categories": cat_str,
            "price": price
        })
    return pd.DataFrame(rows)

def load_datamaps(datamaps_path: str):
    with open(datamaps_path, "r") as f:
        dm = json.load(f)
    return {asin: int(i) - 1 for asin, i in dm["item2id"].items()}

def prepare_items(meta_df: pd.DataFrame, datamaps_path: str, text_template: str):
    meta_df = meta_df.copy()
    if datamaps_path and os.path.exists(datamaps_path):
        asin2id = load_datamaps(datamaps_path)
        meta_df = meta_df[meta_df["asin"].isin(asin2id.keys())]
        meta_df["item_id"] = meta_df["asin"].map(asin2id)
        meta_df = meta_df.sort_values("item_id")
        ids = meta_df["item_id"].tolist()
    else:
        ids = meta_df["asin"].tolist()

    texts = [
        text_template.format(
            title=row["title"],
            brand=row["brand"],
            categories=row["categories"],
            price=row["price"],
        )
        for _, row in meta_df.iterrows()
    ]
    return ids, texts, meta_df
