from __future__ import annotations

import hashlib
import json
from typing import Any


CAMPAIGN_ID = "tile-catalog-warm-4-v2"
CAMPAIGN_NAME = "Warm-light full tile catalog raw capture"
LAYOUT_ID = "tile-catalog-layout-raw-001"
LAYOUT_VERSION = "TILE-CATALOG-CAPTURE-v2"

MANZU = tuple([f"{number}m" for number in range(1, 10)] + ["red5m"])
PINZU = tuple([f"{number}p" for number in range(1, 10)] + ["red5p"])
SOUZU = tuple([f"{number}s" for number in range(1, 10)] + ["red5s"])
HONORS = ("east", "south", "west", "north", "white", "green", "red")

CAPTURE_VARIANTS = (
    {
        "id": "normal-front",
        "label": "暖色・通常・影なし・正面",
        "brightness": "bright",
        "shadow": "none",
        "cameraPose": "front",
        "instruction": "黄色い照明を通常の明るさで点け、37牌すべてが写るように撮る。",
    },
    {
        "id": "dim-front",
        "label": "暖色・暗め・影なし・正面",
        "brightness": "dark",
        "shadow": "none",
        "cameraPose": "front",
        "instruction": "同じ配置のまま照明を暗めにして撮る。",
    },
    {
        "id": "normal-shadow",
        "label": "暖色・通常・部分影・正面",
        "brightness": "bright",
        "shadow": "partial",
        "cameraPose": "front",
        "instruction": "同じ配置のまま、牌の一部へ軽い影を落として撮る。",
    },
    {
        "id": "normal-angled",
        "label": "暖色・通常・影なし・少し斜め",
        "brightness": "bright",
        "shadow": "none",
        "cameraPose": "slight-angle",
        "instruction": "同じ配置のまま、カメラを少し横へ移動または回転して撮る。",
    },
)


def generate_tile_catalog_campaign() -> dict[str, Any]:
    rows = (
        ("萬子", MANZU),
        ("筒子", PINZU),
        ("索子", SOUZU),
        ("字牌", HONORS),
    )
    melds = [
        {
            "ordinal": row_ordinal,
            "kind": "catalog-row",
            "label": label,
            "tiles": _front_slots(tiles),
        }
        for row_ordinal, (label, tiles) in enumerate(rows)
    ]
    catalog_rows = [
        {
            "label": label,
            "region": "melds",
            "tiles": list(tiles),
        }
        for label, tiles in rows
    ]
    layout = {
        "id": LAYOUT_ID,
        "ordinal": 0,
        "layoutVersion": LAYOUT_VERSION,
        "rows": catalog_rows,
        "hand": [],
        "dora": {"visible": [], "ura": []},
        "melds": melds,
    }

    tasks: list[dict[str, Any]] = []
    for task_order, variant in enumerate(CAPTURE_VARIANTS):
        tasks.append(
            {
                "id": f"{CAMPAIGN_ID}-{variant['id']}",
                "campaignId": CAMPAIGN_ID,
                "layoutId": LAYOUT_ID,
                "layoutOrdinal": 0,
                "layoutVersion": LAYOUT_VERSION,
                "hand": [],
                "dora": {"visible": [], "ura": []},
                "melds": melds,
                "environment": {
                    "lighting": "warm",
                    "brightness": variant["brightness"],
                    "shadow": variant["shadow"],
                    "cameraPose": variant["cameraPose"],
                    "variantId": variant["id"],
                    "label": variant["label"],
                    "instruction": variant["instruction"],
                },
                "environmentOrdinal": task_order,
                "repetition": task_order,
                "expected": {
                    "hand": 0,
                    "dora": 0,
                    "meld": sum(len(tiles) for _label, tiles in rows),
                },
                "taskOrder": task_order,
                "catalogRows": catalog_rows,
            }
        )

    document: dict[str, Any] = {
        "id": CAMPAIGN_ID,
        "name": CAMPAIGN_NAME,
        "layoutCount": 1,
        "layoutVersion": LAYOUT_VERSION,
        "environments": [dict(variant) for variant in CAPTURE_VARIANTS],
        "layouts": [layout],
        "tasks": tasks,
        "coverage": {
            tile: len(tasks)
            for tile in (*MANZU, *PINZU, *SOUZU, *HONORS)
        },
    }
    canonical = json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    document["definitionSha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return document


def _front_slots(tiles: tuple[str, ...]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": ordinal,
            "tile": tile,
            "face": "front",
            "rotation": 0,
        }
        for ordinal, tile in enumerate(tiles)
    ]
