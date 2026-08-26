from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal


CAMPAIGN_ID = "initial-120"
CAMPAIGN_NAME = "Initial deployment-layout capture campaign"
VISIBLE_TILE_CODES = tuple(
    [f"{number}{suit}" for suit in "mps" for number in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
    + ["red5m", "red5p", "red5s"]
)
NORMAL_TILE_ORDER = tuple(
    [f"{number}{suit}" for suit in "mps" for number in range(1, 10)]
    + ["east", "south", "west", "north", "white", "green", "red"]
)
ENVIRONMENTS = (
    ("bright", "none"),
    ("bright", "partial"),
    ("dark", "none"),
    ("dark", "partial"),
)
LAYOUT_COUNT = 30

TileFace = Literal["front", "back"]
MeldKind = Literal["chi", "pon", "open-kan", "closed-kan"]


@dataclass(frozen=True)
class Component:
    tiles: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class LayoutDefinition:
    id: str
    ordinal: int
    hand: tuple[dict[str, Any], ...]
    dora_visible: tuple[dict[str, Any], ...]
    dora_ura: tuple[dict[str, Any], ...]
    melds: tuple[dict[str, Any], ...]


def generate_campaign() -> dict[str, Any]:
    layout_coverage: Counter[str] = Counter()
    layouts: list[LayoutDefinition] = []
    for layout_ordinal in range(LAYOUT_COUNT):
        layout = _generate_layout(layout_ordinal, layout_coverage)
        layouts.append(layout)
        for visible_class in _visible_classes(layout):
            layout_coverage[visible_class] += 1

    tasks: list[dict[str, Any]] = []
    task_order = 0
    for layout in layouts:
        for environment_ordinal, (brightness, shadow) in enumerate(ENVIRONMENTS):
            task_id = (
                f"{CAMPAIGN_ID}-layout-{layout.ordinal + 1:03d}-"
                f"{brightness}-{shadow}"
            )
            task = {
                "id": task_id,
                "campaignId": CAMPAIGN_ID,
                "layoutId": layout.id,
                "layoutOrdinal": layout.ordinal,
                "hand": list(layout.hand),
                "dora": {
                    "visible": list(layout.dora_visible),
                    "ura": list(layout.dora_ura),
                },
                "melds": list(layout.melds),
                "environment": {
                    "brightness": brightness,
                    "shadow": shadow,
                },
                "environmentOrdinal": environment_ordinal,
                "expected": {
                    "hand": len(layout.hand),
                    "dora": len(layout.dora_visible) + len(layout.dora_ura),
                    "meld": sum(len(meld["tiles"]) for meld in layout.melds),
                },
                "taskOrder": task_order,
            }
            tasks.append(task)
            task_order += 1

    document = {
        "id": CAMPAIGN_ID,
        "name": CAMPAIGN_NAME,
        "layoutCount": LAYOUT_COUNT,
        "environments": [
            {"brightness": brightness, "shadow": shadow}
            for brightness, shadow in ENVIRONMENTS
        ],
        "layouts": [_layout_to_json(layout) for layout in layouts],
        "tasks": tasks,
        "coverage": dict(sorted(layout_coverage.items())),
    }
    validate_campaign(document)
    canonical = json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    document["definitionSha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return document


def validate_campaign(document: dict[str, Any]) -> None:
    layouts = document["layouts"]
    tasks = document["tasks"]
    if len(layouts) != LAYOUT_COUNT:
        raise ValueError(f"Expected {LAYOUT_COUNT} layouts, received {len(layouts)}")
    expected_task_count = LAYOUT_COUNT * len(ENVIRONMENTS)
    if len(tasks) != expected_task_count:
        raise ValueError(f"Expected {expected_task_count} tasks, received {len(tasks)}")

    coverage: Counter[str] = Counter()
    back_layouts = 0
    for layout in layouts:
        inventory = Counter[str]()
        visible_in_layout: set[str] = set()
        has_back = False
        for slot in _iter_layout_slots(layout):
            tile_code = str(slot["tile"])
            inventory[tile_code] += 1
            if slot["face"] == "back":
                has_back = True
            else:
                visible_in_layout.add(tile_code)
        _validate_inventory(inventory, str(layout["id"]))
        for tile_code in visible_in_layout:
            coverage[tile_code] += 1
        if has_back:
            back_layouts += 1

    missing = sorted(set(VISIBLE_TILE_CODES) - set(coverage))
    if missing:
        raise ValueError(f"Visible tile coverage is missing: {missing}")
    underrepresented = {
        tile_code: coverage[tile_code]
        for tile_code in VISIBLE_TILE_CODES
        if coverage[tile_code] < 3
    }
    if underrepresented:
        raise ValueError(
            "Every visible tile must appear in at least three layouts: "
            f"{underrepresented}"
        )
    if back_layouts < 3:
        raise ValueError(f"Back tiles appear in only {back_layouts} layouts")

    task_ids = [str(task["id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task IDs are not unique")
    task_orders = [int(task["taskOrder"]) for task in tasks]
    if task_orders != list(range(len(tasks))):
        raise ValueError("Task order is not contiguous")


def task_slots(task: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for slot in task["hand"]:
        yield _database_slot(task["id"], "hand", None, None, slot)
    for row_ordinal, row_key in enumerate(("visible", "ura")):
        region = "dora-visible" if row_key == "visible" else "dora-ura"
        for slot in task["dora"][row_key]:
            yield _database_slot(task["id"], region, row_ordinal, None, slot)
    for meld in task["melds"]:
        for slot in meld["tiles"]:
            yield _database_slot(
                task["id"],
                "meld",
                None,
                int(meld["ordinal"]),
                slot,
            )


def _generate_layout(
    layout_ordinal: int,
    existing_coverage: Counter[str],
) -> LayoutDefinition:
    used: Counter[str] = Counter()
    meld_count = layout_ordinal % 5
    meld_kinds: tuple[MeldKind, ...] = tuple(
        ("chi", "pon", "open-kan", "closed-kan")[(layout_ordinal + index) % 4]
        for index in range(meld_count)
    )

    melds: list[dict[str, Any]] = []
    for meld_ordinal, kind in enumerate(meld_kinds):
        component = _select_external_component(
            kind,
            used,
            offset=layout_ordinal * 11 + meld_ordinal * 7,
        )
        _allocate(used, component.tiles)
        melds.append(
            {
                "ordinal": meld_ordinal,
                "kind": kind,
                "tiles": _meld_slots(
                    component.tiles,
                    kind,
                    layout_ordinal,
                    meld_ordinal,
                ),
            }
        )

    concealed_component_count = 4 - meld_count
    concealed_tiles: list[str] = []
    for component_ordinal in range(concealed_component_count):
        component = _select_concealed_component(
            used,
            offset=layout_ordinal * 13 + component_ordinal * 9,
        )
        _allocate(used, component.tiles)
        concealed_tiles.extend(component.tiles)

    pair_tile = _select_pair_tile(used, layout_ordinal)
    _allocate(used, (pair_tile, pair_tile))
    concealed_tiles.extend((pair_tile, pair_tile))
    concealed_tiles.sort(key=_tile_sort_key)

    visible_count = 1 + layout_ordinal % 3
    ura_count = layout_ordinal % 3 if layout_ordinal % 2 == 1 else 0
    dora_visible_tiles = _select_indicator_tiles(
        used,
        visible_count,
        existing_coverage,
        offset=layout_ordinal * 5,
    )
    _allocate(used, dora_visible_tiles)
    dora_ura_tiles = _select_indicator_tiles(
        used,
        ura_count,
        existing_coverage,
        offset=layout_ordinal * 5 + visible_count,
    )
    _allocate(used, dora_ura_tiles)

    layout = LayoutDefinition(
        id=f"layout-{layout_ordinal + 1:03d}",
        ordinal=layout_ordinal,
        hand=tuple(_front_slots(concealed_tiles)),
        dora_visible=tuple(_front_slots(dora_visible_tiles)),
        dora_ura=tuple(_front_slots(dora_ura_tiles)),
        melds=tuple(melds),
    )
    _validate_inventory(
        Counter(slot["tile"] for slot in _iter_layout_slots(_layout_to_json(layout))),
        layout.id,
    )
    return layout


def _select_external_component(
    kind: MeldKind,
    used: Counter[str],
    offset: int,
) -> Component:
    candidates = {
        "chi": _sequence_components(),
        "pon": _triplet_components(),
        "open-kan": _quad_components(),
        "closed-kan": _quad_components(),
    }[kind]
    return _first_available(candidates, used, offset)


def _select_concealed_component(
    used: Counter[str],
    offset: int,
) -> Component:
    candidates = _sequence_components() + _triplet_components()
    return _first_available(candidates, used, offset)


def _select_pair_tile(used: Counter[str], offset: int) -> str:
    candidates = list(VISIBLE_TILE_CODES)
    for index in range(len(candidates)):
        tile = candidates[(offset * 7 + index) % len(candidates)]
        if _can_allocate(used, (tile, tile)):
            return tile
    raise RuntimeError("No pair tile is available")


def _select_indicator_tiles(
    used: Counter[str],
    count: int,
    coverage: Counter[str],
    offset: int,
) -> tuple[str, ...]:
    selected: list[str] = []
    candidates = sorted(
        VISIBLE_TILE_CODES,
        key=lambda tile: (
            coverage[tile],
            (VISIBLE_TILE_CODES.index(tile) - offset) % len(VISIBLE_TILE_CODES),
        ),
    )
    for tile in candidates:
        if len(selected) >= count:
            break
        prospective = tuple(selected + [tile])
        if _can_allocate(used, prospective):
            selected.append(tile)
    if len(selected) != count:
        raise RuntimeError(f"Could not allocate {count} indicator tiles")
    return tuple(selected)


def _sequence_components() -> tuple[Component, ...]:
    components: list[Component] = []
    for suit in "mps":
        for start in range(1, 8):
            tiles = tuple(f"{number}{suit}" for number in range(start, start + 3))
            components.append(Component(tiles=tiles, kind="sequence"))
            if start <= 5 <= start + 2:
                red_tiles = tuple(
                    f"red5{suit}" if number == 5 else f"{number}{suit}"
                    for number in range(start, start + 3)
                )
                components.append(Component(tiles=red_tiles, kind="sequence"))
    return tuple(components)


def _triplet_components() -> tuple[Component, ...]:
    return tuple(
        Component(tiles=(tile, tile, tile), kind="triplet")
        for tile in VISIBLE_TILE_CODES
        if _capacity(tile) >= 3
    )


def _quad_components() -> tuple[Component, ...]:
    return tuple(
        Component(tiles=(tile, tile, tile, tile), kind="quad")
        for tile in VISIBLE_TILE_CODES
        if _capacity(tile) >= 4
    )


def _first_available(
    candidates: tuple[Component, ...],
    used: Counter[str],
    offset: int,
) -> Component:
    for index in range(len(candidates)):
        candidate = candidates[(offset + index) % len(candidates)]
        if _can_allocate(used, candidate.tiles):
            return candidate
    raise RuntimeError("No legal component can be allocated")


def _front_slots(tiles: Iterable[str]) -> list[dict[str, Any]]:
    return [
        {"ordinal": ordinal, "tile": tile, "face": "front", "rotation": 0}
        for ordinal, tile in enumerate(tiles)
    ]


def _meld_slots(
    tiles: tuple[str, ...],
    kind: MeldKind,
    layout_ordinal: int,
    meld_ordinal: int,
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    called_index = (layout_ordinal + meld_ordinal) % len(tiles)
    for ordinal, tile in enumerate(tiles):
        face: TileFace = "front"
        rotation = 0
        if kind == "closed-kan" and ordinal in {0, len(tiles) - 1}:
            face = "back"
        elif ordinal == called_index:
            rotation = 90
        slots.append(
            {
                "ordinal": ordinal,
                "tile": tile,
                "face": face,
                "rotation": rotation,
            }
        )
    return slots


def _capacity(tile: str) -> int:
    if tile in {"5m", "5p", "5s"}:
        return 3
    if tile in {"red5m", "red5p", "red5s"}:
        return 1
    return 4


def _can_allocate(used: Counter[str], tiles: Iterable[str]) -> bool:
    requested = Counter(tiles)
    return all(used[tile] + amount <= _capacity(tile) for tile, amount in requested.items())


def _allocate(used: Counter[str], tiles: Iterable[str]) -> None:
    tiles_tuple = tuple(tiles)
    if not _can_allocate(used, tiles_tuple):
        raise RuntimeError(f"Tile inventory exceeded: {tiles_tuple}, used={dict(used)}")
    used.update(tiles_tuple)


def _validate_inventory(inventory: Counter[str], layout_id: str) -> None:
    unknown = sorted(set(inventory) - set(VISIBLE_TILE_CODES))
    if unknown:
        raise ValueError(f"{layout_id} contains unknown tiles: {unknown}")
    exceeded = {
        tile: (count, _capacity(tile))
        for tile, count in inventory.items()
        if count > _capacity(tile)
    }
    if exceeded:
        raise ValueError(f"{layout_id} exceeds physical tile inventory: {exceeded}")


def _tile_sort_key(tile: str) -> tuple[int, int, int]:
    if tile.endswith("m"):
        return (0, 5 if tile == "red5m" else int(tile[0]), 0 if tile == "red5m" else 1)
    if tile.endswith("p"):
        return (1, 5 if tile == "red5p" else int(tile[0]), 0 if tile == "red5p" else 1)
    if tile.endswith("s"):
        return (2, 5 if tile == "red5s" else int(tile[0]), 0 if tile == "red5s" else 1)
    return (3, NORMAL_TILE_ORDER.index(tile), 0)


def _layout_to_json(layout: LayoutDefinition) -> dict[str, Any]:
    return {
        "id": layout.id,
        "ordinal": layout.ordinal,
        "hand": list(layout.hand),
        "dora": {
            "visible": list(layout.dora_visible),
            "ura": list(layout.dora_ura),
        },
        "melds": list(layout.melds),
    }


def _iter_layout_slots(layout: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from layout["hand"]
    yield from layout["dora"]["visible"]
    yield from layout["dora"]["ura"]
    for meld in layout["melds"]:
        yield from meld["tiles"]


def _visible_classes(layout: LayoutDefinition) -> set[str]:
    document = _layout_to_json(layout)
    classes = {
        str(slot["tile"])
        for slot in _iter_layout_slots(document)
        if slot["face"] == "front"
    }
    if any(slot["face"] == "back" for slot in _iter_layout_slots(document)):
        classes.add("back")
    return classes


def _database_slot(
    task_id: str,
    region: str,
    row_ordinal: int | None,
    group_ordinal: int | None,
    slot: dict[str, Any],
) -> dict[str, Any]:
    slot_key = ":".join(
        [
            task_id,
            region,
            "-" if row_ordinal is None else str(row_ordinal),
            "-" if group_ordinal is None else str(group_ordinal),
            str(slot["ordinal"]),
        ]
    )
    return {
        "slot_key": slot_key,
        "task_id": task_id,
        "region": region,
        "row_ordinal": row_ordinal,
        "group_ordinal": group_ordinal,
        "tile_ordinal": int(slot["ordinal"]),
        "tile_code": str(slot["tile"]),
        "face": str(slot["face"]),
        "rotation": int(slot["rotation"]),
    }
