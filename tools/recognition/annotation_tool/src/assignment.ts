import { boxInside } from './geometry';
import type {
  AnnotationBox,
  CaptureDetail,
  CaptureTask,
  GroupAssignment,
  RegionAssignment,
  RegionKey,
  TileSlot,
  ValidationResult,
} from './types';

export function expectedGroups(
  task: CaptureTask,
  region: RegionKey,
): Array<{ label: string; slots: TileSlot[] }> {
  switch (region) {
    case 'completed_hand': {
      const slots = task.hand.filter((slot) => slot.face === 'front');
      const label = task.campaignId.startsWith('tile-catalog') ? '萬子' : '手牌';
      return slots.length === 0 ? [] : [{ label, slots }];
    }
    case 'dora_indicators': {
      const visible = task.dora.visible.filter((slot) => slot.face === 'front');
      const ura = task.dora.ura.filter((slot) => slot.face === 'front');
      const catalog = task.campaignId.startsWith('tile-catalog');
      return [
        { label: catalog ? '筒子' : '表示ドラ', slots: visible },
        { label: catalog ? '索子' : '裏ドラ', slots: ura },
      ].filter((group) => group.slots.length > 0);
    }
    case 'melds':
      return task.melds
        .map((meld, index) => ({
          label: meld.label ?? `副露${index + 1} ${meldKindLabel(meld.kind)}`,
          slots: meld.tiles.filter((slot) => slot.face === 'front'),
        }))
        .filter((group) => group.slots.length > 0);
  }
}

export function validateAnnotations(
  detail: CaptureDetail,
  boxesByRegion: Record<RegionKey, AnnotationBox[]>,
): ValidationResult {
  const regions = {} as Record<RegionKey, RegionAssignment>;
  let allInside = true;

  for (const region of regionKeys()) {
    const groups = expectedGroups(detail.task, region);
    const boxes = boxesByRegion[region];
    const regionRect = detail.manifest.regionRects[region].pixel;
    const width = Math.max(1, Math.floor(regionRect.width + 0.5));
    const height = Math.max(1, Math.floor(regionRect.height + 0.5));
    const inside = boxes.every((box) => boxInside(box, width, height));
    allInside &&= inside;

    const expectedCounts = groups.map((group) => group.slots.length);
    const expectedCount = expectedCounts.reduce((sum, count) => sum + count, 0);
    const clustered = boxes.length === expectedCount
      ? detail.task.campaignId.startsWith('tile-catalog') && region === 'melds'
        ? partitionCatalogRows(boxes, expectedCounts)
        : partitionBoxesByExpectedCounts(boxes, expectedCounts)
      : clusterBoxesByY(boxes, groups.length);
    const groupAssignments: GroupAssignment[] = groups.map((group, index) => {
      const groupBoxes = [...(clustered[index] ?? [])].sort((a, b) => a.centerX - b.centerX);
      return {
        label: group.label,
        expected: group.slots,
        boxes: groupBoxes,
        valid: groupBoxes.length === group.slots.length,
      };
    });
    const countsValid = (
      boxes.length === expectedCount
      && groupAssignments.every((group) => group.valid)
      && clustered.length === groups.length
    );
    const regionValid = countsValid && inside;
    const outsideBoxIds = new Set(
      boxes
        .filter((box) => !boxInside(box, width, height))
        .map((box) => box.id),
    );
    const labels = new Map<string, { text: string; tentative: boolean }>();

    for (let groupIndex = 0; groupIndex < groupAssignments.length; groupIndex += 1) {
      const group = groupAssignments[groupIndex];
      if (group === undefined) continue;
      for (let boxIndex = 0; boxIndex < group.boxes.length; boxIndex += 1) {
        const box = group.boxes[boxIndex];
        if (box === undefined) continue;
        const slot = group.expected[boxIndex];
        labels.set(box.id, {
          text: slot === undefined ? '余分' : tileLabel(slot),
          tentative: !countsValid || outsideBoxIds.has(box.id),
        });
      }
    }
    for (const box of boxes) {
      if (!labels.has(box.id)) labels.set(box.id, { text: '余分', tentative: true });
    }

    regions[region] = {
      labels,
      groups: groupAssignments,
      expectedCount,
      actualCount: boxes.length,
      valid: regionValid,
    };
  }

  return {
    regions,
    allInside,
    complete: allInside && regionKeys().every((region) => regions[region].valid),
  };
}

export function partitionCatalogRows(
  boxes: AnnotationBox[],
  expectedCounts: number[],
): AnnotationBox[][] {
  const rowSlope = estimateCatalogRowSlope(boxes);
  const ordered = [...boxes].sort((a, b) => {
    const aRowCoordinate = a.centerY - rowSlope * a.centerX;
    const bRowCoordinate = b.centerY - rowSlope * b.centerX;
    return aRowCoordinate - bRowCoordinate || a.centerX - b.centerX;
  });
  return partitionOrderedBoxes(ordered, expectedCounts);
}

export function partitionBoxesByExpectedCounts(
  boxes: AnnotationBox[],
  expectedCounts: number[],
): AnnotationBox[][] {
  const ordered = [...boxes].sort((a, b) => (
    a.centerY - b.centerY || a.centerX - b.centerX
  ));
  return partitionOrderedBoxes(ordered, expectedCounts);
}

function partitionOrderedBoxes(
  ordered: AnnotationBox[],
  expectedCounts: number[],
): AnnotationBox[][] {
  const groups: AnnotationBox[][] = [];
  let cursor = 0;
  for (const expectedCount of expectedCounts) {
    groups.push(ordered.slice(cursor, cursor + expectedCount));
    cursor += expectedCount;
  }
  return groups;
}

function estimateCatalogRowSlope(boxes: AnnotationBox[]): number {
  if (boxes.length < 2) return 0;
  const widths = boxes.map((box) => box.width).sort((a, b) => a - b);
  const heights = boxes.map((box) => box.height).sort((a, b) => a - b);
  const medianWidth = widths[Math.floor(widths.length / 2)] ?? 1;
  const medianHeight = heights[Math.floor(heights.length / 2)] ?? 1;
  const slopes: number[] = [];

  for (let leftIndex = 0; leftIndex < boxes.length; leftIndex += 1) {
    const left = boxes[leftIndex];
    if (left === undefined) continue;
    for (let rightIndex = leftIndex + 1; rightIndex < boxes.length; rightIndex += 1) {
      const right = boxes[rightIndex];
      if (right === undefined) continue;
      const deltaX = right.centerX - left.centerX;
      const deltaY = right.centerY - left.centerY;
      const absoluteDeltaX = Math.abs(deltaX);
      if (
        absoluteDeltaX < medianWidth * 0.45
        || absoluteDeltaX > medianWidth * 2.4
        || Math.abs(deltaY) > medianHeight * 0.7
      ) {
        continue;
      }
      slopes.push(deltaY / deltaX);
    }
  }

  if (slopes.length === 0) return 0;
  slopes.sort((a, b) => a - b);
  return slopes[Math.floor(slopes.length / 2)] ?? 0;
}

export function clusterBoxesByY(
  boxes: AnnotationBox[],
  clusterCount: number,
): AnnotationBox[][] {
  if (clusterCount <= 0) return boxes.length === 0 ? [] : [boxes];
  if (clusterCount === 1) return [[...boxes]];
  if (boxes.length === 0) return Array.from({ length: clusterCount }, () => []);

  const orderedY = boxes.map((box) => box.centerY).sort((a, b) => a - b);
  const centers = Array.from({ length: clusterCount }, (_, index) => {
    const position = Math.round(index * (orderedY.length - 1) / Math.max(1, clusterCount - 1));
    return orderedY[position] ?? 0;
  });
  let assignments = boxes.map(() => -1);

  for (let iteration = 0; iteration < 32; iteration += 1) {
    const next = boxes.map((box) => nearestCenter(box.centerY, centers));
    if (next.every((value, index) => value === assignments[index])) break;
    assignments = next;
    for (let centerIndex = 0; centerIndex < clusterCount; centerIndex += 1) {
      const members = boxes.filter((_, index) => assignments[index] === centerIndex);
      if (members.length > 0) {
        centers[centerIndex] = members.reduce((sum, box) => sum + box.centerY, 0) / members.length;
      }
    }
  }

  const centerOrder = centers
    .map((center, index) => ({ center, index }))
    .sort((a, b) => a.center - b.center)
    .map(({ index }) => index);
  return centerOrder.map((centerIndex) => (
    boxes.filter((_, boxIndex) => assignments[boxIndex] === centerIndex)
  ));
}

export function tileLabel(slot: TileSlot): string {
  const names: Record<string, string> = {
    east: '東',
    south: '南',
    west: '西',
    north: '北',
    white: '白',
    green: '發',
    red: '中',
    red5m: '赤5m',
    red5p: '赤5p',
    red5s: '赤5s',
  };
  const base = names[slot.tile] ?? slot.tile;
  const normalizedRotation = ((slot.rotation % 360) + 360) % 360;
  return normalizedRotation === 0 ? base : `${base} ↻${normalizedRotation}°`;
}

export function regionKeys(): RegionKey[] {
  return ['completed_hand', 'dora_indicators', 'melds'];
}

function nearestCenter(value: number, centers: number[]): number {
  let winner = 0;
  let distance = Number.POSITIVE_INFINITY;
  for (let index = 0; index < centers.length; index += 1) {
    const center = centers[index];
    if (center === undefined) continue;
    const candidate = Math.abs(value - center);
    if (candidate < distance) {
      distance = candidate;
      winner = index;
    }
  }
  return winner;
}

function meldKindLabel(kind: CaptureTask['melds'][number]['kind']): string {
  switch (kind) {
    case 'chi': return 'チー';
    case 'pon': return 'ポン';
    case 'open-kan': return '明槓';
    case 'closed-kan': return '暗槓';
    case 'catalog-row': return 'カタログ行';
  }
}
