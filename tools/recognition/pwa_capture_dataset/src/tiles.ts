import type { MeldLayout, TileCode, TileSlot } from './types';

const HONOR_LABELS: Partial<Record<TileCode, string>> = {
  east: '東', south: '南', west: '西', north: '北',
  white: '白', green: '發', red: '中',
};

export function createTileStrip(slots: TileSlot[], className = ''): HTMLElement {
  const strip = document.createElement('div');
  strip.className = `tile-strip ${className}`.trim();
  if (slots.length === 0) {
    strip.classList.add('empty-tile-strip');
    strip.textContent = 'なし';
    return strip;
  }
  for (const slot of slots) strip.append(createTile(slot));
  return strip;
}

export function createMeldStack(melds: MeldLayout[]): HTMLElement {
  const stack = document.createElement('div');
  stack.className = 'meld-stack';
  if (melds.length === 0) {
    stack.classList.add('empty-meld-stack');
    stack.textContent = 'なし';
    return stack;
  }
  for (const meld of melds) {
    const row = document.createElement('div');
    row.className = 'meld-row';
    const label = document.createElement('span');
    label.className = 'meld-label';
    label.textContent = `${meld.ordinal + 1}. ${meldKindLabel(meld.kind)}`;
    row.append(label, createTileStrip(meld.tiles));
    stack.append(row);
  }
  return stack;
}

export function tileLabel(tile: TileCode): string {
  const honor = HONOR_LABELS[tile];
  if (honor !== undefined) return honor;
  if (tile.startsWith('red5')) return `赤5${tile.at(-1) ?? ''}`;
  return tile;
}

function createTile(slot: TileSlot): HTMLElement {
  const tile = document.createElement('div');
  tile.className = [
    'tile-card',
    slot.face === 'back' ? 'tile-back' : '',
    slot.tile.startsWith('red5') ? 'tile-red' : '',
    slot.rotation === 90 || slot.rotation === 270 ? 'tile-sideways' : '',
  ].filter(Boolean).join(' ');
  tile.dataset.ordinal = String(slot.ordinal);
  tile.title = `${slot.ordinal}: ${slot.tile}, ${slot.face}, ${slot.rotation}°`;
  const face = document.createElement('span');
  face.textContent = slot.face === 'back' ? '裏' : tileLabel(slot.tile);
  tile.append(face);
  if (slot.rotation === 180) tile.style.transform = 'rotate(180deg)';
  return tile;
}

function meldKindLabel(kind: MeldLayout['kind']): string {
  switch (kind) {
    case 'chi': return 'チー';
    case 'pon': return 'ポン';
    case 'open-kan': return '明槓';
    case 'closed-kan': return '暗槓';
  }
}
