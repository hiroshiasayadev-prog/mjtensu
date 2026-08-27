import type { CSSProperties } from 'react';

import type { TileIdentity } from '@/domain';

import { TILE_BACK_ASSET_URL, tileAssetUrl } from './tile-assets';

export type TileFaceSize = 'standard' | 'compact' | 'keyboard';

export interface TileFaceProps {
  readonly tile: TileIdentity;
  readonly compact?: boolean;
  readonly size?: TileFaceSize;
  readonly selected?: boolean;
}

export interface TileBackProps {
  readonly compact?: boolean;
}

const HONOR_LABELS = {
  '1z': '東',
  '2z': '南',
  '3z': '西',
  '4z': '北',
  '5z': '白',
  '6z': '發',
  '7z': '中',
} as const;

const SUIT_LABELS = {
  m: '萬',
  p: '筒',
  s: '索',
} as const;

export function formatTileIdentity(tile: TileIdentity): string {
  if (tile.kind.endsWith('z')) {
    return HONOR_LABELS[tile.kind as keyof typeof HONOR_LABELS];
  }

  const number = tile.kind.at(0) ?? '';
  const suit = tile.kind.at(1);
  const suitLabel =
    suit === 'm' || suit === 'p' || suit === 's' ? SUIT_LABELS[suit] : '';

  return `${tile.red ? '赤' : ''}${number}${suitLabel}`;
}

export function TileFace({
  tile,
  compact = false,
  size,
  selected = false,
}: TileFaceProps) {
  const resolvedSize: TileFaceSize = size ?? (compact ? 'compact' : 'standard');
  const width = resolvedSize === 'compact'
    ? 20
    : resolvedSize === 'keyboard'
      ? 'clamp(28px, 7.5vw, 34px)'
      : 'clamp(22px, 5.6vw, 32px)';
  const wrapperStyle: CSSProperties = {
    display: 'inline-flex',
    width,
    aspectRatio: '3 / 4',
    boxSizing: 'border-box',
    flex: '0 0 auto',
    borderRadius: resolvedSize === 'compact' ? 3 : 5,
    outline: selected ? '2px solid #1971c2' : 'none',
    outlineOffset: selected ? 1 : 0,
    boxShadow: selected ? '0 3px 8px rgba(25, 113, 194, 0.24)' : undefined,
    transform: selected ? 'translateY(-3px)' : undefined,
  };

  return (
    <span
      aria-hidden="true"
      data-selected={selected ? 'true' : 'false'}
      data-tile-face-size={resolvedSize}
      data-tile-asset={tileAssetUrl(tile)}
      style={wrapperStyle}
    >
      <img
        alt=""
        draggable={false}
        src={tileAssetUrl(tile)}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          userSelect: 'none',
        }}
      />
    </span>
  );
}

export function TileBack({ compact = false }: TileBackProps) {
  const width = compact ? 20 : 'clamp(22px, 5.6vw, 32px)';

  return (
    <span
      aria-hidden="true"
      data-tile-face-size={compact ? 'compact' : 'standard'}
      data-tile-asset={TILE_BACK_ASSET_URL}
      style={{
        display: 'inline-flex',
        width,
        aspectRatio: '3 / 4',
        flex: '0 0 auto',
      }}
    >
      <img
        alt=""
        draggable={false}
        src={TILE_BACK_ASSET_URL}
        style={{
          display: 'block',
          width: '100%',
          height: '100%',
          objectFit: 'contain',
          userSelect: 'none',
        }}
      />
    </span>
  );
}
