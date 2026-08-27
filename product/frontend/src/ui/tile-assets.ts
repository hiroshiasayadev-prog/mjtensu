import type { TileIdentity, TileKind } from '@/domain';

type TileAssetKey = TileKind | '5m-red' | '5p-red' | '5s-red';

const TILE_ASSET_URLS: Readonly<Record<TileAssetKey, string>> = {
  '1m': new URL('../../public/tiles/1m.svg', import.meta.url).href,
  '2m': new URL('../../public/tiles/2m.svg', import.meta.url).href,
  '3m': new URL('../../public/tiles/3m.svg', import.meta.url).href,
  '4m': new URL('../../public/tiles/4m.svg', import.meta.url).href,
  '5m': new URL('../../public/tiles/5m.svg', import.meta.url).href,
  '5m-red': new URL('../../public/tiles/5m-red.svg', import.meta.url).href,
  '6m': new URL('../../public/tiles/6m.svg', import.meta.url).href,
  '7m': new URL('../../public/tiles/7m.svg', import.meta.url).href,
  '8m': new URL('../../public/tiles/8m.svg', import.meta.url).href,
  '9m': new URL('../../public/tiles/9m.svg', import.meta.url).href,
  '1p': new URL('../../public/tiles/1p.svg', import.meta.url).href,
  '2p': new URL('../../public/tiles/2p.svg', import.meta.url).href,
  '3p': new URL('../../public/tiles/3p.svg', import.meta.url).href,
  '4p': new URL('../../public/tiles/4p.svg', import.meta.url).href,
  '5p': new URL('../../public/tiles/5p.svg', import.meta.url).href,
  '5p-red': new URL('../../public/tiles/5p-red.svg', import.meta.url).href,
  '6p': new URL('../../public/tiles/6p.svg', import.meta.url).href,
  '7p': new URL('../../public/tiles/7p.svg', import.meta.url).href,
  '8p': new URL('../../public/tiles/8p.svg', import.meta.url).href,
  '9p': new URL('../../public/tiles/9p.svg', import.meta.url).href,
  '1s': new URL('../../public/tiles/1s.svg', import.meta.url).href,
  '2s': new URL('../../public/tiles/2s.svg', import.meta.url).href,
  '3s': new URL('../../public/tiles/3s.svg', import.meta.url).href,
  '4s': new URL('../../public/tiles/4s.svg', import.meta.url).href,
  '5s': new URL('../../public/tiles/5s.svg', import.meta.url).href,
  '5s-red': new URL('../../public/tiles/5s-red.svg', import.meta.url).href,
  '6s': new URL('../../public/tiles/6s.svg', import.meta.url).href,
  '7s': new URL('../../public/tiles/7s.svg', import.meta.url).href,
  '8s': new URL('../../public/tiles/8s.svg', import.meta.url).href,
  '9s': new URL('../../public/tiles/9s.svg', import.meta.url).href,
  '1z': new URL('../../public/tiles/1z.svg', import.meta.url).href,
  '2z': new URL('../../public/tiles/2z.svg', import.meta.url).href,
  '3z': new URL('../../public/tiles/3z.svg', import.meta.url).href,
  '4z': new URL('../../public/tiles/4z.svg', import.meta.url).href,
  '5z': new URL('../../public/tiles/5z.svg', import.meta.url).href,
  '6z': new URL('../../public/tiles/6z.svg', import.meta.url).href,
  '7z': new URL('../../public/tiles/7z.svg', import.meta.url).href,
};

export const TILE_BACK_ASSET_URL = new URL(
  '../../public/tiles/back.svg',
  import.meta.url,
).href;

export function tileAssetUrl(tile: TileIdentity): string {
  const assetKey: TileAssetKey = tile.red && /^5[mps]$/.test(tile.kind)
    ? `${tile.kind}-red` as TileAssetKey
    : tile.kind;
  return TILE_ASSET_URLS[assetKey];
}
