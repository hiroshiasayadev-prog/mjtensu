export type TileKind =
  | '1m'
  | '2m'
  | '3m'
  | '4m'
  | '5m'
  | '6m'
  | '7m'
  | '8m'
  | '9m'
  | '1p'
  | '2p'
  | '3p'
  | '4p'
  | '5p'
  | '6p'
  | '7p'
  | '8p'
  | '9p'
  | '1s'
  | '2s'
  | '3s'
  | '4s'
  | '5s'
  | '6s'
  | '7s'
  | '8s'
  | '9s'
  | '1z'
  | '2z'
  | '3z'
  | '4z'
  | '5z'
  | '6z'
  | '7z';

export interface TileIdentity {
  readonly kind: TileKind;
  readonly red: boolean;
}

export type TileInstanceId = string & {
  readonly __brand: 'TileInstanceId';
};

export interface TileInstance {
  readonly id: TileInstanceId;
  readonly tile: TileIdentity;
}

export type RecognizedMeldGroup =
  | {
      readonly kind: 'chi';
      readonly tiles: readonly [TileInstance, TileInstance, TileInstance];
    }
  | {
      readonly kind: 'pon';
      readonly tiles: readonly [TileInstance, TileInstance, TileInstance];
    }
  | {
      readonly kind: 'open-kan';
      readonly tiles: readonly [
        TileInstance,
        TileInstance,
        TileInstance,
        TileInstance,
      ];
    }
  | {
      readonly kind: 'concealed-kan';
      readonly tiles: readonly [
        TileInstance,
        TileInstance,
        TileInstance,
        TileInstance,
      ];
    }
  | {
      readonly kind: 'unresolved';
      readonly tiles: readonly TileInstance[];
    };

export interface RecognizedStructure {
  readonly completedHand: readonly TileInstance[];
  readonly doraIndicators: readonly TileInstance[];
  readonly meldGroups: readonly RecognizedMeldGroup[];
}
