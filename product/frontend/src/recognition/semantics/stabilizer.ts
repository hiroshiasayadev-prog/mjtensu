import type { TileIdentity } from '@/domain';

import type {
  FrameMeldInterpretation,
  FrameRecognitionDraft,
  FrameRecognitionSnapshot,
} from './types';

const REQUIRED_CONSECUTIVE_RESULTS = 3;

export type SemanticStabilizationState =
  | {
      readonly kind: 'scanning';
    }
  | {
      readonly kind: 'stabilizing';
      readonly candidate: FrameRecognitionDraft;
      readonly consecutive: number;
    }
  | {
      readonly kind: 'confirmed';
      readonly draft: FrameRecognitionDraft;
    };

export class RecognitionSemanticStabilizer {
  private candidate: FrameRecognitionDraft | null = null;
  private consecutive = 0;
  private confirmed: FrameRecognitionDraft | null = null;

  accept(snapshot: FrameRecognitionSnapshot): SemanticStabilizationState {
    if (this.confirmed !== null) {
      return { kind: 'confirmed', draft: this.confirmed };
    }

    if (snapshot.commitEligibility.kind !== 'eligible') {
      this.clearCandidate();
      return { kind: 'scanning' };
    }

    if (
      this.candidate === null ||
      !areFrameRecognitionDraftsEqual(this.candidate, snapshot.draft)
    ) {
      this.candidate = cloneDraft(snapshot.draft);
      this.consecutive = 1;
      return {
        kind: 'stabilizing',
        candidate: this.candidate,
        consecutive: this.consecutive,
      };
    }

    this.consecutive += 1;
    if (this.consecutive >= REQUIRED_CONSECUTIVE_RESULTS) {
      this.confirmed = this.candidate;
      return { kind: 'confirmed', draft: this.confirmed };
    }

    return {
      kind: 'stabilizing',
      candidate: this.candidate,
      consecutive: this.consecutive,
    };
  }

  reset(): void {
    this.confirmed = null;
    this.clearCandidate();
  }

  getState(): SemanticStabilizationState {
    if (this.confirmed !== null) {
      return { kind: 'confirmed', draft: this.confirmed };
    }
    if (this.candidate !== null) {
      return {
        kind: 'stabilizing',
        candidate: this.candidate,
        consecutive: this.consecutive,
      };
    }
    return { kind: 'scanning' };
  }

  private clearCandidate(): void {
    this.candidate = null;
    this.consecutive = 0;
  }
}

export function areFrameRecognitionDraftsEqual(
  left: FrameRecognitionDraft,
  right: FrameRecognitionDraft,
): boolean {
  return (
    tileArraysEqual(left.completedHand, right.completedHand) &&
    tileArraysEqual(left.doraIndicators, right.doraIndicators) &&
    meldArraysEqual(left.meldGroups, right.meldGroups)
  );
}

function meldArraysEqual(
  left: readonly FrameMeldInterpretation[],
  right: readonly FrameMeldInterpretation[],
): boolean {
  return (
    left.length === right.length &&
    left.every((meld, index) => {
      const other = right[index];
      return (
        other !== undefined &&
        meld.kind === other.kind &&
        tileArraysEqual(meld.tiles, other.tiles)
      );
    })
  );
}

function tileArraysEqual(
  left: readonly TileIdentity[],
  right: readonly TileIdentity[],
): boolean {
  return (
    left.length === right.length &&
    left.every((tile, index) => {
      const other = right[index];
      return other !== undefined && tile.kind === other.kind && tile.red === other.red;
    })
  );
}

function cloneDraft(draft: FrameRecognitionDraft): FrameRecognitionDraft {
  return {
    completedHand: draft.completedHand.map(cloneTile),
    doraIndicators: draft.doraIndicators.map(cloneTile),
    meldGroups: draft.meldGroups.map((meld) => ({
      kind: meld.kind,
      tiles: meld.tiles.map(cloneTile),
    }) as FrameMeldInterpretation),
  };
}

function cloneTile(tile: TileIdentity): TileIdentity {
  return { kind: tile.kind, red: tile.red };
}
