export const SEMANTIC_REGIONS = [
  'completed_hand',
  'dora_indicators',
  'melds',
] as const;

export type SemanticRegion = (typeof SEMANTIC_REGIONS)[number];

export interface Rect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface CaptureRegion {
  readonly enabled: boolean;
  readonly sourceRect: Rect;
}

export type CaptureRegions = Readonly<Record<SemanticRegion, CaptureRegion>>;

export interface TensorOutput {
  readonly dims: readonly number[];
  readonly data: unknown;
  readonly type: string;
}

export interface DecodedDetection {
  readonly id: string;
  readonly detectionIndex: number;
  readonly classIndex: number;
  readonly confidence: number;
  readonly box: Rect;
}

export interface RegionDetection extends DecodedDetection {
  readonly region: SemanticRegion;
  readonly sourceBox: Rect;
}
