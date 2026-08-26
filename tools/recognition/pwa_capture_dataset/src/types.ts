export type TileCode =
  | '1m' | '2m' | '3m' | '4m' | '5m' | '6m' | '7m' | '8m' | '9m'
  | '1p' | '2p' | '3p' | '4p' | '5p' | '6p' | '7p' | '8p' | '9p'
  | '1s' | '2s' | '3s' | '4s' | '5s' | '6s' | '7s' | '8s' | '9s'
  | 'east' | 'south' | 'west' | 'north' | 'white' | 'green' | 'red'
  | 'red5m' | 'red5p' | 'red5s';

export type TileFace = 'front' | 'back';
export type TileRotation = 0 | 90 | 180 | 270;

export interface TileSlot {
  ordinal: number;
  tile: TileCode;
  face: TileFace;
  rotation: TileRotation;
}

export interface MeldLayout {
  ordinal: number;
  kind: 'chi' | 'pon' | 'open-kan' | 'closed-kan';
  tiles: TileSlot[];
}

export interface CaptureTask {
  id: string;
  campaignId: string;
  layoutId: string;
  layoutOrdinal: number;
  hand: TileSlot[];
  dora: { visible: TileSlot[]; ura: TileSlot[] };
  melds: MeldLayout[];
  environment: {
    brightness: 'bright' | 'dark';
    shadow: 'none' | 'partial';
  };
  expected: { hand: number; dora: number; meld: number };
}

export interface CampaignOverview {
  campaignId: string;
  name: string;
  totalTasks: number;
  completedTasks: number;
  pendingTasks: number;
  completedLayouts: number;
  totalLayouts: number;
  coverage: Record<string, number>;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type RegionKey = 'completed_hand' | 'dora_indicators' | 'melds';

export interface CaptureLayoutDocument {
  id: string;
  composite: { width: number; height: number; paddingRgb: [number, number, number] };
  regions: Record<RegionKey, {
    label: string;
    sourceAspect: [number, number];
    destination: Rect;
  }>;
}

export interface ModelMetadata {
  name: string;
  source: string;
  sizeBytes: number;
  sha256: string;
  inputShape: number[];
  outputShape: number[];
  generatedAt: string;
}

export interface DetectionRecord {
  detectionIndex: number;
  region: RegionKey | 'invalid';
  confidence: number;
  composite: Rect;
  original: Rect | null;
  preview: Rect | null;
}

export interface CaptureManifest {
  uploadClientId: string;
  taskId: string;
  campaignId: string;
  capturedAt: string;
  original: { width: number; height: number };
  preview: {
    width: number;
    height: number;
    devicePixelRatio: number;
    videoElement: Rect;
    sourceToDisplayScale: number;
    sourceDisplayOffsetX: number;
    sourceDisplayOffsetY: number;
  };
  model: ModelMetadata;
  layoutVersion: string;
  confidenceThreshold: number;
  nmsIouThreshold: number;
  provider: string;
  camera: MediaTrackSettings;
  telemetry: Record<string, unknown>;
  regionRects: Record<RegionKey, {
    enabled: boolean;
    pixel: Rect;
    normalized: Rect;
    display: Rect;
  }>;
  detections: DetectionRecord[];
}

export interface CaptureDraft {
  manifest: CaptureManifest;
  original: Blob;
  composite: Blob;
  handCrop: Blob | null;
  doraCrop: Blob | null;
  meldCrop: Blob | null;
  previewUrl: string;
  compositePreviewUrl: string;
  regionUrls: Partial<Record<RegionKey, string>>;
}

export interface PendingCapture {
  id: string;
  manifest: CaptureManifest;
  original: Blob;
  composite: Blob;
  handCrop: Blob | null;
  doraCrop: Blob | null;
  meldCrop: Blob | null;
}
