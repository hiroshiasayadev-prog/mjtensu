export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type RegionKey = 'completed_hand' | 'dora_indicators' | 'melds';

export interface CaptureLayoutDocument {
  id: string;
  composite: {
    width: number;
    height: number;
    paddingRgb: [number, number, number];
  };
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

export interface TileSlot {
  ordinal: number;
  tile: string;
  face: 'front' | 'back';
  rotation: number;
}

export interface CatalogRow {
  label: string;
  region: RegionKey;
  tiles: string[];
}

export interface CatalogCaptureTask {
  id: string;
  campaignId: string;
  layoutId: string;
  layoutOrdinal: number;
  layoutVersion: string;
  hand: TileSlot[];
  dora: {
    visible: TileSlot[];
    ura: TileSlot[];
  };
  melds: Array<{
    ordinal: number;
    kind: string;
    label?: string;
    tiles: TileSlot[];
  }>;
  environment: {
    lighting: string;
    brightness: string;
    shadow: string;
    cameraPose: string;
    variantId: string;
    label: string;
    instruction: string;
  };
  expected: {
    hand: number;
    dora: number;
    meld: number;
  };
  taskOrder: number;
  catalogRows: CatalogRow[];
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

export interface CatalogCaptureManifest extends CaptureManifest {
  catalog: {
    schemaVersion: 2;
    variantId: string;
    rows: CatalogRow[];
    smartphoneDetector: 'visual-only';
    annotationDetector: 'pc-after-upload';
  };
}

export interface CatalogCaptureDraft {
  manifest: CatalogCaptureManifest;
  original: Blob;
  composite: Blob;
  previewUrl: string;
}

export interface PendingCatalogCapture {
  id: string;
  manifest: CatalogCaptureManifest;
  original: Blob;
  composite: Blob;
}
