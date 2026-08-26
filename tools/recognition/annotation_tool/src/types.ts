export type RegionKey = 'completed_hand' | 'dora_indicators' | 'melds';
export type AnnotationStatus = 'unannotated' | 'draft' | 'complete';

export interface TileSlot {
  ordinal: number;
  tile: string;
  face: 'front' | 'back';
  rotation: number;
}

export interface MeldTask {
  ordinal: number;
  kind: 'chi' | 'pon' | 'open-kan' | 'closed-kan' | 'catalog-row';
  label?: string;
  tiles: TileSlot[];
}

export interface CaptureTask {
  id: string;
  campaignId: string;
  layoutId: string;
  layoutOrdinal: number;
  hand: TileSlot[];
  dora: {
    visible: TileSlot[];
    ura: TileSlot[];
  };
  melds: MeldTask[];
  environment: {
    brightness: string;
    shadow: string;
    lighting?: string;
    cameraPose?: string;
    variantId?: string;
    label?: string;
    instruction?: string;
  };
  expected: {
    hand: number;
    dora: number;
    meld: number;
  };
  taskOrder: number;
}

export interface CampaignSummary {
  campaignId: string;
  name: string;
  captureCount: number;
  completeCount: number;
  draftCount: number;
}

export interface CaptureSummary {
  captureId: string;
  taskId: string;
  capturedAt: string;
  layoutId: string;
  layoutOrdinal: number;
  environment: {
    brightness: string;
    shadow: string;
    label?: string | null;
  };
  taskOrder: number;
  annotationStatus: AnnotationStatus;
  annotationUpdatedAt: string | null;
}

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectionSummary {
  detectionIndex: number;
  region: RegionKey | 'invalid';
  confidence: number;
  original: Rect | null;
}

export interface AnnotationBox {
  id: string;
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  angleDeg: number;
}

export interface AnnotationDocument {
  schemaVersion: 1;
  captureId: string;
  boxes: Record<RegionKey, AnnotationBox[]>;
}

export interface CaptureDetail {
  captureId: string;
  taskId: string;
  campaignId: string;
  taskOrder: number;
  task: CaptureTask;
  manifest: {
    original: { width: number; height: number };
    regionRects: Record<RegionKey, {
      enabled: boolean;
      pixel: Rect;
      normalized: Rect;
      display: Rect;
    }>;
  };
  original: {
    path: string;
    width: number;
    height: number;
  };
  regionPaths: Record<RegionKey, string | null>;
  detections: DetectionSummary[];
  annotation: null | {
    status: 'draft' | 'complete';
    schemaVersion: number;
    updatedAt: string;
    document: AnnotationDocument;
  };
}

export interface GroupAssignment {
  label: string;
  expected: TileSlot[];
  boxes: AnnotationBox[];
  valid: boolean;
}

export interface RegionAssignment {
  labels: Map<string, { text: string; tentative: boolean }>;
  groups: GroupAssignment[];
  expectedCount: number;
  actualCount: number;
  valid: boolean;
}

export interface ValidationResult {
  regions: Record<RegionKey, RegionAssignment>;
  allInside: boolean;
  complete: boolean;
}
