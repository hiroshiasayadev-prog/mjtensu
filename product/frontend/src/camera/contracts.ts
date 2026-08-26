export interface Size {
  readonly width: number;
  readonly height: number;
}

export interface CameraOpenRequest {
  readonly facingMode: 'environment';
}

export interface CameraFrame {
  readonly image: CanvasImageSource;
  readonly size: Size;
  readonly capturedAtMs: number;
}

export interface CameraPreview {
  attach(video: HTMLVideoElement): void;
  detach(): void;
}

export interface CameraSession {
  readonly preview: CameraPreview;

  captureLatest(): CameraFrame | null;
  stop(): Promise<void>;
}

export interface CameraService {
  open(request: CameraOpenRequest): Promise<CameraSession>;
}

export type CameraRuntimeError =
  | { readonly kind: 'permission-denied' }
  | { readonly kind: 'device-not-found' }
  | { readonly kind: 'device-unavailable' }
  | { readonly kind: 'unsupported' }
  | { readonly kind: 'runtime-failure'; readonly cause: unknown };
