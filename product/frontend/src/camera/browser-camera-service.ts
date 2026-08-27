import type {
  CameraFrame,
  CameraOpenRequest,
  CameraPreview,
  CameraRuntimeError,
  CameraService,
  CameraSession,
} from './contracts';

export interface BrowserCameraServiceOptions {
  readonly mediaDevices?: Pick<MediaDevices, 'getUserMedia'>;
  readonly createCanvas?: () => HTMLCanvasElement;
  readonly now?: () => number;
}

export function createBrowserCameraService(
  options: BrowserCameraServiceOptions = {},
): CameraService {
  return {
    async open(request: CameraOpenRequest): Promise<CameraSession> {
      const mediaDevices = options.mediaDevices ?? globalThis.navigator?.mediaDevices;
      if (mediaDevices?.getUserMedia === undefined) {
        throw { kind: 'unsupported' } satisfies CameraRuntimeError;
      }

      let stream: MediaStream;
      try {
        stream = await mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: request.facingMode },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });
      } catch (cause) {
        throw normalizeCameraOpenError(cause);
      }

      return new BrowserCameraSession(
        stream,
        options.createCanvas ?? (() => document.createElement('canvas')),
        options.now ?? (() => performance.now()),
      );
    },
  };
}

class BrowserCameraSession implements CameraSession {
  readonly preview: CameraPreview;
  private attachedVideo: HTMLVideoElement | null = null;
  private playbackRetryTarget: HTMLVideoElement | null = null;
  private stopped = false;

  constructor(
    private readonly stream: MediaStream,
    private readonly createCanvas: () => HTMLCanvasElement,
    private readonly now: () => number,
  ) {
    this.preview = {
      attach: (video) => this.attachPreview(video),
      detach: () => this.detachPreview(),
    };
  }

  captureLatest(): CameraFrame | null {
    const video = this.attachedVideo;
    if (
      this.stopped ||
      video === null ||
      video.readyState < 2 ||
      video.videoWidth <= 0 ||
      video.videoHeight <= 0
    ) {
      return null;
    }

    const canvas = this.createCanvas();
    const captureSize = canonicalCaptureSize(video.videoWidth);
    canvas.width = captureSize.width;
    canvas.height = captureSize.height;
    const context = canvas.getContext('2d');
    if (context === null) {
      return null;
    }

    try {
      // Recognition always consumes the same 16:9 geometry that the preview
      // exposes with object-fit: cover. Center-crop the raw MediaStream frame
      // instead of stretching or rotating it; viewport orientation is a UI
      // layout concern and must not alter camera coordinates.
      drawCanonicalFrame(
        context,
        video,
        video.videoWidth,
        video.videoHeight,
        canvas.width,
        canvas.height,
      );
    } catch (error) {
      if (isTransientVideoFrameError(error)) {
        // Safari can expose readyState/video dimensions just before the first
        // decoded frame is drawable after camera permission. This is warmup,
        // not a Recognition inference failure; the realtime loop will retry.
        return null;
      }
      throw error;
    }
    return {
      image: canvas,
      size: { width: canvas.width, height: canvas.height },
      capturedAtMs: this.now(),
    };
  }

  async stop(): Promise<void> {
    if (this.stopped) {
      return;
    }
    this.stopped = true;
    this.detachPreview();
    for (const track of this.stream.getTracks()) {
      track.stop();
    }
  }

  private attachPreview(video: HTMLVideoElement): void {
    if (this.stopped) {
      return;
    }

    if (this.attachedVideo !== null && this.attachedVideo !== video) {
      this.detachPreview();
    }

    this.clearPlaybackRetry();
    this.attachedVideo = video;
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    if (video.readyState >= 2) {
      this.startPreviewPlayback(video);
    } else {
      this.waitForPreviewData(video);
    }
  }

  private startPreviewPlayback(video: HTMLVideoElement): void {
    void video.play().catch(() => {
      if (this.stopped || this.attachedVideo !== video) {
        return;
      }

      // A newly-permitted iOS camera may still be transitioning even after
      // media data becomes visible. Retry once without reopening the healthy
      // camera session; captureLatest() keeps returning null during warmup.
      void Promise.resolve().then(() => {
        if (this.stopped || this.attachedVideo !== video) {
          return;
        }
        void video.play().catch(() => undefined);
      });
    });
  }

  private waitForPreviewData(video: HTMLVideoElement): void {
    this.clearPlaybackRetry();
    this.playbackRetryTarget = video;
    video.addEventListener(
      'loadeddata',
      this.retryPreviewPlayback,
      { once: true },
    );
  }

  private readonly retryPreviewPlayback = () => {
    const video = this.playbackRetryTarget;
    this.playbackRetryTarget = null;
    if (video === null || this.stopped || this.attachedVideo !== video) {
      return;
    }
    this.startPreviewPlayback(video);
  };

  private clearPlaybackRetry(): void {
    const video = this.playbackRetryTarget;
    if (video === null) {
      return;
    }
    this.playbackRetryTarget = null;
    video.removeEventListener('loadeddata', this.retryPreviewPlayback);
  }

  private detachPreview(): void {
    const video = this.attachedVideo;
    if (video === null) {
      return;
    }

    this.attachedVideo = null;
    this.clearPlaybackRetry();
    try {
      video.pause();
    } catch {
      // Some test/browser implementations do not provide a functional pause().
    }
    if (video.srcObject === this.stream) {
      video.srcObject = null;
    }
  }
}

function normalizeCameraOpenError(cause: unknown): CameraRuntimeError {
  const name = domExceptionName(cause);

  switch (name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return { kind: 'permission-denied' };

    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return { kind: 'device-not-found' };

    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return { kind: 'device-unavailable' };

    case 'OverconstrainedError':
    case 'ConstraintNotSatisfiedError':
    case 'TypeError':
      return { kind: 'unsupported' };

    default:
      return { kind: 'runtime-failure', cause };
  }
}

function canonicalCaptureSize(
  videoWidth: number,
): { readonly width: number; readonly height: number } {
  const width = Math.max(1, Math.min(1280, Math.round(videoWidth)));
  return {
    width,
    height: Math.max(1, Math.round(width * 9 / 16)),
  };
}

function drawCanonicalFrame(
  context: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): void {
  const crop = centerCrop(sourceWidth, sourceHeight, 16 / 9);
  context.drawImage(
    video,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    targetWidth,
    targetHeight,
  );
}

function centerCrop(
  width: number,
  height: number,
  targetAspect: number,
): { readonly x: number; readonly y: number; readonly width: number; readonly height: number } {
  const sourceAspect = width / height;
  if (sourceAspect > targetAspect) {
    const cropWidth = height * targetAspect;
    return {
      x: (width - cropWidth) / 2,
      y: 0,
      width: cropWidth,
      height,
    };
  }

  const cropHeight = width / targetAspect;
  return {
    x: 0,
    y: (height - cropHeight) / 2,
    width,
    height: cropHeight,
  };
}

function isTransientVideoFrameError(value: unknown): boolean {
  const name = domExceptionName(value);
  return name === 'InvalidStateError' || name === 'AbortError';
}

function domExceptionName(value: unknown): string | null {
  if (
    typeof value === 'object' &&
    value !== null &&
    'name' in value &&
    typeof value.name === 'string'
  ) {
    return value.name;
  }
  return null;
}
