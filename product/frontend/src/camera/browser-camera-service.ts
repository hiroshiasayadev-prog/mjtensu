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
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext('2d');
    if (context === null) {
      return null;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);
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

    this.attachedVideo = video;
    video.srcObject = this.stream;
    video.muted = true;
    video.playsInline = true;
    void video.play().catch(() => {
      // Presentation readiness is reflected by captureLatest() returning null
      // until the browser exposes a usable current frame.
    });
  }

  private detachPreview(): void {
    const video = this.attachedVideo;
    if (video === null) {
      return;
    }

    this.attachedVideo = null;
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
