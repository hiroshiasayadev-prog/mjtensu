import { Box, Text } from '@mantine/core';
import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';

import { navigateToTop } from './navigation';
import {
  CaptureRegionOverlay,
  RECOGNITION_CAPTURE_REGIONS,
  captureSurfaceLayout,
  landscapeUiSurfaceLayout,
  usePortraitViewport,
} from './recognition-page';

const RECORDING_DURATION_MS = 30_000;
const TAKE_COUNT = 5;

type CameraStatus = 'preparing' | 'ready' | 'failed';
type RecordingStatus = 'idle' | 'recording' | 'ready';

interface RecordedTake {
  readonly video: File;
  readonly metadata: File;
}

export function IntegrationCapturePage() {
  const navigate = useNavigate();
  const isPortraitViewport = usePortraitViewport();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const recordingStartedAtRef = useRef<number | null>(null);
  const stopTimerRef = useRef<number | null>(null);
  const elapsedTimerRef = useRef<number | null>(null);
  const mountedRef = useRef(true);
  const [cameraStatus, setCameraStatus] = useState<CameraStatus>('preparing');
  const [recordingStatus, setRecordingStatus] = useState<RecordingStatus>('idle');
  const [takeNumber, setTakeNumber] = useState(1);
  const [elapsedMs, setElapsedMs] = useState(0);
  const [recordedTake, setRecordedTake] = useState<RecordedTake | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const clearRecordingTimers = useCallback(() => {
    if (stopTimerRef.current !== null) window.clearTimeout(stopTimerRef.current);
    if (elapsedTimerRef.current !== null) window.clearInterval(elapsedTimerRef.current);
    stopTimerRef.current = null;
    elapsedTimerRef.current = null;
  }, []);

  const stopRecording = useCallback(() => {
    clearRecordingTimers();
    const recorder = recorderRef.current;
    if (recorder !== null && recorder.state !== 'inactive') recorder.stop();
  }, [clearRecordingTimers]);

  useEffect(() => {
    mountedRef.current = true;
    let cancelled = false;
    const mediaDevices = navigator.mediaDevices;

    if (mediaDevices?.getUserMedia === undefined) {
      setCameraStatus('failed');
      setErrorMessage('このブラウザではカメラを利用できません。');
      return () => {
        mountedRef.current = false;
      };
    }

    void mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    }).then(async (stream) => {
      if (cancelled) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (video !== null) {
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play().catch(() => undefined);
      }
      if (mountedRef.current) setCameraStatus('ready');
    }, (error: unknown) => {
      if (!cancelled && mountedRef.current) {
        setCameraStatus('failed');
        setErrorMessage(cameraErrorMessage(error));
      }
    });

    return () => {
      cancelled = true;
      mountedRef.current = false;
      clearRecordingTimers();
      const recorder = recorderRef.current;
      if (recorder !== null && recorder.state !== 'inactive') recorder.stop();
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      if (videoRef.current !== null) videoRef.current.srcObject = null;
    };
  }, [clearRecordingTimers]);

  const startRecording = useCallback(() => {
    const stream = streamRef.current;
    if (stream === null || cameraStatus !== 'ready') return;
    if (typeof MediaRecorder === 'undefined') {
      setErrorMessage('このブラウザは動画録画に対応していません。');
      return;
    }

    setErrorMessage(null);
    setRecordedTake(null);
    setElapsedMs(0);
    chunksRef.current = [];
    const mimeType = selectRecordingMimeType();
    const recorder = mimeType === null
      ? new MediaRecorder(stream)
      : new MediaRecorder(stream, { mimeType });
    recorderRef.current = recorder;
    const take = takeNumber;
    const startedAtIso = new Date().toISOString();
    recordingStartedAtRef.current = performance.now();

    recorder.addEventListener('dataavailable', (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    });
    recorder.addEventListener('stop', () => {
      clearRecordingTimers();
      const endedAtIso = new Date().toISOString();
      const durationMs = recordingStartedAtRef.current === null
        ? RECORDING_DURATION_MS
        : Math.min(RECORDING_DURATION_MS, performance.now() - recordingStartedAtRef.current);
      recordingStartedAtRef.current = null;
      const type = recorder.mimeType || mimeType || 'video/webm';
      const extension = type.includes('mp4') ? 'mp4' : 'webm';
      const stamp = startedAtIso.replaceAll(':', '-').replaceAll('.', '-');
      const video = new File(
        chunksRef.current,
        `mjtensu-integration-take-${String(take).padStart(2, '0')}-${stamp}.${extension}`,
        { type },
      );
      const metadata = createMetadataFile({
        take,
        startedAtIso,
        endedAtIso,
        durationMs,
        mimeType: type,
        stream,
        isPortraitViewport,
      });
      if (mountedRef.current) {
        setElapsedMs(durationMs);
        setRecordedTake({ video, metadata });
        setRecordingStatus('ready');
      }
    });

    recorder.start(1000);
    setRecordingStatus('recording');
    elapsedTimerRef.current = window.setInterval(() => {
      const started = recordingStartedAtRef.current;
      if (started !== null && mountedRef.current) {
        setElapsedMs(Math.min(RECORDING_DURATION_MS, performance.now() - started));
      }
    }, 100);
    stopTimerRef.current = window.setTimeout(stopRecording, RECORDING_DURATION_MS);
  }, [
    cameraStatus,
    clearRecordingTimers,
    isPortraitViewport,
    stopRecording,
    takeNumber,
  ]);

  const resetTake = useCallback(() => {
    setRecordedTake(null);
    setRecordingStatus('idle');
    setElapsedMs(0);
    setErrorMessage(null);
  }, []);
  const advanceTake = useCallback(() => {
    setTakeNumber((current) => Math.min(TAKE_COUNT, current + 1));
    resetTake();
  }, [resetTake]);

  const saveTake = useCallback(() => {
    if (recordedTake === null) return;
    void shareOrDownloadFiles([recordedTake.video, recordedTake.metadata]).catch((error: unknown) => {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      if (mountedRef.current) {
        setErrorMessage(error instanceof Error ? error.message : String(error));
      }
    });
  }, [recordedTake]);

  const statusText = cameraStatus === 'preparing'
    ? 'カメラ準備中'
    : cameraStatus === 'failed'
      ? 'カメラ起動失敗'
      : recordingStatus === 'recording'
        ? 'REC'
        : recordingStatus === 'ready'
          ? '録画完了'
          : '撮影準備OK';

  return (
    <Box
      data-testid="integration-capture-viewport"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 1000,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden',
        background: '#000',
      }}
    >
      <Box
        data-testid="integration-capture-surface"
        style={{
          position: 'relative',
          ...captureSurfaceLayout(isPortraitViewport),
          aspectRatio: isPortraitViewport ? '9 / 16' : '16 / 9',
          overflow: 'hidden',
          background: '#111',
        }}
      >
        <video
          ref={videoRef}
          aria-label="撮影カメラプレビュー"
          autoPlay
          muted
          playsInline
          style={{
            position: 'absolute',
            inset: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            pointerEvents: 'none',
          }}
        />

        <Box
          data-testid="integration-capture-landscape-ui"
          style={{
            ...landscapeUiSurfaceLayout(isPortraitViewport),
            aspectRatio: '16 / 9',
            display: 'grid',
            pointerEvents: 'auto',
          }}
        >
          <CaptureRegionOverlay
            snapshot={null}
            regions={RECOGNITION_CAPTURE_REGIONS}
          />

          <Box style={HUD_STYLE}>
            <Text size="xs" fw={700}>Integration capture</Text>
            <Text size="xs">
              Take {takeNumber} / {TAKE_COUNT} · {formatSeconds(elapsedMs)} / 30.0s · {statusText}
            </Text>
          </Box>

          <button
            aria-label="撮影を終了"
            onClick={() => {
              stopRecording();
              navigateToTop(navigate);
            }}
            style={{ ...CONTROL_STYLE, position: 'absolute', right: '4%', top: '4%' }}
            type="button"
          >
            終了
          </button>

          <Box style={BOTTOM_ACTIONS_STYLE}>
            {recordingStatus === 'idle' ? (
              <button
                disabled={cameraStatus !== 'ready'}
                onClick={startRecording}
                style={PRIMARY_CONTROL_STYLE}
                type="button"
              >
                REC 30s
              </button>
            ) : null}
            {recordingStatus === 'recording' ? (
              <button onClick={stopRecording} style={PRIMARY_CONTROL_STYLE} type="button">
                停止
              </button>
            ) : null}
            {recordingStatus === 'ready' ? (
              <>
                <button onClick={resetTake} style={CONTROL_STYLE} type="button">
                  撮り直す
                </button>
                <button onClick={saveTake} style={PRIMARY_CONTROL_STYLE} type="button">
                  保存
                </button>
                {takeNumber < TAKE_COUNT ? (
                  <button onClick={advanceTake} style={CONTROL_STYLE} type="button">
                    次へ
                  </button>
                ) : null}
              </>
            ) : null}
          </Box>

          {errorMessage === null ? null : (
            <Box role="alert" style={ERROR_STYLE}>{errorMessage}</Box>
          )}
        </Box>
      </Box>
    </Box>
  );
}

function createMetadataFile({
  take,
  startedAtIso,
  endedAtIso,
  durationMs,
  mimeType,
  stream,
  isPortraitViewport,
}: {
  readonly take: number;
  readonly startedAtIso: string;
  readonly endedAtIso: string;
  readonly durationMs: number;
  readonly mimeType: string;
  readonly stream: MediaStream;
  readonly isPortraitViewport: boolean;
}): File {
  const track = stream.getVideoTracks()[0] ?? null;
  const payload = {
    schemaVersion: 1,
    take,
    targetDurationMs: RECORDING_DURATION_MS,
    durationMs,
    startedAtIso,
    endedAtIso,
    mimeType,
    trackSettings: track?.getSettings() ?? null,
    logicalCapture: {
      aspectRatio: isPortraitViewport ? '9:16' : '16:9',
      rotation: isPortraitViewport ? -90 : 0,
    },
    recognitionRegions: RECOGNITION_CAPTURE_REGIONS,
    userAgent: navigator.userAgent,
    viewport: {
      width: window.innerWidth,
      height: window.innerHeight,
      devicePixelRatio: window.devicePixelRatio,
    },
  };
  const stamp = startedAtIso.replaceAll(':', '-').replaceAll('.', '-');
  return new File(
    [JSON.stringify(payload, null, 2)],
    `mjtensu-integration-take-${String(take).padStart(2, '0')}-${stamp}.json`,
    { type: 'application/json' },
  );
}

async function shareOrDownloadFiles(files: File[]): Promise<void> {
  const shareNavigator = navigator as Navigator & {
    readonly canShare?: (data: { files?: File[] }) => boolean;
    readonly share?: (data: { files?: File[]; title?: string }) => Promise<void>;
  };
  if (
    shareNavigator.share !== undefined &&
    shareNavigator.canShare?.({ files }) === true
  ) {
    await shareNavigator.share({ files, title: 'mjtensu integration capture' });
    return;
  }

  for (const file of files) downloadFile(file);
}

function downloadFile(file: File): void {
  const url = URL.createObjectURL(file);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = file.name;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function selectRecordingMimeType(): string | null {
  const candidates = [
    'video/mp4;codecs=avc1.42E01E',
    'video/mp4',
    'video/webm;codecs=vp8',
    'video/webm',
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? null;
}

function formatSeconds(milliseconds: number): string {
  return (milliseconds / 1000).toFixed(1);
}

function cameraErrorMessage(error: unknown): string {
  if (error instanceof DOMException) return `${error.name}: ${error.message}`;
  return error instanceof Error ? error.message : String(error);
}

const CONTROL_STYLE = {
  pointerEvents: 'auto',
  touchAction: 'manipulation',
  appearance: 'none',
  border: '1px solid rgba(0,0,0,0.12)',
  borderRadius: 6,
  padding: '7px 12px',
  minHeight: 34,
  background: 'rgba(255,255,255,0.92)',
  color: '#1b1d22',
  font: 'inherit',
  fontSize: 13,
  fontWeight: 700,
  lineHeight: 1.2,
} as const;
const PRIMARY_CONTROL_STYLE = {
  ...CONTROL_STYLE,
  minWidth: 88,
  background: 'rgba(255,255,255,0.98)',
  boxShadow: '0 0 0 2px rgba(220,0,0,0.72)',
} as const;

const HUD_STYLE = {
  position: 'absolute',
  left: '4%',
  top: '4%',
  zIndex: 30,
  padding: '5px 8px',
  borderRadius: 6,
  background: 'rgba(0,0,0,0.68)',
  color: '#fff',
  lineHeight: 1.15,
  pointerEvents: 'none',
} as const;

const BOTTOM_ACTIONS_STYLE = {
  position: 'absolute',
  left: '50%',
  bottom: '4%',
  zIndex: 30,
  transform: 'translateX(-50%)',
  display: 'flex',
  gap: 8,
  alignItems: 'center',
  pointerEvents: 'none',
} as const;

const ERROR_STYLE = {
  position: 'absolute',
  left: '50%',
  top: '50%',
  zIndex: 40,
  transform: 'translate(-50%, -50%)',
  maxWidth: '70%',
  padding: '8px 10px',
  borderRadius: 6,
  background: 'rgba(120,0,0,0.9)',
  color: '#fff',
  fontSize: 12,
} as const;
