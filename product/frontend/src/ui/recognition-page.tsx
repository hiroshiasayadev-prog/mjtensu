import {
  Box,
  Button,
  Group,
  Paper,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import type { RecognizedStructure, TileIdentity } from '@/domain';
import type { RecognitionRuntimeError } from '@/recognition';
import { DEFAULT_RULE_PROFILE } from '@/scoring';

import { useApplicationStore } from './application-state';
import {
  navigateAfterRecognitionConfirmed,
  navigateToTop,
} from './navigation';

export type RecognitionRegion =
  | 'completed-hand'
  | 'dora-indicators'
  | 'melds';

export interface NormalizedRect {
  readonly x: number;
  readonly y: number;
  readonly width: number;
  readonly height: number;
}

export interface RecognitionPageSize {
  readonly width: number;
  readonly height: number;
}

export interface RecognitionPageCameraFrame {
  readonly image: CanvasImageSource;
  readonly size: RecognitionPageSize;
  readonly capturedAtMs: number;
}

export type RecognitionPageCameraError =
  | { readonly kind: 'permission-denied' }
  | { readonly kind: 'device-not-found' }
  | { readonly kind: 'device-unavailable' }
  | { readonly kind: 'unsupported' }
  | { readonly kind: 'runtime-failure'; readonly cause: unknown };

export interface RecognitionPageCameraPreview {
  attach(video: HTMLVideoElement): void;
  detach(): void;
}

export interface RecognitionPageCameraSession {
  readonly preview: RecognitionPageCameraPreview;
  captureLatest(): RecognitionPageCameraFrame | null;
  stop(): Promise<void>;
}

export interface RecognitionPageCameraService {
  open(request: {
    readonly facingMode: 'environment';
  }): Promise<RecognitionPageCameraSession>;
}

export interface RecognitionPageRuntime {
  initialize(): Promise<void>;
}

export interface RecognitionPageFrame {
  readonly source: CanvasImageSource;
  readonly sourceSize: RecognitionPageSize;
  readonly regions: Readonly<Record<RecognitionRegion, NormalizedRect>>;
  readonly capturedAtMs: number;
}

export interface RecognitionPageFrameSource {
  captureLatest(): RecognitionPageFrame | null;
}

export interface RecognitionObservation {
  readonly id: string;
  readonly region: RecognitionRegion;
  readonly box: NormalizedRect;
  readonly tile: TileIdentity | null;
}

export type RecognitionMeldInterpretation =
  | 'chi'
  | 'pon'
  | 'open-kan'
  | 'concealed-kan'
  | 'unresolved';

export interface RecognitionMeldObservationGroup {
  readonly id: string;
  readonly memberObservationIds: readonly string[];
  readonly interpretation: RecognitionMeldInterpretation | null;
}

export interface RecognitionFrameSnapshot {
  readonly observations: readonly RecognitionObservation[];
  readonly meldGroups: readonly RecognitionMeldObservationGroup[];
}

export type RecognitionPageRealtimeUpdate =
  | {
      readonly kind: 'scanning';
      readonly snapshot: RecognitionFrameSnapshot;
    }
  | {
      readonly kind: 'stabilizing';
      readonly snapshot: RecognitionFrameSnapshot;
    }
  | {
      readonly kind: 'confirmed';
      readonly result: RecognizedStructure;
    };

export interface RecognitionPageRealtimeListener {
  onUpdate(update: RecognitionPageRealtimeUpdate): void;
  onError(error: RecognitionRuntimeError): void;
}

export interface RecognitionPageRun {
  stop(): void;
}

export interface RecognitionPageRealtimeRecognizer {
  start(
    source: RecognitionPageFrameSource,
    listener: RecognitionPageRealtimeListener,
  ): RecognitionPageRun;
  reset(): void;
}

export interface RecognitionPageServices {
  readonly camera: RecognitionPageCameraService;
  readonly runtime: RecognitionPageRuntime;
  readonly recognizer: RecognitionPageRealtimeRecognizer;
}

export interface RecognitionPageServicesProviderProps {
  readonly children: ReactNode;
  readonly services: RecognitionPageServices;
}

const RecognitionPageServicesContext =
  createContext<RecognitionPageServices | null>(null);

export function RecognitionPageServicesProvider({
  children,
  services,
}: RecognitionPageServicesProviderProps) {
  return (
    <RecognitionPageServicesContext.Provider value={services}>
      {children}
    </RecognitionPageServicesContext.Provider>
  );
}

export const RECOGNITION_CAPTURE_REGIONS = {
  'dora-indicators': {
    x: 0.04,
    y: 0.08,
    width: 0.62,
    height: 0.2593464052,
  },
  'completed-hand': {
    x: 0.04,
    y: 0.6606535948,
    width: 0.62,
    height: 0.2593464052,
  },
  melds: {
    x: 0.72,
    y: 0.2866666667,
    width: 0.24,
    height: 0.4266666667,
  },
} as const satisfies Readonly<Record<RecognitionRegion, NormalizedRect>>;

export interface RecognitionPageViewProps {
  readonly camera: RecognitionPageCameraService;
  readonly runtime: RecognitionPageRuntime;
  readonly recognizer: RecognitionPageRealtimeRecognizer;
  readonly onAbandon: () => void;
  readonly onConfirmed: (result: RecognizedStructure) => void;
}

type PreparationStatus = 'preparing' | 'ready' | 'failed';

export function RecognitionPageView({
  camera,
  runtime,
  recognizer,
  onAbandon,
  onConfirmed,
}: RecognitionPageViewProps) {
  const [cameraStatus, setCameraStatus] =
    useState<PreparationStatus>('preparing');
  const [runtimeStatus, setRuntimeStatus] =
    useState<PreparationStatus>('preparing');
  const [cameraError, setCameraError] = useState<unknown>(null);
  const [runtimeError, setRuntimeError] = useState<unknown>(null);
  const [cameraSession, setCameraSession] =
    useState<RecognitionPageCameraSession | null>(null);
  const [snapshot, setSnapshot] = useState<RecognitionFrameSnapshot | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mountedRef = useRef(false);
  const cameraAttemptRef = useRef(0);
  const runtimeAttemptRef = useRef(0);
  const committedRef = useRef(false);
  const isLandscape = useLandscapeOrientation();

  const prepareCamera = useCallback(() => {
    const attempt = ++cameraAttemptRef.current;
    setCameraStatus('preparing');
    setCameraError(null);

    void camera.open({ facingMode: 'environment' }).then(
      (session) => {
        if (!mountedRef.current || cameraAttemptRef.current !== attempt) {
          void session.stop();
          return;
        }
        setCameraSession(session);
        setCameraStatus('ready');
      },
      (error: unknown) => {
        if (!mountedRef.current || cameraAttemptRef.current !== attempt) {
          return;
        }
        setCameraStatus('failed');
        setCameraError(error);
      },
    );
  }, [camera]);

  const prepareRuntime = useCallback(() => {
    const attempt = ++runtimeAttemptRef.current;
    committedRef.current = false;
    setRuntimeStatus('preparing');
    setRuntimeError(null);
    setSnapshot(null);

    void runtime.initialize().then(
      () => {
        if (!mountedRef.current || runtimeAttemptRef.current !== attempt) {
          return;
        }
        setRuntimeStatus('ready');
      },
      (error: unknown) => {
        if (!mountedRef.current || runtimeAttemptRef.current !== attempt) {
          return;
        }
        setRuntimeStatus('failed');
        setRuntimeError(error);
      },
    );
  }, [runtime]);

  useEffect(() => {
    mountedRef.current = true;
    prepareCamera();
    prepareRuntime();

    return () => {
      mountedRef.current = false;
      cameraAttemptRef.current += 1;
      runtimeAttemptRef.current += 1;
    };
  }, [prepareCamera, prepareRuntime]);

  useEffect(() => {
    if (cameraSession === null || videoRef.current === null) {
      return undefined;
    }

    const session = cameraSession;
    session.preview.attach(videoRef.current);

    return () => {
      session.preview.detach();
      void session.stop();
    };
  }, [cameraSession]);

  const handleRuntimeFailure = useCallback((error: unknown) => {
    if (!mountedRef.current || committedRef.current) {
      return;
    }
    setRuntimeStatus('failed');
    setRuntimeError(error);
    setSnapshot(null);
  }, []);

  useEffect(() => {
    if (
      cameraSession === null ||
      cameraStatus !== 'ready' ||
      runtimeStatus !== 'ready' ||
      !isLandscape ||
      committedRef.current
    ) {
      return undefined;
    }

    let active = true;
    let run: RecognitionPageRun | null = null;
    let stopRequested = false;
    let runStopped = false;
    const stopRun = () => {
      stopRequested = true;
      if (run !== null && !runStopped) {
        runStopped = true;
        run.stop();
      }
    };

    try {
      recognizer.reset();
      const source = createRecognitionFrameSource(cameraSession);
      const startedRun = recognizer.start(source, {
        onUpdate(update) {
          if (!active || !mountedRef.current || committedRef.current) {
            return;
          }

          if (update.kind === 'confirmed') {
            committedRef.current = true;
            active = false;
            stopRun();
            onConfirmed(update.result);
            return;
          }

          setSnapshot(update.snapshot);
        },
        onError(error) {
          if (!active || !mountedRef.current || committedRef.current) {
            return;
          }
          active = false;
          stopRun();
          handleRuntimeFailure(error);
        },
      });
      run = startedRun;

      if (stopRequested || !active || committedRef.current) {
        stopRun();
      }
    } catch (error) {
      active = false;
      handleRuntimeFailure(error);
    }

    return () => {
      active = false;
      stopRun();
    };
  }, [
    cameraSession,
    cameraStatus,
    handleRuntimeFailure,
    isLandscape,
    onConfirmed,
    recognizer,
    runtimeStatus,
  ]);

  const preparationMessage = getPreparationMessage(
    cameraStatus,
    runtimeStatus,
    isLandscape,
  );

  return (
    <Stack gap="md" py="md">
      <Group justify="space-between" align="center">
        <Title order={1}>認識</Title>
        <Button variant="subtle" onClick={onAbandon}>
          トップへ戻る
        </Button>
      </Group>

      <Box
        data-testid="recognition-capture-surface"
        style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '16 / 9',
          overflow: 'hidden',
          borderRadius: 8,
          background: '#111',
        }}
      >
        {cameraStatus === 'ready' ? (
          <>
            <video
              ref={videoRef}
              aria-label="カメラプレビュー"
              autoPlay
              muted
              playsInline
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                objectFit: 'fill',
              }}
            />
            <CaptureRegionOverlay snapshot={snapshot} />
          </>
        ) : cameraStatus === 'failed' ? (
          <OwnedRecoveryPanel
            owner="camera"
            message={cameraErrorMessage(cameraError)}
            onRetry={prepareCamera}
            onTop={onAbandon}
          />
        ) : (
          <CaptureRegionOverlay snapshot={null} />
        )}

        {cameraStatus !== 'failed' && runtimeStatus !== 'failed' ? (
          <Paper
            role="status"
            px="sm"
            py={6}
            radius="md"
            shadow="sm"
            style={{
              position: 'absolute',
              left: '50%',
              bottom: 12,
              transform: 'translateX(-50%)',
              whiteSpace: 'nowrap',
              background: 'rgba(255,255,255,0.9)',
            }}
          >
            <Text size="sm">{preparationMessage}</Text>
          </Paper>
        ) : null}

        {cameraStatus !== 'failed' && runtimeStatus === 'failed' ? (
          <OwnedRecoveryPanel
            owner="recognition"
            message={runtimeErrorMessage(runtimeError)}
            onRetry={prepareRuntime}
            onTop={onAbandon}
          />
        ) : null}
      </Box>

      {cameraStatus === 'failed' && runtimeStatus === 'failed' ? (
        <OwnedRecoveryPanel
          owner="recognition"
          message={runtimeErrorMessage(runtimeError)}
          onRetry={prepareRuntime}
          onTop={onAbandon}
          inline
        />
      ) : null}

      {!isLandscape ? (
        <Text role="note" size="sm" c="dimmed">
          認識を開始するには端末を横向きにしてください。
        </Text>
      ) : null}
    </Stack>
  );
}

export function RecognitionPage() {
  const services = useContext(RecognitionPageServicesContext);
  const navigate = useNavigate();
  const location = useLocation();
  const beginNewRecognitionAttempt = useApplicationStore(
    (state) => state.beginNewRecognitionAttempt,
  );
  const createScoringSession = useApplicationStore(
    (state) => state.createScoringSession,
  );

  useEffect(() => {
    if (shouldClearSessionForRecognition(location.state)) {
      beginNewRecognitionAttempt();
    }
  }, [beginNewRecognitionAttempt, location.state]);

  const handleConfirmed = useCallback(
    (result: RecognizedStructure) => {
      if (services === null) {
        return;
      }
      createScoringSession(result, DEFAULT_RULE_PROFILE);
      navigateAfterRecognitionConfirmed(navigate);
    },
    [createScoringSession, navigate, services],
  );

  if (services === null) {
    return (
      <Stack gap="md" py="xl">
        <Title order={1}>認識</Title>
        <Text role="status">認識サービスを準備しています。</Text>
        <Button variant="light" onClick={() => navigateToTop(navigate)}>
          トップへ戻る
        </Button>
      </Stack>
    );
  }

  return (
    <RecognitionPageView
      camera={services.camera}
      runtime={services.runtime}
      recognizer={services.recognizer}
      onAbandon={() => navigateToTop(navigate)}
      onConfirmed={handleConfirmed}
    />
  );
}

function CaptureRegionOverlay({
  snapshot,
}: {
  readonly snapshot: RecognitionFrameSnapshot | null;
}) {
  const maskId = useId().replaceAll(':', '');
  const observationsById = useMemo(
    () => new Map(snapshot?.observations.map((item) => [item.id, item]) ?? []),
    [snapshot],
  );

  return (
    <>
      <svg
        aria-hidden="true"
        data-testid="recognition-outside-mask"
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
      >
        <defs>
          <mask id={maskId}>
            <rect x="0" y="0" width="1" height="1" fill="white" />
            {Object.values(RECOGNITION_CAPTURE_REGIONS).map((rect, index) => (
              <rect
                key={index}
                x={rect.x}
                y={rect.y}
                width={rect.width}
                height={rect.height}
                fill="black"
              />
            ))}
          </mask>
        </defs>
        <rect
          x="0"
          y="0"
          width="1"
          height="1"
          fill="rgba(0,0,0,0.58)"
          mask={`url(#${maskId})`}
        />
      </svg>

      <CaptureFrame region="dora-indicators" label="ドラ" />
      <CaptureFrame region="completed-hand" label="手牌" />
      <CaptureFrame region="melds" label="副露" />

      {snapshot?.observations.map((observation) => (
        <ObservationBox key={observation.id} observation={observation} />
      ))}

      {snapshot === null ? null : (
        <MeldGroupOverlay
          groups={snapshot.meldGroups}
          observationsById={observationsById}
        />
      )}
    </>
  );
}

function CaptureFrame({
  region,
  label,
}: {
  readonly region: RecognitionRegion;
  readonly label: string;
}) {
  const rect = RECOGNITION_CAPTURE_REGIONS[region];

  return (
    <Box
      aria-label={`${label}認識領域`}
      data-recognition-region={region}
      style={{
        ...rectStyle(rect),
        border: '2px solid rgba(255,255,255,0.9)',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}
    >
      <Text
        size="xs"
        fw={700}
        style={{
          position: 'absolute',
          left: 4,
          top: 2,
          color: 'white',
          textShadow: '0 1px 2px black',
        }}
      >
        {label}
      </Text>
    </Box>
  );
}

function ObservationBox({
  observation,
}: {
  readonly observation: RecognitionObservation;
}) {
  const label = observation.tile === null
    ? '未解決'
    : tileLabel(observation.tile);

  return (
    <Box
      aria-label={`認識候補 ${label}`}
      data-testid="recognition-observation-box"
      style={{
        ...rectStyle(observation.box),
        border: observation.tile === null
          ? '2px dashed rgba(255,255,255,0.95)'
          : '2px solid rgba(255,255,255,0.95)',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}
    >
      <Box
        style={{
          position: 'absolute',
          right: -1,
          top: -1,
          minWidth: 20,
          padding: '1px 4px',
          borderRadius: 3,
          background: 'rgba(255,255,255,0.94)',
          color: '#111',
          fontSize: 11,
          fontWeight: 700,
          lineHeight: 1.35,
          textAlign: 'center',
        }}
      >
        {observation.tile === null ? '?' : tileLabel(observation.tile)}
      </Box>
    </Box>
  );
}

function MeldGroupOverlay({
  groups,
  observationsById,
}: {
  readonly groups: readonly RecognitionMeldObservationGroup[];
  readonly observationsById: ReadonlyMap<string, RecognitionObservation>;
}) {
  return (
    <>
      <svg
        aria-hidden="true"
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}
      >
        {groups.flatMap((group) => {
          const members = group.memberObservationIds
            .map((id) => observationsById.get(id))
            .filter((item): item is RecognitionObservation => item !== undefined);

          return members.slice(1).map((member, index) => {
            const previous = members[index];
            return (
              <line
                key={`${group.id}-${previous.id}-${member.id}`}
                x1={centerX(previous.box)}
                y1={centerY(previous.box)}
                x2={centerX(member.box)}
                y2={centerY(member.box)}
                stroke="white"
                strokeWidth="0.007"
                strokeDasharray="0.012 0.009"
              />
            );
          });
        })}
      </svg>

      {groups.map((group) => (
        <MeldPreview
          key={group.id}
          group={group}
          observationsById={observationsById}
        />
      ))}
    </>
  );
}

function MeldPreview({
  group,
  observationsById,
}: {
  readonly group: RecognitionMeldObservationGroup;
  readonly observationsById: ReadonlyMap<string, RecognitionObservation>;
}) {
  const members = group.memberObservationIds
    .map((id) => observationsById.get(id))
    .filter((item): item is RecognitionObservation => item !== undefined);

  if (members.length === 0) {
    return null;
  }

  const recognizedLabels = members.map((member) =>
    member.tile === null ? '?' : tileLabel(member.tile),
  );
  const logicalPreview = group.interpretation === 'concealed-kan'
    ? ['裏', ...recognizedLabels, '裏']
    : recognizedLabels;
  const minY = Math.min(...members.map((member) => member.box.y));
  const center = members.reduce((sum, member) => sum + centerX(member.box), 0) /
    members.length;
  const interpretation = meldInterpretationLabel(group.interpretation);

  return (
    <Paper
      aria-label={`${interpretation}プレビュー ${logicalPreview.join(' ')}`}
      data-testid="meld-group-preview"
      px={5}
      py={2}
      radius="sm"
      style={{
        position: 'absolute',
        left: `${center * 100}%`,
        top: `${Math.max(0.01, minY - 0.075) * 100}%`,
        transform: 'translateX(-50%)',
        background: 'rgba(255,255,255,0.92)',
        fontSize: 10,
        lineHeight: 1.25,
        whiteSpace: 'nowrap',
        pointerEvents: 'none',
      }}
    >
      <Text size="xs" fw={700}>
        {logicalPreview.join(' ')}
      </Text>
    </Paper>
  );
}

function OwnedRecoveryPanel({
  owner,
  message,
  onRetry,
  onTop,
  inline = false,
}: {
  readonly owner: 'camera' | 'recognition';
  readonly message: string;
  readonly onRetry: () => void;
  readonly onTop: () => void;
  readonly inline?: boolean;
}) {
  return (
    <Paper
      role="alert"
      data-recovery-owner={owner}
      p="md"
      radius="md"
      shadow="md"
      style={inline ? undefined : {
        position: 'absolute',
        left: '50%',
        top: '50%',
        width: 'min(90%, 420px)',
        transform: 'translate(-50%, -50%)',
        background: 'rgba(255,255,255,0.96)',
        zIndex: 10,
      }}
    >
      <Stack gap="sm">
        <Text fw={700}>{message}</Text>
        <Group gap="xs">
          <Button
            size="xs"
            aria-label={owner === 'camera' ? 'カメラを再試行' : '認識モデルを再試行'}
            onClick={onRetry}
          >
            再試行
          </Button>
          <Button
            size="xs"
            variant="light"
            aria-label={owner === 'camera' ? 'カメラエラーからトップへ' : '認識エラーからトップへ'}
            onClick={onTop}
          >
            トップへ
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}

function createRecognitionFrameSource(
  camera: RecognitionPageCameraSession,
): RecognitionPageFrameSource {
  return {
    captureLatest() {
      const frame = camera.captureLatest();
      if (frame === null) {
        return null;
      }
      return {
        source: frame.image,
        sourceSize: frame.size,
        regions: RECOGNITION_CAPTURE_REGIONS,
        capturedAtMs: frame.capturedAtMs,
      };
    },
  };
}

function useLandscapeOrientation(): boolean {
  const query = '(orientation: landscape)';
  const [isLandscape, setIsLandscape] = useState(() => {
    if (typeof window === 'undefined' || window.matchMedia === undefined) {
      return true;
    }
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (window.matchMedia === undefined) {
      return undefined;
    }
    const media = window.matchMedia(query);
    const update = () => setIsLandscape(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  return isLandscape;
}

function getPreparationMessage(
  cameraStatus: PreparationStatus,
  runtimeStatus: PreparationStatus,
  isLandscape: boolean,
): string {
  if (cameraStatus !== 'ready') {
    return 'カメラを起動しています';
  }
  if (runtimeStatus !== 'ready') {
    return '認識モデルを準備しています';
  }
  if (!isLandscape) {
    return '端末を横向きにしてください';
  }
  return '認識しています';
}

function cameraErrorMessage(error: unknown): string {
  if (!isObjectWithKind(error)) {
    return 'カメラの起動に失敗しました';
  }
  switch (error.kind) {
    case 'permission-denied':
      return 'カメラの使用が許可されていません';
    case 'device-not-found':
      return '利用できるカメラが見つかりません';
    case 'device-unavailable':
      return 'カメラを使用できません';
    case 'unsupported':
      return 'このブラウザではカメラを利用できません';
    default:
      return 'カメラの起動に失敗しました';
  }
}

function runtimeErrorMessage(error: unknown): string {
  if (!isObjectWithKind(error)) {
    return '認識モデルを準備できませんでした';
  }
  switch (error.kind) {
    case 'model-asset-unavailable':
      return '認識モデルを取得できませんでした';
    case 'inference-failure':
      return '認識処理を続行できませんでした';
    case 'model-integrity-failure':
    case 'model-incompatible':
    case 'execution-provider-unavailable':
    case 'model-initialization-failure':
    default:
      return '認識モデルを準備できませんでした';
  }
}

function shouldClearSessionForRecognition(state: unknown): boolean {
  return (
    typeof state === 'object' &&
    state !== null &&
    'clearActiveScoringSession' in state &&
    state.clearActiveScoringSession === true
  );
}

function isObjectWithKind(value: unknown): value is { readonly kind: string } {
  return typeof value === 'object' && value !== null && 'kind' in value;
}

function rectStyle(rect: NormalizedRect) {
  return {
    position: 'absolute' as const,
    left: `${rect.x * 100}%`,
    top: `${rect.y * 100}%`,
    width: `${rect.width * 100}%`,
    height: `${rect.height * 100}%`,
  };
}

function centerX(rect: NormalizedRect): number {
  return rect.x + rect.width / 2;
}

function centerY(rect: NormalizedRect): number {
  return rect.y + rect.height / 2;
}

function tileLabel(tile: TileIdentity): string {
  return `${tile.red ? '赤' : ''}${tile.kind}`;
}

function meldInterpretationLabel(
  interpretation: RecognitionMeldInterpretation | null,
): string {
  switch (interpretation) {
    case 'chi':
      return 'チー';
    case 'pon':
      return 'ポン';
    case 'open-kan':
      return '明槓';
    case 'concealed-kan':
      return '暗槓';
    case 'unresolved':
    case null:
      return '副露';
  }
}
