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

import type { CameraService, CameraSession } from '@/camera';
import type { RecognizedStructure, TileIdentity } from '@/domain';
import type {
  FrameObservationId,
  FrameRecognitionSnapshot,
  MeldGroupObservation,
  NormalizedRect,
  RecognitionFrameSource,
  RecognitionRegion,
  RecognitionRun,
  RecognitionRuntime,
  RealtimeRecognizer,
  TileObservation,
} from '@/recognition';
import { DEFAULT_RULE_PROFILE } from '@/scoring';

import { useApplicationStore } from './application-state';
import {
  navigateAfterRecognitionConfirmed,
  navigateToTop,
} from './navigation';

export interface RecognitionPageServices {
  readonly camera: CameraService;
  readonly runtime: RecognitionRuntime;
  readonly recognizer: RealtimeRecognizer;
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
    y: 0.22,
    width: 0.62,
    height: 0.2593464052,
  },
  'completed-hand': {
    x: 0.04,
    y: 0.5,
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
  readonly camera: CameraService;
  readonly runtime: RecognitionRuntime;
  readonly recognizer: RealtimeRecognizer;
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
  const [cameraSession, setCameraSession] = useState<CameraSession | null>(null);
  const [snapshot, setSnapshot] = useState<FrameRecognitionSnapshot | null>(null);
  const [recognitionProgress, setRecognitionProgress] = useState<
    'scanning' | 'stabilizing' | null
  >(null);
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
    setRecognitionProgress(null);

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
    setRecognitionProgress(null);
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
    let run: RecognitionRun | null = null;
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
          setRecognitionProgress(update.kind);
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
    recognitionProgress,
    snapshot,
  );

  return (
    <Box
      data-testid="recognition-viewport"
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
        data-testid="recognition-capture-surface"
        style={{
          position: 'relative',
          width: 'min(100vw, 177.7778dvh)',
          height: 'min(100dvh, 56.25vw)',
          aspectRatio: '16 / 9',
          overflow: 'hidden',
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
        ) : (
          <CaptureRegionOverlay snapshot={null} />
        )}

        <Button
          aria-label="認識を終了"
          size="compact-sm"
          variant="light"
          onClick={onAbandon}
          style={{
            position: 'absolute',
            top: 'max(10px, env(safe-area-inset-top))',
            right: 'max(10px, env(safe-area-inset-right))',
            zIndex: 20,
            background: 'rgba(255,255,255,0.92)',
          }}
        >
          終了
        </Button>

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
              bottom: 'max(10px, env(safe-area-inset-bottom))',
              transform: 'translateX(-50%)',
              whiteSpace: 'nowrap',
              background: 'rgba(255,255,255,0.9)',
              zIndex: 20,
            }}
          >
            <Text size="sm">{preparationMessage}</Text>
          </Paper>
        ) : null}

        {cameraStatus === 'failed' && runtimeStatus !== 'failed' ? (
          <OwnedRecoveryPanel
            owner="camera"
            message={cameraErrorMessage(cameraError)}
            onRetry={prepareCamera}
            onTop={onAbandon}
          />
        ) : null}

        {cameraStatus !== 'failed' && runtimeStatus === 'failed' ? (
          <OwnedRecoveryPanel
            owner="recognition"
            message={runtimeErrorMessage(runtimeError)}
            onRetry={prepareRuntime}
            onTop={onAbandon}
          />
        ) : null}

        {cameraStatus === 'failed' && runtimeStatus === 'failed' ? (
          <Stack
            gap="sm"
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              width: 'min(90%, 420px)',
              transform: 'translate(-50%, -50%)',
              zIndex: 20,
            }}
          >
            <OwnedRecoveryPanel
              owner="camera"
              message={cameraErrorMessage(cameraError)}
              onRetry={prepareCamera}
              onTop={onAbandon}
              inline
            />
            <OwnedRecoveryPanel
              owner="recognition"
              message={runtimeErrorMessage(runtimeError)}
              onRetry={prepareRuntime}
              onTop={onAbandon}
              inline
            />
          </Stack>
        ) : null}

        {!isLandscape ? (
          <Paper
            role="note"
            px="sm"
            py={6}
            radius="md"
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              whiteSpace: 'nowrap',
              background: 'rgba(255,255,255,0.92)',
              zIndex: 20,
            }}
          >
            <Text size="sm">認識を開始するには端末を横向きにしてください。</Text>
          </Paper>
        ) : null}
      </Box>
    </Box>
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
  readonly snapshot: FrameRecognitionSnapshot | null;
}) {
  const maskId = useId().replaceAll(':', '');
  const observationsById = useMemo(
    () => new Map<FrameObservationId, TileObservation>(
      snapshot?.observations.map((item) => [item.id, item] as const) ?? [],
    ),
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
  readonly observation: TileObservation;
}) {
  const tile = observation.classification.kind === 'tile'
    ? observation.classification.tile
    : null;
  const label = tile === null ? '未解決' : tileLabel(tile);

  return (
    <Box
      aria-label={`認識候補 ${label}`}
      data-testid="recognition-observation-box"
      style={{
        ...rectStyle(observation.bbox),
        border: tile === null
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
        {tile === null ? '?' : tileLabel(tile)}
      </Box>
    </Box>
  );
}

function MeldGroupOverlay({
  groups,
  observationsById,
}: {
  readonly groups: readonly MeldGroupObservation[];
  readonly observationsById: ReadonlyMap<FrameObservationId, TileObservation>;
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
            .filter((item): item is TileObservation => item !== undefined);

          return members.slice(1).map((member, index) => {
            const previous = members[index];
            return (
              <line
                key={`${group.memberObservationIds.join('-')}-${previous.id}-${member.id}`}
                x1={centerX(previous.bbox)}
                y1={centerY(previous.bbox)}
                x2={centerX(member.bbox)}
                y2={centerY(member.bbox)}
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
          key={group.memberObservationIds.join('|')}
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
  readonly group: MeldGroupObservation;
  readonly observationsById: ReadonlyMap<FrameObservationId, TileObservation>;
}) {
  const members = group.memberObservationIds
    .map((id) => observationsById.get(id))
    .filter((item): item is TileObservation => item !== undefined);

  if (members.length === 0) {
    return null;
  }

  const recognizedLabels = members.map((member) =>
    member.classification.kind === 'tile'
      ? tileLabel(member.classification.tile)
      : '?',
  );
  const logicalPreview = group.interpretation.kind === 'concealed-kan'
    ? ['裏', ...recognizedLabels, '裏']
    : recognizedLabels;
  const minY = Math.min(...members.map((member) => member.bbox.y));
  const center = members.reduce((sum, member) => sum + centerX(member.bbox), 0) /
    members.length;
  const interpretation = meldInterpretationLabel(group.interpretation.kind);

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
  camera: CameraSession,
): RecognitionFrameSource {
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
  recognitionProgress: 'scanning' | 'stabilizing' | null,
  snapshot: FrameRecognitionSnapshot | null,
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
  if (recognitionProgress === 'stabilizing') {
    return '認識結果を安定確認しています';
  }
  if (
    recognitionProgress === 'scanning' &&
    snapshot?.commitEligibility.kind === 'ineligible'
  ) {
    if (snapshot.commitEligibility.reason === 'unresolved-meld-geometry') {
      return '副露の配置を調整してください';
    }

    const completedHandTiles = countRecognizedTiles(snapshot, 'completed-hand');
    const meldTiles = countRecognizedTiles(snapshot, 'melds');
    return `認識しています（有効牌 ${completedHandTiles + meldTiles}/10、手牌 ${completedHandTiles}/2）`;
  }
  return '認識しています';
}

function countRecognizedTiles(
  snapshot: FrameRecognitionSnapshot,
  region: RecognitionRegion,
): number {
  return snapshot.observations.filter(
    (observation) =>
      observation.region === region && observation.classification.kind === 'tile',
  ).length;
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
  interpretation: MeldGroupObservation['interpretation']['kind'],
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
      return '副露';
  }
}
