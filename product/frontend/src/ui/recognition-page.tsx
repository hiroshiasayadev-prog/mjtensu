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

import type {
  CameraFrameAspect,
  CameraFrameRotation,
  CameraService,
  CameraSession,
} from '@/camera';
import type { RecognizedStructure, TileIdentity } from '@/domain';
import type {
  FrameObservationId,
  FrameRecognitionSnapshot,
  RecognitionDebugCapture,
  MeldGroupObservation,
  NormalizedRect,
  RecognitionFrameSource,
  RecognitionRegion,
  RecognitionRun,
  RecognitionRuntime,
  RecognitionRuntimeDiagnostics,
  RealtimeRecognizer,
  TileObservation,
} from '@/recognition';
import { DEFAULT_RULE_PROFILE } from '@/scoring';

import { useApplicationStore } from './application-state';
import {
  navigateAfterRecognitionConfirmed,
  navigateToTop,
} from './navigation';
import { TILE_BACK_ASSET_URL, tileAssetUrl } from './tile-assets';

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

const MELD_TILT_WARNING_SHOW_RADIANS = (30 * Math.PI) / 180;
const MELD_TILT_WARNING_CLEAR_RADIANS = (25 * Math.PI) / 180;
const MELD_TILT_WARNING_CONSECUTIVE_FRAMES = 3;

interface MeldTiltWarningState {
  readonly visible: boolean;
  readonly highTiltConsecutive: number;
  readonly lowTiltConsecutive: number;
}

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
  const [meldTiltWarningVisible, setMeldTiltWarningVisible] = useState(false);
  const [debugCaptureStatus, setDebugCaptureStatus] = useState<
    'idle' | 'capturing' | 'ready' | 'failed'
  >('idle');
  const [debugCaptureFile, setDebugCaptureFile] = useState<File | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const mountedRef = useRef(false);
  const cameraAttemptRef = useRef(0);
  const runtimeAttemptRef = useRef(0);
  const committedRef = useRef(false);
  const meldTiltWarningStateRef = useRef<MeldTiltWarningState>(
    initialMeldTiltWarningState(),
  );
  const isPortraitViewport = usePortraitViewport();
  const captureAspectRatio: CameraFrameAspect = isPortraitViewport
    ? '9:16'
    : '16:9';
  const captureRotation: CameraFrameRotation = isPortraitViewport ? -90 : 0;

  const resetMeldTiltWarning = useCallback(() => {
    meldTiltWarningStateRef.current = initialMeldTiltWarningState();
    setMeldTiltWarningVisible(false);
  }, []);

  const observeMeldTilt = useCallback((commonAngleRadians: number | null) => {
    const next = advanceMeldTiltWarning(
      meldTiltWarningStateRef.current,
      commonAngleRadians,
    );
    meldTiltWarningStateRef.current = next;
    setMeldTiltWarningVisible(next.visible);
  }, []);

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
    resetMeldTiltWarning();

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
  }, [resetMeldTiltWarning, runtime]);

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
    resetMeldTiltWarning();
  }, [resetMeldTiltWarning]);

  useEffect(() => {
    if (
      cameraSession === null ||
      cameraStatus !== 'ready' ||
      runtimeStatus !== 'ready' ||
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
      setSnapshot(null);
      setRecognitionProgress(null);
      resetMeldTiltWarning();
      recognizer.reset();
      const source = createRecognitionFrameSource(
        cameraSession,
        captureAspectRatio,
        captureRotation,
      );
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

          observeMeldTilt(update.snapshot.meldCommonAngleRadians);
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
    isPortraitViewport,
    observeMeldTilt,
    onConfirmed,
    recognizer,
    resetMeldTiltWarning,
    runtimeStatus,
  ]);

  const handleDebugCapture = useCallback(() => {
    if (debugCaptureFile !== null) {
      shareOrDownloadRecognitionDebugFile(debugCaptureFile, () => {
        if (!mountedRef.current) {
          return;
        }
        setDebugCaptureFile(null);
        setDebugCaptureStatus('idle');
      });
      return;
    }

    if (runtime.requestDebugCapture === undefined) {
      return;
    }

    setDebugCaptureStatus('capturing');
    void runtime.requestDebugCapture().then(
      (capture) => {
        if (!mountedRef.current) {
          return;
        }
        try {
          setDebugCaptureFile(
            createRecognitionDebugFile(capture, runtime.getDiagnostics?.() ?? null),
          );
          setDebugCaptureStatus('ready');
        } catch {
          setDebugCaptureFile(null);
          setDebugCaptureStatus('failed');
        }
      },
      () => {
        if (!mountedRef.current) {
          return;
        }
        setDebugCaptureFile(null);
        setDebugCaptureStatus('failed');
      },
    );
  }, [debugCaptureFile, runtime]);

  const preparationMessage = getPreparationMessage(
    cameraStatus,
    runtimeStatus,
    recognitionProgress,
    snapshot,
    meldTiltWarningVisible,
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
          ...captureSurfaceLayout(isPortraitViewport),
          aspectRatio: isPortraitViewport ? '9 / 16' : '16 / 9',
          overflow: 'hidden',
          background: '#111',
        }}
      >
        {cameraStatus === 'ready' ? (
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
              objectFit: 'cover',
              pointerEvents: 'none',
            }}
          />
        ) : null}

        <Box
          data-testid="recognition-landscape-ui-surface"
          style={{
            ...landscapeUiSurfaceLayout(isPortraitViewport),
            aspectRatio: '16 / 9',
            pointerEvents: 'auto',
          }}
        >
          {runtime.requestDebugCapture === undefined ? null : (
            <Button
              aria-label={
                debugCaptureFile === null ? '認識デバッグを採取' : '認識デバッグを保存'
              }
              data-testid="recognition-debug-capture"
              size="compact-sm"
              variant="light"
              disabled={debugCaptureStatus === 'capturing'}
              onClick={handleDebugCapture}
              style={{
                position: 'absolute',
                top: 'max(10px, env(safe-area-inset-top))',
                left: 'max(10px, env(safe-area-inset-left))',
                zIndex: 200,
                pointerEvents: 'auto',
                touchAction: 'manipulation',
                background: 'rgba(255,255,255,0.92)',
              }}
            >
              {debugCaptureButtonLabel(debugCaptureStatus)}
            </Button>
          )}

          <Button
            aria-label="認識を終了"
            data-testid="recognition-global-exit"
            size="compact-sm"
            variant="light"
            onClick={onAbandon}
            style={{
              position: 'absolute',
              top: 'max(10px, env(safe-area-inset-top))',
              right: 'max(10px, env(safe-area-inset-right))',
              zIndex: 200,
              pointerEvents: 'auto',
              touchAction: 'manipulation',
              background: 'rgba(255,255,255,0.92)',
            }}
          >
            終了
          </Button>

          <CaptureRegionOverlay
            snapshot={snapshot}
            regions={RECOGNITION_CAPTURE_REGIONS}
          />

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

          {cameraStatus === 'failed' || runtimeStatus === 'failed' ? (
          <Box
            data-testid="recognition-recovery-layer"
            style={{
              position: 'absolute',
              inset: 0,
              zIndex: 100,
              display: 'grid',
              placeItems: 'center',
              padding: 12,
              pointerEvents: 'auto',
              touchAction: 'manipulation',
            }}
          >
            <Stack gap="sm" style={{ width: 'min(90%, 420px)' }}>
              {cameraStatus === 'failed' ? (
                <OwnedRecoveryPanel
                  owner="camera"
                  message={cameraErrorMessage(cameraError)}
                  onRetry={prepareCamera}
                  onTop={onAbandon}
                  inline
                />
              ) : null}
              {runtimeStatus === 'failed' ? (
                <OwnedRecoveryPanel
                  owner="recognition"
                  message={runtimeErrorMessage(runtimeError)}
                  detail={runtimeErrorDiagnostic(runtimeError)}
                  onRetry={prepareRuntime}
                  onTop={onAbandon}
                  inline
                />
              ) : null}
            </Stack>
          </Box>
          ) : null}
        </Box>
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
  regions,
}: {
  readonly snapshot: FrameRecognitionSnapshot | null;
  readonly regions: Readonly<Record<RecognitionRegion, NormalizedRect>>;
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
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
      >
        <defs>
          <mask id={maskId}>
            <rect x="0" y="0" width="1" height="1" fill="white" />
            {Object.values(regions).map((rect, index) => (
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

      <CaptureFrame region="dora-indicators" label="ドラ" regions={regions} />
      <CaptureFrame region="completed-hand" label="手牌" regions={regions} />
      <CaptureFrame region="melds" label="副露" regions={regions} />

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
  regions,
}: {
  readonly region: RecognitionRegion;
  readonly label: string;
  readonly regions: Readonly<Record<RecognitionRegion, NormalizedRect>>;
}) {
  const rect = regions[region];

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
  const feedbackState = tile === null ? 'unresolved' : 'recognized';
  const label = tile === null ? '未解決' : tileAccessibleLabel(tile);

  return (
    <Box
      aria-label={`認識候補 ${label}`}
      data-testid="recognition-observation-box"
      data-recognition-state={feedbackState}
      style={{
        ...rectStyle(observation.bbox),
        border: tile === null
          ? '2px dashed #ffd43b'
          : '2px solid #51cf66',
        boxShadow: tile === null
          ? '0 0 0 1px rgba(0,0,0,0.55), 0 0 7px rgba(255,212,59,0.9)'
          : '0 0 0 1px rgba(0,0,0,0.55), 0 0 7px rgba(81,207,102,0.85)',
        boxSizing: 'border-box',
        pointerEvents: 'none',
      }}
    >
      <Box
        data-testid="recognition-observation-identity"
        style={{
          position: 'absolute',
          left: '50%',
          bottom: 'calc(100% + 2px)',
          transform: 'translateX(-50%)',
        }}
      >
        {tile === null ? <UnresolvedTileFace /> : <TileImage tile={tile} />}
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
        style={{
          position: 'absolute',
          inset: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
        }}
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
                data-testid="meld-group-connector"
                data-overlay-kind="meld-connector"
                x1={centerX(previous.bbox)}
                y1={centerY(previous.bbox)}
                x2={centerX(member.bbox)}
                y2={centerY(member.bbox)}
                stroke="#22b8cf"
                strokeWidth="0.009"
                strokeDasharray="0.014 0.008"
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

  const recognizedTiles = members.map((member) =>
    member.classification.kind === 'tile'
      ? member.classification.tile
      : null,
  );
  const logicalPreview: readonly MeldPreviewFace[] =
    group.interpretation.kind === 'concealed-kan'
      ? ['back', ...recognizedTiles, 'back']
      : recognizedTiles;
  const minY = Math.min(...members.map((member) => member.bbox.y));
  const center = members.reduce((sum, member) => sum + centerX(member.bbox), 0) /
    members.length;
  const interpretation = meldInterpretationLabel(group.interpretation.kind);

  return (
    <Paper
      aria-label={`${interpretation}プレビュー ${logicalPreview.map(meldPreviewFaceLabel).join(' ')}`}
      data-testid="meld-group-preview"
      data-overlay-kind="meld-preview"
      px={4}
      py={3}
      radius="sm"
      style={{
        position: 'absolute',
        left: `${center * 100}%`,
        top: `${minY * 100}%`,
        transform: 'translate(-50%, calc(-100% - 34px))',
        background: 'rgba(8,68,83,0.92)',
        border: '1px solid #22b8cf',
        boxShadow: '0 1px 5px rgba(0,0,0,0.45)',
        whiteSpace: 'nowrap',
        pointerEvents: 'none',
      }}
    >
      <Group gap={2} wrap="nowrap">
        {logicalPreview.map((face, index) => (
          <MeldPreviewTileFace key={index} face={face} />
        ))}
      </Group>
    </Paper>
  );
}

type MeldPreviewFace = TileIdentity | null | 'back';

function TileImage({
  tile,
  compact = false,
}: {
  readonly tile: TileIdentity;
  readonly compact?: boolean;
}) {
  const width = compact ? 18 : 22;
  const height = compact ? 25 : 30;

  return (
    <img
      aria-hidden="true"
      data-testid="recognition-tile-face"
      data-red-five={tile.red ? 'true' : 'false'}
      src={tileAssetUrl(tile)}
      width={width}
      height={height}
      alt=""
      draggable={false}
      style={{
        display: 'block',
        width,
        height,
        objectFit: 'contain',
        filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.55))',
      }}
    />
  );
}

function UnresolvedTileFace() {
  return (
    <Box
      aria-hidden="true"
      data-testid="recognition-unresolved-face"
      style={{
        width: 22,
        height: 30,
        display: 'grid',
        placeItems: 'center',
        border: '1px solid #5f4b00',
        borderRadius: 3,
        background: '#fff3bf',
        color: '#5f4b00',
        boxShadow: '0 1px 3px rgba(0,0,0,0.5)',
        fontSize: 16,
        fontWeight: 900,
        lineHeight: 1,
      }}
    >
      ?
    </Box>
  );
}

function MeldPreviewTileFace({ face }: { readonly face: MeldPreviewFace }) {
  if (face === 'back') {
    return (
      <img
        aria-hidden="true"
        data-testid="meld-preview-tile-back"
        src={TILE_BACK_ASSET_URL}
        width={18}
        height={25}
        alt=""
        draggable={false}
        style={{
          display: 'block',
          width: 18,
          height: 25,
          objectFit: 'contain',
        }}
      />
    );
  }

  if (face === null) {
    return <UnresolvedTileFace />;
  }

  return <TileImage tile={face} compact />;
}

function tileAccessibleLabel(tile: TileIdentity): string {
  const rank = Number(tile.kind[0]);
  const rankLabel = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'][rank] ?? '？';
  const redPrefix = tile.red ? '赤' : '';

  if (tile.kind.endsWith('m')) {
    return `${redPrefix}${rankLabel}萬`;
  }
  if (tile.kind.endsWith('p')) {
    return `${redPrefix}${rankLabel}筒`;
  }
  if (tile.kind.endsWith('s')) {
    return `${redPrefix}${rankLabel}索`;
  }

  return ['?', '東', '南', '西', '北', '白', '發', '中'][rank] ?? '?';
}

function meldPreviewFaceLabel(face: MeldPreviewFace): string {
  if (face === 'back') {
    return '裏';
  }
  if (face === null) {
    return '未解決';
  }
  return tileAccessibleLabel(face);
}

function OwnedRecoveryPanel({
  owner,
  message,
  detail,
  onRetry,
  onTop,
  inline = false,
}: {
  readonly owner: 'camera' | 'recognition';
  readonly message: string;
  readonly detail?: string | null;
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
      style={inline ? { pointerEvents: 'auto' } : {
        position: 'absolute',
        left: '50%',
        top: '50%',
        width: 'min(90%, 420px)',
        transform: 'translate(-50%, -50%)',
        background: 'rgba(255,255,255,0.96)',
        zIndex: 30,
        pointerEvents: 'auto',
      }}
    >
      <Stack gap="sm">
        <Text fw={700}>{message}</Text>
        {detail === null || detail === undefined ? null : (
          <Text size="xs" c="dimmed" data-testid="recognition-error-diagnostic">
            診断: {detail}
          </Text>
        )}
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
  aspectRatio: CameraFrameAspect,
  rotation: CameraFrameRotation,
): RecognitionFrameSource {
  return {
    captureLatest() {
      const frame = camera.captureLatest({ aspectRatio, rotation });
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

const PORTRAIT_CAPTURE_SURFACE_LAYOUT = {
  width: 'min(100vw, 56.25dvh)',
  height: 'min(100dvh, 177.7778vw)',
} as const;

const LANDSCAPE_CAPTURE_SURFACE_LAYOUT = {
  width: 'min(100vw, 177.7778dvh)',
  height: 'min(100dvh, 56.25vw)',
} as const;

const PORTRAIT_LANDSCAPE_UI_SURFACE_LAYOUT = {
  position: 'absolute',
  left: '50%',
  top: '50%',
  width: '177.7778%',
  height: '56.25%',
  transform: 'translate(-50%, -50%) rotate(90deg)',
  transformOrigin: 'center',
} as const;

const LANDSCAPE_UI_SURFACE_LAYOUT = {
  position: 'absolute',
  inset: 0,
  width: '100%',
  height: '100%',
} as const;

function captureSurfaceLayout(isPortraitViewport: boolean): {
  readonly width: string;
  readonly height: string;
} {
  return isPortraitViewport
    ? PORTRAIT_CAPTURE_SURFACE_LAYOUT
    : LANDSCAPE_CAPTURE_SURFACE_LAYOUT;
}

function landscapeUiSurfaceLayout(isPortraitViewport: boolean) {
  return isPortraitViewport
    ? PORTRAIT_LANDSCAPE_UI_SURFACE_LAYOUT
    : LANDSCAPE_UI_SURFACE_LAYOUT;
}

function usePortraitViewport(): boolean {
  const [isPortrait, setIsPortrait] = useState(() => viewportIsPortrait());

  useEffect(() => {
    const update = () => setIsPortrait(viewportIsPortrait());
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return isPortrait;
}

function viewportIsPortrait(): boolean {
  if (typeof window === 'undefined') {
    return false;
  }
  return window.innerHeight > window.innerWidth;
}

function debugCaptureButtonLabel(
  status: 'idle' | 'capturing' | 'ready' | 'failed',
): string {
  switch (status) {
    case 'capturing':
      return '採取中…';
    case 'ready':
      return 'デバッグ保存';
    case 'failed':
      return '再採取';
    case 'idle':
      return 'デバッグ採取';
  }
}

function createRecognitionDebugFile(
  capture: RecognitionDebugCapture,
  runtimeDiagnostics: RecognitionRuntimeDiagnostics | null,
): File {
  const payload = {
    schemaVersion: 1,
    capture,
    runtimeDiagnostics,
    environment: {
      userAgent: navigator.userAgent,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight,
        devicePixelRatio: window.devicePixelRatio,
      },
      screen: {
        width: window.screen.width,
        height: window.screen.height,
        orientationType: window.screen.orientation?.type ?? null,
        orientationAngle: window.screen.orientation?.angle ?? null,
      },
    },
  } as const;
  const timestamp = capture.createdAtIso.replaceAll(':', '-').replaceAll('.', '-');
  return new File(
    [JSON.stringify(payload)],
    `mjtensu-recognition-debug-${timestamp}.json`,
    { type: 'application/json' },
  );
}

function shareOrDownloadRecognitionDebugFile(
  file: File,
  onCompleted: () => void,
): void {
  const shareNavigator = navigator as Navigator & {
    readonly canShare?: (data: { files?: File[] }) => boolean;
    readonly share?: (data: { files?: File[]; title?: string }) => Promise<void>;
  };

  let canShareFile = false;
  try {
    canShareFile =
      shareNavigator.share !== undefined &&
      shareNavigator.canShare?.({ files: [file] }) === true;
  } catch {
    canShareFile = false;
  }

  if (canShareFile && shareNavigator.share !== undefined) {
    void shareNavigator.share({
      files: [file],
      title: 'mjtensu recognition debug',
    }).then(
      onCompleted,
      (error: unknown) => {
        if (isAbortError(error)) {
          return;
        }
        downloadRecognitionDebugFile(file);
        onCompleted();
      },
    );
    return;
  }

  downloadRecognitionDebugFile(file);
  onCompleted();
}

function downloadRecognitionDebugFile(file: File): void {
  const url = URL.createObjectURL(file);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = file.name;
  anchor.style.display = 'none';
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function getPreparationMessage(
  cameraStatus: PreparationStatus,
  runtimeStatus: PreparationStatus,
  recognitionProgress: 'scanning' | 'stabilizing' | null,
  snapshot: FrameRecognitionSnapshot | null,
  meldTiltWarningVisible: boolean,
): string {
  if (cameraStatus !== 'ready') {
    return 'カメラを起動しています';
  }
  if (runtimeStatus !== 'ready') {
    return '認識モデルを準備しています';
  }
  if (meldTiltWarningVisible) {
    return '牌の並びを水平にすると認識が安定します';
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

function initialMeldTiltWarningState(): MeldTiltWarningState {
  return {
    visible: false,
    highTiltConsecutive: 0,
    lowTiltConsecutive: 0,
  };
}

function advanceMeldTiltWarning(
  state: MeldTiltWarningState,
  commonAngleRadians: number | null,
): MeldTiltWarningState {
  if (commonAngleRadians === null || !Number.isFinite(commonAngleRadians)) {
    return state.visible
      ? { ...state, lowTiltConsecutive: 0 }
      : { ...state, highTiltConsecutive: 0 };
  }

  const absoluteAngle = Math.abs(commonAngleRadians);
  if (!state.visible) {
    if (absoluteAngle <= MELD_TILT_WARNING_SHOW_RADIANS) {
      return { ...state, highTiltConsecutive: 0 };
    }

    const highTiltConsecutive = state.highTiltConsecutive + 1;
    if (highTiltConsecutive < MELD_TILT_WARNING_CONSECUTIVE_FRAMES) {
      return { ...state, highTiltConsecutive };
    }
    return {
      visible: true,
      highTiltConsecutive: 0,
      lowTiltConsecutive: 0,
    };
  }

  if (absoluteAngle >= MELD_TILT_WARNING_CLEAR_RADIANS) {
    return { ...state, lowTiltConsecutive: 0 };
  }

  const lowTiltConsecutive = state.lowTiltConsecutive + 1;
  if (lowTiltConsecutive < MELD_TILT_WARNING_CONSECUTIVE_FRAMES) {
    return { ...state, lowTiltConsecutive };
  }
  return initialMeldTiltWarningState();
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

function runtimeErrorDiagnostic(error: unknown): string | null {
  if (
    typeof error !== 'object' ||
    error === null ||
    !('kind' in error) ||
    !('model' in error) ||
    typeof error.kind !== 'string' ||
    typeof error.model !== 'string'
  ) {
    return null;
  }

  const base = `${error.model} / ${error.kind}`;
  if (!('cause' in error)) {
    return base;
  }

  const cause = error.cause;
  if (cause instanceof Error) {
    return `${base} / ${cause.name}: ${cause.message}`;
  }
  if (
    typeof cause === 'object' &&
    cause !== null &&
    'name' in cause &&
    typeof cause.name === 'string'
  ) {
    const message = 'message' in cause && typeof cause.message === 'string'
      ? `: ${cause.message}`
      : '';
    return `${base} / ${cause.name}${message}`;
  }
  return base;
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
