import { MantineProvider } from '@mantine/core';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import {
  MemoryRouter,
  Route,
  Routes,
  useNavigate,
} from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  createApplicationStore,
  createScoringSessionService,
  createScoringSessionFixture,
  type ApplicationStore,
} from '@/application';
import type {
  RecognizedStructure,
  TileInstance,
  TileInstanceId,
} from '@/domain';
import type { RecognitionRuntimeError } from '@/recognition';
import type { ScoringService } from '@/scoring';
import {
  ApplicationStateProvider,
  RECOGNITION_CAPTURE_REGIONS,
  RecognitionPage,
  RecognitionPageServicesProvider,
  RecognitionPageView,
  type RecognitionFrameSnapshot,
  type RecognitionPageCameraService,
  type RecognitionPageCameraSession,
  type RecognitionPageFrameSource,
  type RecognitionPageRealtimeListener,
  type RecognitionPageRealtimeRecognizer,
  type RecognitionPageRun,
  type RecognitionPageRuntime,
  type RecognitionPageServices,
} from '@/ui';

import { createDeferred, createFakeService } from './support';

beforeEach(() => {
  vi.spyOn(window, 'matchMedia').mockImplementation((query) =>
    mediaQueryList(query, query === '(orientation: landscape)'),
  );
});

afterEach(() => {
  vi.restoreAllMocks();
});

function mediaQueryList(query: string, matches: boolean): MediaQueryList {
  return {
    matches,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  };
}

function createCameraSession() {
  const attach = vi.fn();
  const detach = vi.fn();
  const stop = vi.fn(async () => undefined);
  const frame = {
    image: document.createElement('canvas'),
    size: { width: 1280, height: 720 },
    capturedAtMs: 123,
  };
  const captureLatest = vi.fn(() => frame);
  const session: RecognitionPageCameraSession = {
    preview: { attach, detach },
    captureLatest,
    stop,
  };

  return { session, attach, detach, stop, captureLatest, frame };
}

function createRecognizerHarness() {
  let listener: RecognitionPageRealtimeListener | null = null;
  let source: RecognitionPageFrameSource | null = null;
  const runs: Array<{ stop: ReturnType<typeof vi.fn> }> = [];
  const reset = vi.fn();
  const start = vi.fn(
    (
      nextSource: RecognitionPageFrameSource,
      nextListener: RecognitionPageRealtimeListener,
    ): RecognitionPageRun => {
      source = nextSource;
      listener = nextListener;
      const stop = vi.fn();
      runs.push({ stop });
      return { stop };
    },
  );
  const recognizer: RecognitionPageRealtimeRecognizer = { start, reset };

  return {
    recognizer,
    reset,
    start,
    runs,
    getListener: () => listener,
    getSource: () => source,
  };
}

function renderView({
  camera,
  runtime,
  recognizer,
}: {
  readonly camera: RecognitionPageCameraService;
  readonly runtime: RecognitionPageRuntime;
  readonly recognizer: RecognitionPageRealtimeRecognizer;
}) {
  const onAbandon = vi.fn();
  const onConfirmed = vi.fn();

  render(
    <MantineProvider>
      <RecognitionPageView
        camera={camera}
        runtime={runtime}
        recognizer={recognizer}
        onAbandon={onAbandon}
        onConfirmed={onConfirmed}
      />
    </MantineProvider>,
  );

  return { onAbandon, onConfirmed };
}

function createScoringSessionPort() {
  const scoring = createFakeService<ScoringService>({
    validateWinningStructure: () => ({ kind: 'valid' }),
    preview: () => ({ kind: 'no-yaku' }),
    calculate: () => {
      throw new Error('not used by Recognition page');
    },
  });
  return createScoringSessionService(scoring);
}

describe('RecognitionPageView preparation and capture surface', () => {
  it('shows the fixed semantic frames as the visible recognition boundary', () => {
    const cameraDeferred = createDeferred<RecognitionPageCameraSession>();
    const runtimeDeferred = createDeferred<void>();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: () => cameraDeferred.promise },
      runtime: { initialize: () => runtimeDeferred.promise },
      recognizer: recognizer.recognizer,
    });

    expect(screen.getByLabelText('ドラ認識領域')).toBeVisible();
    expect(screen.getByLabelText('手牌認識領域')).toBeVisible();
    expect(screen.getByLabelText('副露認識領域')).toBeVisible();
    expect(screen.getByTestId('recognition-outside-mask')).toBeInTheDocument();
    expect(
      (RECOGNITION_CAPTURE_REGIONS['completed-hand'].width /
        RECOGNITION_CAPTURE_REGIONS['completed-hand'].height) *
        (16 / 9),
    ).toBeCloseTo(17 / 4, 8);
    expect(
      (RECOGNITION_CAPTURE_REGIONS.melds.width /
        RECOGNITION_CAPTURE_REGIONS.melds.height) *
        (16 / 9),
    ).toBeCloseTo(1, 8);
    expect(screen.queryByText(/14/)).not.toBeInTheDocument();
  });

  it('opens camera and runtime in parallel, exposes preview first, then starts realtime only when both are ready', async () => {
    const cameraDeferred = createDeferred<RecognitionPageCameraSession>();
    const runtimeDeferred = createDeferred<void>();
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi.fn(() => cameraDeferred.promise);
    const initialize = vi.fn(() => runtimeDeferred.promise);

    renderView({
      camera: { open },
      runtime: { initialize },
      recognizer: recognizer.recognizer,
    });

    expect(open).toHaveBeenCalledTimes(1);
    expect(initialize).toHaveBeenCalledTimes(1);
    expect(recognizer.start).not.toHaveBeenCalled();

    await act(async () => {
      cameraDeferred.resolve(cameraSession.session);
      await cameraDeferred.promise;
    });

    await waitFor(() => expect(cameraSession.attach).toHaveBeenCalledTimes(1));
    expect(screen.getByRole('status')).toHaveTextContent(
      '認識モデルを準備しています',
    );
    expect(recognizer.start).not.toHaveBeenCalled();

    await act(async () => {
      runtimeDeferred.resolve();
      await runtimeDeferred.promise;
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));
    expect(screen.getByRole('status')).toHaveTextContent('認識しています');

    const source = recognizer.getSource();
    expect(source).not.toBeNull();
    const recognitionFrame = source?.captureLatest();
    expect(recognitionFrame).toEqual({
      source: cameraSession.frame.image,
      sourceSize: cameraSession.frame.size,
      regions: RECOGNITION_CAPTURE_REGIONS,
      capturedAtMs: cameraSession.frame.capturedAtMs,
    });
  });

  it('keeps realtime stopped when runtime is ready before the camera', async () => {
    const cameraDeferred = createDeferred<RecognitionPageCameraSession>();
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: () => cameraDeferred.promise },
      runtime: { initialize: async () => undefined },
      recognizer: recognizer.recognizer,
    });

    await waitFor(() =>
      expect(screen.getByRole('status')).toHaveTextContent(
        'カメラを起動しています',
      ),
    );
    expect(recognizer.start).not.toHaveBeenCalled();

    await act(async () => {
      cameraDeferred.resolve(cameraSession.session);
      await cameraDeferred.promise;
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));
  });

  it('does not start realtime in portrait even when camera and runtime are ready', async () => {
    vi.mocked(window.matchMedia).mockImplementation((query) =>
      mediaQueryList(query, false),
    );
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: async () => cameraSession.session },
      runtime: { initialize: async () => undefined },
      recognizer: recognizer.recognizer,
    });

    await waitFor(() => expect(cameraSession.attach).toHaveBeenCalledTimes(1));
    expect(screen.getByText('認識を開始するには端末を横向きにしてください。')).toBeVisible();
    expect(recognizer.start).not.toHaveBeenCalled();
  });
});

describe('RecognitionPageView live feedback', () => {
  it('renders semantic observations and concealed-kan preview without inventing hidden boxes', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: async () => cameraSession.session },
      runtime: { initialize: async () => undefined },
      recognizer: recognizer.recognizer,
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));

    const snapshot: RecognitionFrameSnapshot = {
      observations: [
        {
          id: 'hand-1',
          region: 'completed-hand',
          box: { x: 0.08, y: 0.72, width: 0.05, height: 0.13 },
          tile: { kind: '1m', red: false },
        },
        {
          id: 'meld-1',
          region: 'melds',
          box: { x: 0.75, y: 0.43, width: 0.05, height: 0.12 },
          tile: { kind: '5s', red: false },
        },
        {
          id: 'meld-2',
          region: 'melds',
          box: { x: 0.84, y: 0.43, width: 0.05, height: 0.12 },
          tile: { kind: '5s', red: false },
        },
        {
          id: 'dora-unresolved',
          region: 'dora-indicators',
          box: { x: 0.1, y: 0.12, width: 0.05, height: 0.12 },
          tile: null,
        },
      ],
      meldGroups: [
        {
          id: 'kan-1',
          memberObservationIds: ['meld-1', 'meld-2'],
          interpretation: 'concealed-kan',
        },
      ],
    };

    act(() => {
      recognizer.getListener()?.onUpdate({ kind: 'scanning', snapshot });
    });

    expect(screen.getAllByTestId('recognition-observation-box')).toHaveLength(4);
    expect(screen.getByLabelText('認識候補 1m')).toBeVisible();
    expect(screen.getByLabelText('認識候補 未解決')).toBeVisible();
    expect(
      screen.getByLabelText('暗槓プレビュー 裏 5s 5s 裏'),
    ).toBeVisible();
  });
});

describe('RecognitionPageView owner-specific recovery', () => {
  it('retries camera without reinitializing an already-ready runtime', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi
      .fn<RecognitionPageCameraService['open']>()
      .mockRejectedValueOnce({ kind: 'device-unavailable' })
      .mockResolvedValueOnce(cameraSession.session);
    const initialize = vi.fn(async () => undefined);

    renderView({
      camera: { open },
      runtime: { initialize },
      recognizer: recognizer.recognizer,
    });

    expect(
      await screen.findByText('カメラを使用できません'),
    ).toBeVisible();
    expect(initialize).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'カメラを再試行' }));

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));
    expect(open).toHaveBeenCalledTimes(2);
    expect(initialize).toHaveBeenCalledTimes(1);
  });

  it('retries runtime without reopening a healthy camera', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi.fn(async () => cameraSession.session);
    const initialize = vi
      .fn<RecognitionPageRuntime['initialize']>()
      .mockRejectedValueOnce({
        kind: 'model-initialization-failure',
        model: 'detector',
        cause: new Error('fixture'),
      })
      .mockResolvedValueOnce(undefined);

    renderView({
      camera: { open },
      runtime: { initialize },
      recognizer: recognizer.recognizer,
    });

    expect(
      await screen.findByText('認識モデルを準備できませんでした'),
    ).toBeVisible();
    await waitFor(() => expect(cameraSession.attach).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByRole('button', { name: '認識モデルを再試行' }));

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));
    expect(open).toHaveBeenCalledTimes(1);
    expect(initialize).toHaveBeenCalledTimes(2);
    expect(cameraSession.detach).not.toHaveBeenCalled();
  });

  it('keeps both failure owners independently retryable', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi
      .fn<RecognitionPageCameraService['open']>()
      .mockRejectedValueOnce({ kind: 'permission-denied' })
      .mockResolvedValueOnce(cameraSession.session);
    const initialize = vi
      .fn<RecognitionPageRuntime['initialize']>()
      .mockRejectedValueOnce({
        kind: 'model-asset-unavailable',
        model: 'tile-classifier',
      })
      .mockResolvedValueOnce(undefined);

    renderView({
      camera: { open },
      runtime: { initialize },
      recognizer: recognizer.recognizer,
    });

    expect(await screen.findByText('カメラの使用が許可されていません')).toBeVisible();
    expect(screen.getByText('認識モデルを取得できませんでした')).toBeVisible();

    fireEvent.click(screen.getByRole('button', { name: '認識モデルを再試行' }));
    await waitFor(() => expect(initialize).toHaveBeenCalledTimes(2));
    expect(open).toHaveBeenCalledTimes(1);

    fireEvent.click(screen.getByRole('button', { name: 'カメラを再試行' }));
    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));
    expect(open).toHaveBeenCalledTimes(2);
    expect(initialize).toHaveBeenCalledTimes(2);
  });

  it('stops a fatal realtime run and retries runtime while preserving the camera session', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi.fn(async () => cameraSession.session);
    const initialize = vi.fn(async () => undefined);

    renderView({
      camera: { open },
      runtime: { initialize },
      recognizer: recognizer.recognizer,
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));

    const error: RecognitionRuntimeError = {
      kind: 'inference-failure',
      model: 'detector',
      cause: new Error('fixture'),
    };
    act(() => {
      recognizer.getListener()?.onError(error);
    });

    expect(
      await screen.findByText('認識処理を続行できませんでした'),
    ).toBeVisible();
    expect(recognizer.runs[0]?.stop).toHaveBeenCalled();
    expect(cameraSession.detach).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: '認識モデルを再試行' }));

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(2));
    expect(open).toHaveBeenCalledTimes(1);
    expect(initialize).toHaveBeenCalledTimes(2);
    expect(cameraSession.attach).toHaveBeenCalledTimes(1);
  });
});

describe('RecognitionPage confirmed transition', () => {
  it('replaces the Application session, replaces Recognition history, and tears down route-owned work', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const services: RecognitionPageServices = {
      camera: { open: async () => cameraSession.session },
      runtime: { initialize: async () => undefined },
      recognizer: recognizer.recognizer,
      scoringSession: createScoringSessionPort(),
    };
    const store = createApplicationStore({
      activeScoringSession: createScoringSessionFixture({
        tileIdPrefix: 'previous',
      }),
    });

    renderRecognitionRoute(services, store);

    await waitFor(() => expect(store.getState().activeScoringSession).toBeNull());
    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));

    const result = recognizedStructure();
    act(() => {
      recognizer.getListener()?.onUpdate({ kind: 'confirmed', result });
    });

    expect(await screen.findByRole('heading', { name: 'Conditions target' })).toBeVisible();
    expect(store.getState().activeScoringSession).toEqual(
      expect.objectContaining({
        structure: result,
        winningTileId: tileId('hand-right'),
        latestResult: null,
      }),
    );
    await waitFor(() => expect(cameraSession.stop).toHaveBeenCalled());
    expect(recognizer.runs[0]?.stop).toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Back' }));

    expect(await screen.findByRole('heading', { name: 'Top target' })).toBeVisible();
    expect(screen.queryByRole('heading', { name: '認識' })).not.toBeInTheDocument();
  });
});

function renderRecognitionRoute(
  services: RecognitionPageServices,
  store: ApplicationStore,
) {
  render(
    <MantineProvider>
      <ApplicationStateProvider store={store}>
        <RecognitionPageServicesProvider services={services}>
          <MemoryRouter
            initialEntries={[
              '/',
              {
                pathname: '/recognition',
                state: { clearActiveScoringSession: true },
              },
            ]}
            initialIndex={1}
          >
            <Routes>
              <Route path="/" element={<h1>Top target</h1>} />
              <Route path="/recognition" element={<RecognitionPage />} />
              <Route path="/conditions" element={<ConditionsTarget />} />
            </Routes>
          </MemoryRouter>
        </RecognitionPageServicesProvider>
      </ApplicationStateProvider>
    </MantineProvider>,
  );
}

function ConditionsTarget() {
  const navigate = useNavigate();
  return (
    <>
      <h1>Conditions target</h1>
      <button type="button" onClick={() => navigate(-1)}>
        Back
      </button>
    </>
  );
}

function recognizedStructure(): RecognizedStructure {
  return {
    completedHand: [tile('hand-left', '1m'), tile('hand-right', '2m')],
    doraIndicators: [],
    meldGroups: [],
  };
}

function tile(id: string, kind: TileInstance['tile']['kind']): TileInstance {
  return {
    id: tileId(id),
    tile: { kind, red: false },
  };
}

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}
