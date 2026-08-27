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
import type { CameraService, CameraSession } from '@/camera';
import type {
  RecognizedStructure,
  TileInstance,
  TileInstanceId,
} from '@/domain';
import type {
  FrameObservationId,
  FrameRecognitionSnapshot,
  RecognitionFrameSource,
  RecognitionRuntime,
  RecognitionRuntimeError,
  RealtimeRecognitionListener,
  RealtimeRecognizer,
} from '@/recognition';
import type { ScoringService } from '@/scoring';
import {
  ApplicationStateProvider,
  RECOGNITION_CAPTURE_REGIONS,
  RecognitionPage,
  RecognitionPageServicesProvider,
  RecognitionPageView,
  type RecognitionPageServices,
} from '@/ui';

import { createDeferred, createFakeService } from './support';

beforeEach(() => {
  setViewportSize(844, 390);
});

afterEach(() => {
  vi.restoreAllMocks();
});

function setViewportSize(width: number, height: number): void {
  Object.defineProperties(window, {
    innerWidth: { configurable: true, value: width },
    innerHeight: { configurable: true, value: height },
  });
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
  const portraitLockedFrame = {
    image: document.createElement('canvas'),
    size: { width: 1280, height: 720 },
    capturedAtMs: 124,
  };
  const captureLatest = vi.fn((options?: {
    readonly aspectRatio?: '16:9' | '9:16';
    readonly rotation?: 0 | 90 | -90;
  }) =>
    options?.aspectRatio === '9:16' && options.rotation === -90
      ? portraitLockedFrame
      : frame,
  );
  const session: CameraSession = {
    preview: { attach, detach },
    captureLatest,
    stop,
  };

  return {
    session,
    attach,
    detach,
    stop,
    captureLatest,
    frame,
    portraitLockedFrame,
  };
}

function createRecognizerHarness() {
  let listener: RealtimeRecognitionListener | null = null;
  let source: RecognitionFrameSource | null = null;
  const runs: Array<{ stop: ReturnType<typeof vi.fn> }> = [];
  const reset = vi.fn();
  const start = vi.fn(
    (
      nextSource: RecognitionFrameSource,
      nextListener: RealtimeRecognitionListener,
    ) => {
      source = nextSource;
      listener = nextListener;
      const stop = vi.fn();
      runs.push({ stop });
      return { stop };
    },
  );
  const dispose = vi.fn(async () => undefined);
  const recognizer: RealtimeRecognizer = { start, reset, dispose };

  return {
    recognizer,
    reset,
    start,
    runs,
    dispose,
    getListener: () => listener,
    getSource: () => source,
  };
}

function renderView({
  camera,
  runtime,
  recognizer,
}: {
  readonly camera: CameraService;
  readonly runtime: RecognitionRuntime;
  readonly recognizer: RealtimeRecognizer;
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

function createRuntime(
  initialize: RecognitionRuntime['initialize'] = async () => undefined,
): RecognitionRuntime {
  return {
    initialize,
    createPipeline() {
      throw new Error('Recognition page test does not create pipelines.');
    },
    async dispose() {},
  };
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
    const cameraDeferred = createDeferred<CameraSession>();
    const runtimeDeferred = createDeferred<void>();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: () => cameraDeferred.promise },
      runtime: createRuntime(() => runtimeDeferred.promise),
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

    const dora = RECOGNITION_CAPTURE_REGIONS['dora-indicators'];
    const hand = RECOGNITION_CAPTURE_REGIONS['completed-hand'];
    const verticalGap = hand.y - (dora.y + dora.height);
    expect(verticalGap).toBeGreaterThanOrEqual(0);
    expect(verticalGap).toBeLessThan(0.05);

    const viewport = screen.getByTestId('recognition-viewport');
    const captureSurface = screen.getByTestId('recognition-capture-surface');
    expect(viewport.style.position).toBe('fixed');
    expect(viewport.style.inset).toBe('0px');
    expect(captureSurface.style.width).toBe('min(100vw, 177.7778dvh)');
    expect(captureSurface.style.height).toBe('min(100dvh, 56.25vw)');
    const exitButton = screen.getByRole('button', { name: '認識を終了' });
    expect(exitButton).toBeVisible();
    expect(exitButton.parentElement).toBe(viewport);
    expect(exitButton.style.position).toBe('absolute');
    expect(exitButton.style.zIndex).toBe('200');
  });

  it('opens camera and runtime in parallel, exposes preview first, then starts realtime only when both are ready', async () => {
    const cameraDeferred = createDeferred<CameraSession>();
    const runtimeDeferred = createDeferred<void>();
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi.fn(() => cameraDeferred.promise);
    const initialize = vi.fn(() => runtimeDeferred.promise);

    renderView({
      camera: { open },
      runtime: createRuntime(initialize),
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
    expect(screen.getByLabelText('カメラプレビュー').style.pointerEvents).toBe('none');
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
    const cameraDeferred = createDeferred<CameraSession>();
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: () => cameraDeferred.promise },
      runtime: createRuntime(),
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

  it('presents the same landscape UI in a portrait-locked viewport', async () => {
    setViewportSize(390, 844);
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: async () => cameraSession.session },
      runtime: createRuntime(),
      recognizer: recognizer.recognizer,
    });

    await waitFor(() => expect(cameraSession.attach).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));

    const captureSurface = screen.getByTestId('recognition-capture-surface');
    const landscapeUi = screen.getByTestId('recognition-landscape-ui-surface');
    const preview = screen.getByLabelText('カメラプレビュー');
    const handRegion = screen.getByLabelText('手牌認識領域');

    expect(captureSurface.style.width).toBe('min(100vw, 56.25dvh)');
    expect(captureSurface.style.height).toBe('min(100dvh, 177.7778vw)');
    expect(captureSurface.style.aspectRatio).toBe('9 / 16');
    expect(preview.style.inset).toBe('0px');
    expect(preview.style.width).toBe('100%');
    expect(preview.style.height).toBe('100%');
    expect(preview.style.objectFit).toBe('cover');
    expect(preview.style.transform).toBe('');

    expect(landscapeUi.style.width).toBe('177.7778%');
    expect(landscapeUi.style.height).toBe('56.25%');
    expect(landscapeUi.style.aspectRatio).toBe('16 / 9');
    expect(landscapeUi.style.transform).toBe(
      'translate(-50%, -50%) rotate(90deg)',
    );
    expect(landscapeUi.style.pointerEvents).toBe('auto');
    expect(handRegion.style.left).toBe('4%');
    expect(handRegion.style.top).toBe('50%');
    expect(handRegion.style.width).toBe('62%');
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
    expect(screen.queryByText(/端末を横向きにしてください/)).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('認識しています');
    const globalExit = screen.getByTestId('recognition-global-exit');
    expect(globalExit.parentElement).toBe(screen.getByTestId('recognition-viewport'));

    expect(recognizer.getSource()?.captureLatest()).toEqual({
      source: cameraSession.portraitLockedFrame.image,
      sourceSize: cameraSession.portraitLockedFrame.size,
      regions: RECOGNITION_CAPTURE_REGIONS,
      capturedAtMs: cameraSession.portraitLockedFrame.capturedAtMs,
    });
    expect(cameraSession.captureLatest).toHaveBeenLastCalledWith({
      aspectRatio: '9:16',
      rotation: -90,
    });

    act(() => {
      setViewportSize(844, 390);
      window.dispatchEvent(new Event('resize'));
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(2));
    expect(recognizer.runs[0]?.stop).toHaveBeenCalled();
    expect(cameraSession.attach).toHaveBeenCalledTimes(1);
    expect(captureSurface.style.width).toBe('min(100vw, 177.7778dvh)');
    expect(captureSurface.style.height).toBe('min(100dvh, 56.25vw)');
    expect(captureSurface.style.aspectRatio).toBe('16 / 9');
    expect(landscapeUi.style.inset).toBe('0px');
    expect(landscapeUi.style.width).toBe('100%');
    expect(landscapeUi.style.height).toBe('100%');
    expect(landscapeUi.style.transform).toBe('');
    expect(globalExit).toBeVisible();
    expect(globalExit.parentElement).toBe(screen.getByTestId('recognition-viewport'));
    expect(handRegion.style.left).toBe('4%');
    expect(handRegion.style.top).toBe('50%');
    expect(handRegion.style.width).toBe('62%');

    expect(recognizer.getSource()?.captureLatest()).toEqual({
      source: cameraSession.frame.image,
      sourceSize: cameraSession.frame.size,
      regions: RECOGNITION_CAPTURE_REGIONS,
      capturedAtMs: cameraSession.frame.capturedAtMs,
    });
    expect(cameraSession.captureLatest).toHaveBeenLastCalledWith({
      aspectRatio: '16:9',
      rotation: 0,
    });
  });
});

describe('RecognitionPageView live feedback', () => {
  it('renders semantic observations and concealed-kan preview without inventing hidden boxes', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: async () => cameraSession.session },
      runtime: createRuntime(),
      recognizer: recognizer.recognizer,
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));

    const handTile = { kind: '7z', red: false } as const;
    const meldTile = { kind: '5s', red: true } as const;
    const concealedKan = {
      kind: 'concealed-kan',
      tiles: [meldTile, meldTile, meldTile, meldTile],
    } as const;
    const snapshot: FrameRecognitionSnapshot = {
      observations: [
        {
          id: observationId('hand-1'),
          region: 'completed-hand',
          bbox: { x: 0.08, y: 0.72, width: 0.05, height: 0.13 },
          classification: { kind: 'tile', tile: handTile },
        },
        {
          id: observationId('meld-1'),
          region: 'melds',
          bbox: { x: 0.75, y: 0.43, width: 0.05, height: 0.12 },
          classification: { kind: 'tile', tile: meldTile },
        },
        {
          id: observationId('meld-2'),
          region: 'melds',
          bbox: { x: 0.84, y: 0.43, width: 0.05, height: 0.12 },
          classification: { kind: 'tile', tile: meldTile },
        },
        {
          id: observationId('dora-unresolved'),
          region: 'dora-indicators',
          bbox: { x: 0.1, y: 0.12, width: 0.05, height: 0.12 },
          classification: { kind: 'invalid' },
        },
      ],
      meldGroups: [
        {
          memberObservationIds: [
            observationId('meld-1'),
            observationId('meld-2'),
          ],
          interpretation: concealedKan,
        },
      ],
      draft: {
        completedHand: [handTile],
        doraIndicators: [],
        meldGroups: [concealedKan],
      },
      commitEligibility: {
        kind: 'ineligible',
        reason: 'insufficient-visible-tiles',
      },
    };

    act(() => {
      recognizer.getListener()?.onUpdate({ kind: 'scanning', snapshot });
    });

    expect(screen.getAllByTestId('recognition-observation-box')).toHaveLength(4);
    const recognized = screen.getByLabelText('認識候補 中');
    const unresolved = screen.getByLabelText('認識候補 未解決');
    expect(recognized).toBeVisible();
    expect(unresolved).toBeVisible();
    expect(recognized).toHaveAttribute('data-recognition-state', 'recognized');
    expect(unresolved).toHaveAttribute('data-recognition-state', 'unresolved');
    expect(recognized.style.border).not.toBe(unresolved.style.border);
    expect(screen.getByTestId('recognition-unresolved-face')).toBeVisible();

    const recognizedIdentity = recognized.querySelector<HTMLElement>(
      '[data-testid="recognition-observation-identity"]',
    );
    expect(recognizedIdentity).not.toBeNull();
    expect(recognizedIdentity?.style.left).toBe('50%');
    expect(recognizedIdentity?.style.bottom).toBe('calc(100% + 2px)');
    expect(recognizedIdentity?.style.transform).toBe('translateX(-50%)');
    const recognizedTileImage = recognized.querySelector<HTMLImageElement>(
      '[data-testid="recognition-tile-face"]',
    );
    expect(recognizedTileImage?.getAttribute('src')).toContain('7z.svg');
    expect(recognizedTileImage?.width).toBe(22);
    expect(recognizedTileImage?.height).toBe(30);

    const connector = screen.getByTestId('meld-group-connector');
    expect(connector).toHaveAttribute('data-overlay-kind', 'meld-connector');
    expect(connector.getAttribute('stroke')).not.toBe('white');
    const meldPreview = screen.getByLabelText(
      '暗槓プレビュー 裏 赤五索 赤五索 裏',
    );
    expect(meldPreview).toBeVisible();
    expect(meldPreview).toHaveAttribute('data-overlay-kind', 'meld-preview');
    const concealedBacks = screen.getAllByTestId('meld-preview-tile-back');
    expect(concealedBacks).toHaveLength(2);
    expect(concealedBacks[0]?.getAttribute('src')).toContain('back.svg');

    const tileFaces = screen.getAllByTestId('recognition-tile-face');
    expect(
      tileFaces.some((face) => face.getAttribute('src')?.includes('5s-red.svg')),
    ).toBe(true);
    expect(
      tileFaces.filter((face) => face.getAttribute('data-red-five') === 'true'),
    ).toHaveLength(4);
    expect(screen.queryByText('7z')).not.toBeInTheDocument();
    expect(screen.queryByText('5s')).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent(
      '認識しています（有効牌 3/10、手牌 1/2）',
    );

    act(() => {
      recognizer.getListener()?.onUpdate({ kind: 'stabilizing', snapshot });
    });

    expect(screen.getByRole('status')).toHaveTextContent(
      '認識結果を安定確認しています',
    );
  });

  it('surfaces unresolved meld geometry instead of an opaque scanning state', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();

    renderView({
      camera: { open: async () => cameraSession.session },
      runtime: createRuntime(),
      recognizer: recognizer.recognizer,
    });

    await waitFor(() => expect(recognizer.start).toHaveBeenCalledTimes(1));

    const snapshot: FrameRecognitionSnapshot = {
      observations: [],
      meldGroups: [],
      draft: {
        completedHand: [],
        doraIndicators: [],
        meldGroups: [],
      },
      commitEligibility: {
        kind: 'ineligible',
        reason: 'unresolved-meld-geometry',
      },
    };

    act(() => {
      recognizer.getListener()?.onUpdate({ kind: 'scanning', snapshot });
    });

    expect(screen.getByRole('status')).toHaveTextContent(
      '副露の配置を調整してください',
    );
  });
});

describe('RecognitionPageView owner-specific recovery', () => {
  it('retries camera without reinitializing an already-ready runtime', async () => {
    const cameraSession = createCameraSession();
    const recognizer = createRecognizerHarness();
    const open = vi
      .fn<CameraService['open']>()
      .mockRejectedValueOnce({ kind: 'device-unavailable' })
      .mockResolvedValueOnce(cameraSession.session);
    const initialize = vi.fn(async () => undefined);

    renderView({
      camera: { open },
      runtime: createRuntime(initialize),
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
      .fn<RecognitionRuntime['initialize']>()
      .mockRejectedValueOnce({
        kind: 'model-initialization-failure',
        model: 'detector',
        cause: new Error('fixture'),
      })
      .mockResolvedValueOnce(undefined);

    renderView({
      camera: { open },
      runtime: createRuntime(initialize),
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
      .fn<CameraService['open']>()
      .mockRejectedValueOnce({ kind: 'permission-denied' })
      .mockResolvedValueOnce(cameraSession.session);
    const initialize = vi
      .fn<RecognitionRuntime['initialize']>()
      .mockRejectedValueOnce({
        kind: 'model-asset-unavailable',
        model: 'tile-classifier',
      })
      .mockResolvedValueOnce(undefined);

    renderView({
      camera: { open },
      runtime: createRuntime(initialize),
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
      runtime: createRuntime(initialize),
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
    expect(screen.getByTestId('recognition-error-diagnostic')).toHaveTextContent(
      'detector / inference-failure / Error: fixture',
    );
    const recoveryLayer = screen.getByTestId('recognition-recovery-layer');
    expect(recoveryLayer.style.pointerEvents).toBe('auto');
    expect(recoveryLayer.style.touchAction).toBe('manipulation');
    const recovery = screen.getByText('認識処理を続行できませんでした').closest('[data-recovery-owner="recognition"]');
    expect(recovery).not.toBeNull();
    expect((recovery as HTMLElement).style.pointerEvents).toBe('auto');
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
    const scoringSession = createScoringSessionPort();
    const services: RecognitionPageServices = {
      camera: { open: async () => cameraSession.session },
      runtime: createRuntime(),
      recognizer: recognizer.recognizer,
    };
    const store = createApplicationStore(
      {
        activeScoringSession: createScoringSessionFixture({
          tileIdPrefix: 'previous',
        }),
      },
      { scoringSessionService: scoringSession },
    );

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

function observationId(value: string): FrameObservationId {
  return value as FrameObservationId;
}

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}
