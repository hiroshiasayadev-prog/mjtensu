import '@mantine/core/styles.css';

import { MantineProvider } from '@mantine/core';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';

import { AppRoutes } from '@/app';
import {
  createApplicationStore,
  createCorrectionEditorService,
  createScoringSessionService,
} from '@/application';
import type { CameraService, CameraSession } from '@/camera';
import type {
  RecognizedStructure,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import type { RecognitionRuntime, RealtimeRecognizer } from '@/recognition';
import type {
  ScoringCalculation,
  ScoringDraft,
  ScoringInput,
  ScoringPreview,
  ScoringService,
  YakuEntry,
} from '@/scoring';
import {
  ApplicationStateProvider,
  RecognitionPageServicesProvider,
  ScoringFlowServicesProvider,
  type RecognitionPageServices,
} from '@/ui';

interface FakeFlowDiagnostics {
  cameraOpenCalls: number;
  cameraStopCalls: number;
  runtimeInitializeCalls: number;
  recognizerStartCalls: number;
  recognizerStopCalls: number;
}

declare global {
  interface Window {
    __MJTENSU_E2E__: FakeFlowDiagnostics;
  }
}

type FakeFlowScenario =
  | 'primary'
  | 'camera-slow'
  | 'runtime-slow'
  | 'camera-retry'
  | 'runtime-retry';

const query = new URLSearchParams(window.location.search);
const scenario = parseScenario(query.get('scenario'));
const initialRoute = parseInitialRoute(query.get('route'));
const diagnostics: FakeFlowDiagnostics = {
  cameraOpenCalls: 0,
  cameraStopCalls: 0,
  runtimeInitializeCalls: 0,
  recognizerStartCalls: 0,
  recognizerStopCalls: 0,
};
window.__MJTENSU_E2E__ = diagnostics;
window.history.replaceState(null, '', initialRoute);

const scoringService = createFakeScoringService();
const scoringSession = createScoringSessionService(scoringService);
const correctionEditor = createCorrectionEditorService(scoringService);
const applicationStore = createApplicationStore(
  {},
  { scoringSessionService: scoringSession },
);
const recognitionServices = createRecognitionServices(scenario, diagnostics);

const rootElement = document.getElementById('root');
if (rootElement === null) {
  throw new Error('Fake-flow root element was not found.');
}

createRoot(rootElement).render(
  <MantineProvider>
    <ApplicationStateProvider store={applicationStore}>
      <ScoringFlowServicesProvider services={{ correctionEditor, scoringSession }}>
        <RecognitionPageServicesProvider services={recognitionServices}>
          <BrowserRouter>
            <AppRoutes />
          </BrowserRouter>
        </RecognitionPageServicesProvider>
      </ScoringFlowServicesProvider>
    </ApplicationStateProvider>
  </MantineProvider>,
);

function parseScenario(value: string | null): FakeFlowScenario {
  switch (value) {
    case 'camera-slow':
    case 'runtime-slow':
    case 'camera-retry':
    case 'runtime-retry':
      return value;
    default:
      return 'primary';
  }
}

function parseInitialRoute(value: string | null): string {
  if (value === '/conditions' || value === '/result') {
    return value;
  }
  return '/';
}

function createRecognitionServices(
  selectedScenario: FakeFlowScenario,
  state: FakeFlowDiagnostics,
): RecognitionPageServices {
  const canvas = document.createElement('canvas');
  let recognitionSequence = 0;

  const camera: CameraService = {
    async open(_request) {
      state.cameraOpenCalls += 1;
      const attempt = state.cameraOpenCalls;

      if (selectedScenario === 'camera-retry' && attempt === 1) {
        throw { kind: 'permission-denied' } as const;
      }
      if (selectedScenario === 'camera-slow' && attempt === 1) {
        await delay(350);
      }

      return createCameraSession(canvas, state);
    },
  };

  const runtime: RecognitionRuntime = {
    async initialize() {
      state.runtimeInitializeCalls += 1;
      const attempt = state.runtimeInitializeCalls;

      if (selectedScenario === 'runtime-retry' && attempt === 1) {
        throw { kind: 'model-asset-unavailable', model: 'detector' } as const;
      }
      if (selectedScenario === 'runtime-slow' && attempt === 1) {
        await delay(350);
      }
    },
    createPipeline() {
      throw new Error('Fake-flow Recognition runtime does not create pipelines.');
    },
    async dispose() {},
  };

  const recognizer: RealtimeRecognizer = {
    reset() {},
    start(_source, listener) {
      state.recognizerStartCalls += 1;
      recognitionSequence += 1;
      let active = true;

      const timer = window.setTimeout(() => {
        if (!active) {
          return;
        }
        listener.onUpdate({
          kind: 'confirmed',
          result: createRecognizedStructure(`recognition-${recognitionSequence}`),
        });
      }, 60);

      return {
        stop() {
          if (!active) {
            return;
          }
          active = false;
          window.clearTimeout(timer);
          state.recognizerStopCalls += 1;
        },
      };
    },
    async dispose() {},
  };

  return {
    camera,
    runtime,
    recognizer,
  };
}

function createCameraSession(
  canvas: HTMLCanvasElement,
  state: FakeFlowDiagnostics,
): CameraSession {
  let stopped = false;

  return {
    preview: {
      attach() {},
      detach() {},
    },
    captureLatest() {
      if (stopped) {
        return null;
      }
      return {
        image: canvas,
        size: { width: 1280, height: 720 },
        capturedAtMs: 1,
      };
    },
    async stop() {
      if (stopped) {
        return;
      }
      stopped = true;
      state.cameraStopCalls += 1;
    },
  };
}

function createFakeScoringService(): ScoringService {
  return {
    validateWinningStructure(structure) {
      if (structure.completedHand.length !== 14) {
        return {
          kind: 'invalid-structure',
          issues: [{ kind: 'completed-hand-count' }],
        };
      }
      return { kind: 'valid' };
    },

    preview(draft) {
      return fakePreview(draft);
    },

    calculate(input) {
      const draft: ScoringDraft = {
        structure: {
          completedHand: input.completedHand,
          meldGroups: [],
          doraIndicators: input.doraIndicators.map((tile, index) => ({
            id: `calculation-dora-${index}` as TileInstanceId,
            tile,
          })),
        },
        winningTileId: input.winningTileId,
        conditions: input.conditions,
      };
      const preview = fakePreview(draft);
      if (preview.kind !== 'ready' || preview.yaku.length === 0) {
        throw new RangeError('Fake scoring calculation requires a ready preview.');
      }
      return fakeCalculation(input, preview.yaku);
    },
  };
}

function fakePreview(draft: ScoringDraft): ScoringPreview {
  const winningTile = draft.structure.completedHand.find(
    (tile) => tile.id === draft.winningTileId,
  );

  if (winningTile === undefined) {
    return {
      kind: 'invalid-input',
      issues: [{ kind: 'winning-tile-not-in-completed-hand' }],
    };
  }

  if (winningTile.tile.kind === '9s') {
    return { kind: 'invalid-winning-shape' };
  }

  if (
    draft.conditions.roundWind === 'west' &&
    draft.conditions.seatWind === 'north'
  ) {
    return {
      kind: 'invalid-input',
      issues: [{ kind: 'contradictory-conditions' }],
    };
  }

  if (
    draft.structure.completedHand[0]?.tile.kind === '9p' ||
    (draft.conditions.winMethod === 'ron' && draft.conditions.riichi === 'none')
  ) {
    return { kind: 'no-yaku' };
  }

  return {
    kind: 'ready',
    yaku: readyYaku(draft),
  };
}

function readyYaku(draft: ScoringDraft): readonly YakuEntry[] {
  if (draft.conditions.winMethod === 'tsumo') {
    return [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }];
  }
  return [
    {
      kind: 'regular',
      id: draft.conditions.riichi === 'double-riichi' ? 'double-riichi' : 'riichi',
      han: draft.conditions.riichi === 'double-riichi' ? 2 : 1,
    },
  ];
}

function fakeCalculation(
  input: ScoringInput,
  yaku: readonly YakuEntry[],
): ScoringCalculation {
  const corrected = input.completedHand[0]?.tile.kind === '2m';
  const dealer = input.conditions.seatWind === 'east';
  const totalPoints = corrected ? 12000 : dealer ? 6000 : 4000;
  const payment = input.conditions.winMethod === 'ron'
    ? { kind: 'ron' as const, amount: totalPoints }
    : dealer
      ? { kind: 'tsumo-dealer' as const, eachOpponent: totalPoints / 3 }
      : {
          kind: 'tsumo-non-dealer' as const,
          dealerPays: totalPoints / 2,
          nonDealerPays: totalPoints / 4,
        };

  return {
    yaku,
    dora: { dora: 0, akaDora: 0 },
    han: yaku.reduce(
      (sum, entry) => sum + (entry.kind === 'regular' ? entry.han : 0),
      0,
    ),
    fu: {
      kind: 'standard',
      base: 20,
      menzenRon: input.conditions.winMethod === 'ron' ? 10 : 0,
      tsumo: input.conditions.winMethod === 'tsumo' ? 2 : 0,
      melds: 0,
      pair: 0,
      wait: 0,
      rawTotal: 30,
      rounded: 30,
    },
    limit: null,
    winnerRole: dealer ? 'dealer' : 'non-dealer',
    winMethod: input.conditions.winMethod,
    payment,
    totalPoints,
  };
}

function createRecognizedStructure(prefix: string): RecognizedStructure {
  const kinds: readonly TileKind[] = [
    '1m',
    '9s',
    '2m',
    '3m',
    '4m',
    '5m',
    '6m',
    '7m',
    '8m',
    '9m',
    '1p',
    '2p',
    '3p',
    '4p',
  ];

  return {
    completedHand: kinds.map((kind, index) =>
      tile(`${prefix}-hand-${index + 1}`, kind),
    ),
    doraIndicators: [tile(`${prefix}-dora-1`, '5z')],
    meldGroups: [],
  };
}

function tile(id: string, kind: TileKind): TileInstance {
  return {
    id: id as TileInstanceId,
    tile: { kind, red: false },
  };
}

async function delay(ms: number): Promise<void> {
  await new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}
