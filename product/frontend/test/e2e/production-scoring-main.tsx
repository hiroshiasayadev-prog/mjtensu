import '@mantine/core/styles.css';

import { createRoot } from 'react-dom/client';

import { App, createProductionServiceGraph } from '@/app';
import type {
  RecognizedStructure,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import { registerProductionPwaLifecycle } from '@/pwa';
import { DEFAULT_RULE_PROFILE } from '@/scoring';

interface ProductionScoringDiagnostics {
  status: 'booting' | 'ready' | 'failed';
  error?: string;
}

declare global {
  interface Window {
    __MJTENSU_PRODUCTION_SCORING__: ProductionScoringDiagnostics;
  }
}

window.__MJTENSU_PRODUCTION_SCORING__ = { status: 'booting' };
void bootstrap();

async function bootstrap(): Promise<void> {
  try {
    const root = document.getElementById('root');
    if (root === null) {
      throw new Error('Application root element #root was not found.');
    }

    const services = await createProductionServiceGraph();
    const structure = verificationStructure();
    services.applicationStore.getState().createScoringSession(
      structure,
      DEFAULT_RULE_PROFILE,
    );
    services.applicationStore.getState().updateScoringSession({
      kind: 'select-winning-tile',
      tileId: structure.completedHand[12].id,
    });
    services.applicationStore.getState().updateScoringSession({
      kind: 'replace-conditions',
      conditions: {
        winMethod: 'ron',
        roundWind: 'east',
        seatWind: 'south',
        riichi: 'none',
        ippatsu: false,
        rinshan: false,
        chankan: false,
        haitei: false,
        houtei: false,
        tenhou: false,
        chiihou: false,
      },
    });

    window.history.replaceState({}, '', '/conditions');
    createRoot(root).render(
      <App
        applicationStore={services.applicationStore}
        recognitionPageServices={services.recognitionPageServices}
        scoringFlowServices={services.scoringFlowServices}
      />,
    );

    await registerProductionPwaLifecycle();
    window.__MJTENSU_PRODUCTION_SCORING__ = { status: 'ready' };
  } catch (error) {
    window.__MJTENSU_PRODUCTION_SCORING__ = {
      status: 'failed',
      error: error instanceof Error ? `${error.name}: ${error.message}` : String(error),
    };
  }
}

function verificationStructure(): RecognizedStructure {
  return {
    completedHand: [
      tile('s03-1', '2m'),
      tile('s03-2', '3m'),
      tile('s03-3', '4m'),
      tile('s03-4', '3p'),
      tile('s03-5', '4p'),
      tile('s03-6', '5p', true),
      tile('s03-7', '4s'),
      tile('s03-8', '5s'),
      tile('s03-9', '6s'),
      tile('s03-10', '6m'),
      tile('s03-11', '7m'),
      tile('s03-12', '8m'),
      tile('s03-13', '6p'),
      tile('s03-14', '6p'),
    ],
    doraIndicators: [],
    meldGroups: [],
  };
}

function tile(id: string, kind: TileKind, red = false): TileInstance {
  return {
    id: id as TileInstanceId,
    tile: { kind, red },
  };
}
