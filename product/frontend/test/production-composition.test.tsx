import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import {
  App,
  AppRoutes,
  createProductionServiceGraph,
} from '@/app';
import type { CameraService } from '@/camera';
import type { RecognizedStructure, TileInstanceId } from '@/domain';
import type { ProductionRecognitionServices } from '@/recognition';
import {
  DEFAULT_RULE_PROFILE,
  type ScoringService,
} from '@/scoring';

function controlledDependencies() {
  const camera: CameraService = {
    open: vi.fn(async () => ({
      preview: {
        attach: vi.fn(),
        detach: vi.fn(),
      },
      captureLatest: () => null,
      stop: vi.fn(async () => undefined),
    })),
  };
  const prefetch = vi.fn(async () => undefined);
  const recognitionDispose = vi.fn(async () => undefined);
  const runtime = {
    initialize: vi.fn(async () => undefined),
    createPipeline: vi.fn(() => {
      throw new Error('pipeline creation is not expected in this test');
    }),
    dispose: vi.fn(async () => undefined),
  };
  const recognizer = {
    start: vi.fn(() => ({ stop: vi.fn() })),
    reset: vi.fn(),
    dispose: vi.fn(async () => undefined),
  };
  const recognition: ProductionRecognitionServices = {
    assets: { prefetch },
    runtime,
    recognizer,
    prefetch,
    dispose: recognitionDispose,
  };
  const scoring: ScoringService = {
    validateWinningStructure: () => ({ kind: 'valid' }),
    preview: () => ({ kind: 'no-yaku' }),
    calculate: () => {
      throw new Error('calculation is not expected in this test');
    },
  };

  return {
    camera,
    prefetch,
    recognition,
    recognitionDispose,
    runtime,
    scoring,
  };
}

describe('production service composition', () => {
  it('constructs one stable public service graph and binds the Application store to the real service seam', async () => {
    const controlled = controlledDependencies();
    const loadScoringService = vi.fn(async () => controlled.scoring);
    const graph = await createProductionServiceGraph({
      createCameraService: () => controlled.camera,
      createRecognitionServices: () => controlled.recognition,
      loadScoringService,
    });

    expect(loadScoringService).toHaveBeenCalledTimes(1);
    expect(graph.recognitionPageServices).toEqual({
      camera: controlled.camera,
      runtime: controlled.recognition.runtime,
      recognizer: controlled.recognition.recognizer,
    });
    expect(graph.scoringFlowServices.scoringSession).toBe(graph.scoringSession);

    const session = graph.applicationStore.getState().createScoringSession(
      minimalStructure(),
      DEFAULT_RULE_PROFILE,
    );
    expect(session.winningTileId).toBe('tile-1');
    expect(graph.applicationStore.getState().activeScoringSession).toBe(session);
  });

  it('renders Top with injected production-boundary services without starting recognition acquisition or runtime initialization', async () => {
    const controlled = controlledDependencies();
    const graph = await createProductionServiceGraph({
      createCameraService: () => controlled.camera,
      createRecognitionServices: () => controlled.recognition,
      loadScoringService: async () => controlled.scoring,
    });

    render(
      <App
        applicationStore={graph.applicationStore}
        recognitionPageServices={graph.recognitionPageServices}
        router={
          <MemoryRouter initialEntries={['/']}>
            <AppRoutes />
          </MemoryRouter>
        }
        scoringFlowServices={graph.scoringFlowServices}
      />,
    );

    expect(screen.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
    expect(controlled.prefetch).not.toHaveBeenCalled();
    expect(controlled.runtime.initialize).not.toHaveBeenCalled();

    await graph.prefetchRecognitionModels();
    expect(controlled.prefetch).toHaveBeenCalledTimes(1);
  });

  it('keeps app-lifetime recognition resources owned by the graph until explicit graph disposal', async () => {
    const controlled = controlledDependencies();
    const graph = await createProductionServiceGraph({
      createCameraService: () => controlled.camera,
      createRecognitionServices: () => controlled.recognition,
      loadScoringService: async () => controlled.scoring,
    });

    expect(controlled.recognitionDispose).not.toHaveBeenCalled();
    await graph.dispose();
    await graph.dispose();
    expect(controlled.recognitionDispose).toHaveBeenCalledTimes(1);
  });
});

function minimalStructure(): RecognizedStructure {
  return {
    completedHand: [
      {
        id: 'tile-1' as TileInstanceId,
        tile: { kind: '1m', red: false },
      },
    ],
    doraIndicators: [],
    meldGroups: [],
  };
}
