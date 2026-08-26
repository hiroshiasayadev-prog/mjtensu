import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { App, AppRoutes, productionRouteTable } from '@/app';
import {
  createApplicationStore,
  createCorrectionEditorService,
  createScoringSessionFixture,
  createScoringSessionService,
  type ApplicationStore,
} from '@/application';
import type { ScoringService } from '@/scoring';

const routeScoringService: ScoringService = {
  validateWinningStructure: () => ({ kind: 'valid' }),
  preview: () => ({ kind: 'no-yaku' }),
  calculate: () => {
    throw new Error('route fixture does not calculate');
  },
};

const routeScoringFlowServices = {
  correctionEditor: createCorrectionEditorService(routeScoringService),
};

function createActiveApplicationStore(tileIdPrefix: string): ApplicationStore {
  return createApplicationStore(
    {
      activeScoringSession: createScoringSessionFixture({ tileIdPrefix }),
    },
    {
      scoringSessionService: createScoringSessionService(routeScoringService),
    },
  );
}

function renderRoute(
  route: string,
  applicationStore: ApplicationStore = createApplicationStore(),
  withScoringFlowServices = true,
) {
  render(
    <App
      applicationStore={applicationStore}
      scoringFlowServices={
        withScoringFlowServices ? routeScoringFlowServices : undefined
      }
      router={
        <MemoryRouter initialEntries={[route]}>
          <AppRoutes />
        </MemoryRouter>
      }
    />,
  );
}

describe('production shell routing', () => {
  it('declares the production routes and guards session-owned pages', () => {
    expect(productionRouteTable).toEqual([
      {
        name: 'top',
        path: '/',
        requiresActiveScoringSession: false,
      },
      {
        name: 'recognition',
        path: '/recognition',
        requiresActiveScoringSession: false,
      },
      {
        name: 'recognitionCorrection',
        path: '/recognition/correction',
        requiresActiveScoringSession: true,
      },
      {
        name: 'conditions',
        path: '/conditions',
        requiresActiveScoringSession: true,
      },
      {
        name: 'result',
        path: '/result',
        requiresActiveScoringSession: true,
      },
      {
        name: 'help',
        path: '/help',
        requiresActiveScoringSession: false,
      },
    ]);
  });

  it.each([
    ['/', 'mjtensu'],
    ['/recognition', '認識'],
    ['/help', '使い方'],
  ])('renders the %s page boundary', (route, heading) => {
    renderRoute(route);

    expect(screen.getByRole('heading', { name: heading })).toBeVisible();
  });

  it.each([
    ['/recognition/correction', '認識結果を修正'],
    ['/conditions', '条件入力'],
    ['/result', '結果'],
  ])('renders guarded %s when an active session exists', (route, heading) => {
    renderRoute(route, createActiveApplicationStore('active'));

    expect(screen.getByRole('heading', { name: heading })).toBeVisible();
    expect(screen.getByText('現在の和了牌: active-winning')).toBeVisible();
  });

  it.each([
    ['/conditions', '条件入力'],
    ['/recognition/correction', '認識結果を修正'],
  ])(
    'shows an explicit unavailable state for %s without scoring-flow composition',
    (route, heading) => {
      renderRoute(route, createActiveApplicationStore('unavailable'), false);

      expect(screen.getByRole('heading', { name: heading })).toBeVisible();
      expect(screen.getByRole('alert')).toHaveTextContent(
        '点数計算サービスを利用できません。',
      );
      expect(screen.queryByText('役なし')).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: '計算する' })).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: '修正を確定' })).not.toBeInTheDocument();
    },
  );

  it.each([
    '/recognition/correction',
    '/conditions',
    '/result',
    '/conditions?winningTileId=url-only',
  ])(
    'redirects %s to Top when no active session exists',
    (route) => {
      renderRoute(route);

      expect(screen.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
      expect(screen.queryByText('現在の和了牌: url-only')).not.toBeInTheDocument();
    },
  );

  it('keeps Top Help round-trip navigation from creating a scoring session', () => {
    const applicationStore = createApplicationStore();

    renderRoute('/', applicationStore);

    fireEvent.click(screen.getByRole('button', { name: '使い方' }));
    expect(screen.getByRole('heading', { name: '使い方' })).toBeVisible();
    expect(applicationStore.getState().activeScoringSession).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'トップへ戻る' }));
    expect(screen.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
    expect(applicationStore.getState().activeScoringSession).toBeNull();
  });

  it('starts Recognition only through the explicit Top action and clears prior state', async () => {
    const applicationStore = createActiveApplicationStore('previous');

    renderRoute('/', applicationStore);

    fireEvent.click(screen.getByRole('button', { name: '判定する' }));

    expect(screen.getByRole('heading', { name: '認識' })).toBeVisible();
    await waitFor(() =>
      expect(applicationStore.getState().activeScoringSession).toBeNull(),
    );
  });
});
