import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { App, AppRoutes, productionRouteTable } from '@/app';
import {
  createApplicationStore,
  createScoringSessionFixture,
  type ApplicationStore,
} from '@/application';

function renderRoute(
  route: string,
  applicationStore: ApplicationStore = createApplicationStore(),
) {
  render(
    <App
      applicationStore={applicationStore}
      router={
        <MemoryRouter initialEntries={[route]}>
          <AppRoutes />
        </MemoryRouter>
      }
    />,
  );
}

describe('production shell routing', () => {
  it('declares the five production routes and guards session-owned pages', () => {
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
    ['/conditions', '条件入力'],
    ['/result', '結果'],
  ])('renders guarded %s when an active session exists', (route, heading) => {
    renderRoute(
      route,
      createApplicationStore({
        activeScoringSession: createScoringSessionFixture({
          tileIdPrefix: 'active',
        }),
      }),
    );

    expect(screen.getByRole('heading', { name: heading })).toBeVisible();
    expect(screen.getByText('現在の和了牌: active-winning')).toBeVisible();
  });

  it.each(['/conditions', '/result', '/conditions?winningTileId=url-only'])(
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
    const applicationStore = createApplicationStore({
      activeScoringSession: createScoringSessionFixture({
        tileIdPrefix: 'previous',
      }),
    });

    renderRoute('/', applicationStore);

    fireEvent.click(screen.getByRole('button', { name: '判定する' }));

    expect(screen.getByRole('heading', { name: '認識' })).toBeVisible();
    await waitFor(() =>
      expect(applicationStore.getState().activeScoringSession).toBeNull(),
    );
  });
});
