import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import { App, AppRoutes } from '@/app';
import {
  createApplicationStore,
  INITIAL_SCORING_CONDITIONS,
  type ApplicationStore,
} from '@/application';
import type {
  RecognizedMeldGroup,
  RecognizedStructure,
  TileInstance,
  TileInstanceId,
} from '@/domain';
import { DEFAULT_RULE_PROFILE, type ScoringCalculation } from '@/scoring';

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}

function tile(
  id: string,
  kind: TileInstance['tile']['kind'],
  red = false,
): TileInstance {
  return {
    id: tileId(id),
    tile: { kind, red },
  };
}

const completedHand = [
  tile('hand-1', '1m'),
  tile('hand-2', '2m'),
  tile('hand-3', '3m'),
  tile('hand-4', '4p'),
  tile('winning', '5p', true),
] as const;

const meldTiles = [
  tile('meld-1', '7s'),
  tile('meld-2', '8s'),
  tile('meld-3', '9s'),
] as const;

const doraIndicators = [tile('dora-1', '5z'), tile('dora-2', '5m')] as const;

const structure: RecognizedStructure = {
  completedHand,
  doraIndicators,
  meldGroups: [
    {
      kind: 'chi',
      tiles: meldTiles,
    } satisfies RecognizedMeldGroup,
  ],
};

const standardFu = {
  kind: 'standard',
  base: 20,
  menzenRon: 10,
  tsumo: 0,
  melds: 4,
  pair: 2,
  wait: 2,
  rawTotal: 38,
  rounded: 40,
} as const;

const baseCalculation: ScoringCalculation = {
  yaku: [
    { kind: 'regular', id: 'riichi', han: 1 },
    { kind: 'regular', id: 'sanshoku-doujun', han: 2 },
  ],
  dora: { dora: 2, akaDora: 1 },
  han: 6,
  fu: standardFu,
  limit: { kind: 'haneman' },
  winnerRole: 'non-dealer',
  winMethod: 'ron',
  payment: { kind: 'ron', amount: 12000 },
  totalPoints: 12000,
};

function calculation(
  overrides: Partial<ScoringCalculation> = {},
): ScoringCalculation {
  return {
    ...baseCalculation,
    ...overrides,
  };
}

function renderResult(
  latestResult: ScoringCalculation,
  route = '/result',
): ApplicationStore {
  const applicationStore = createApplicationStore({
    activeScoringSession: {
      structure,
      winningTileId: tileId('winning'),
      conditions: INITIAL_SCORING_CONDITIONS,
      ruleProfile: DEFAULT_RULE_PROFILE,
      latestResult,
    },
  });

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

  return applicationStore;
}

describe('result page presentation', () => {
  it('renders recognized evidence and marks the selected winning tile instance', () => {
    renderResult(calculation());

    expect(screen.getByRole('heading', { name: '結果' })).toBeVisible();
    expect(screen.getByLabelText('赤5筒 和了牌')).toHaveAttribute(
      'data-winning',
      'true',
    );
    expect(screen.getByText('白')).toBeVisible();
    expect(screen.getByText('5萬')).toBeVisible();
    expect(screen.getByLabelText('chi meld')).toBeVisible();
  });

  it('renders product yaku names and separates indicator dora from aka dora', () => {
    renderResult(calculation());

    expect(screen.getByText('立直')).toBeVisible();
    expect(screen.getByText('三色同順')).toBeVisible();
    expect(screen.getByText('ドラ')).toBeVisible();
    expect(screen.getByText('赤ドラ')).toBeVisible();
    expect(screen.getByText('6翻')).toBeVisible();
  });

  it.each([
    {
      name: 'ron',
      payment: { kind: 'ron', amount: 7700 } as const,
      totalPoints: 7700,
      expected: 'ロン 放銃者 7,700点',
    },
    {
      name: 'dealer tsumo',
      payment: { kind: 'tsumo-dealer', eachOpponent: 4000 } as const,
      totalPoints: 12000,
      winnerRole: 'dealer' as const,
      expected: 'ツモ 親: 各家 4,000点',
    },
    {
      name: 'non-dealer tsumo',
      payment: {
        kind: 'tsumo-non-dealer',
        dealerPays: 2600,
        nonDealerPays: 1300,
      } as const,
      totalPoints: 5200,
      expected: 'ツモ 子: 子 1,300点 / 親 2,600点',
    },
  ])('renders $name payment from ScoringPayment fields', (fixture) => {
    renderResult(
      calculation({
        payment: fixture.payment,
        totalPoints: fixture.totalPoints,
        winnerRole: fixture.winnerRole ?? 'non-dealer',
      }),
    );

    expect(screen.getByText(fixture.expected)).toBeVisible();
  });

  it.each([
    {
      name: 'non-limit',
      fixture: calculation({ han: 3, fu: standardFu, limit: null }),
      expected: ['40符', '3翻', '通常'],
    },
    {
      name: 'kiriage mangan',
      fixture: calculation({
        han: 4,
        fu: { ...standardFu, rounded: 30 },
        limit: { kind: 'mangan', kiriage: true },
      }),
      expected: ['30符', '4翻', '切り上げ満貫'],
    },
    {
      name: 'baiman',
      fixture: calculation({ han: 8, limit: { kind: 'baiman' } }),
      expected: ['40符', '8翻', '倍満'],
    },
    {
      name: 'counted yakuman',
      fixture: calculation({
        yaku: [{ kind: 'regular', id: 'chinitsu', han: 6 }],
        dora: { dora: 7, akaDora: 0 },
        han: 13,
        fu: null,
        limit: { kind: 'yakuman', units: 1, counted: true },
        totalPoints: 32000,
      }),
      expected: ['13翻', '数え役満 1倍', '32,000点'],
    },
    {
      name: 'actual yakuman',
      fixture: calculation({
        yaku: [{ kind: 'yakuman', id: 'daisangen' }],
        dora: { dora: 0, akaDora: 0 },
        han: null,
        fu: null,
        limit: { kind: 'yakuman', units: 2, counted: false },
        totalPoints: 64000,
      }),
      expected: ['大三元', '役満 2倍', '64,000点'],
    },
  ])('renders score hierarchy for $name', ({ fixture, expected }) => {
    renderResult(fixture);

    for (const text of expected) {
      expect(screen.getByText(text)).toBeVisible();
    }
  });

  it('does not invent han or fu detail for actual yakuman', () => {
    renderResult(
      calculation({
        yaku: [{ kind: 'yakuman', id: 'daisangen' }],
        dora: { dora: 0, akaDora: 0 },
        han: null,
        fu: null,
        limit: { kind: 'yakuman', units: 1, counted: false },
      }),
    );

    expect(screen.queryByRole('button', { name: '符の詳細' })).not.toBeInTheDocument();
    expect(screen.queryByText(/符$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\d+翻/)).not.toBeInTheDocument();
  });

  it('renders aggregate standard fu detail without per-meld reconstruction', () => {
    renderResult(calculation());

    fireEvent.click(screen.getByRole('button', { name: '符の詳細' }));

    const dialog = screen.getByRole('dialog', { name: '符の詳細' });
    expect(within(dialog).getByText('副底')).toBeVisible();
    expect(within(dialog).getByText('門前ロン')).toBeVisible();
    expect(within(dialog).getByText('面子')).toBeVisible();
    expect(within(dialog).getByText('合計')).toBeVisible();
    expect(within(dialog).getByText('切り上げ後')).toBeVisible();
    expect(within(dialog).getByText('38符')).toBeVisible();
    expect(within(dialog).getByText('40符')).toBeVisible();
  });

  it('renders chiitoitsu as fixed 25 fu', () => {
    renderResult(
      calculation({
        yaku: [{ kind: 'regular', id: 'chiitoitsu', han: 2 }],
        han: 2,
        fu: { kind: 'chiitoitsu', fixed: 25 },
        limit: null,
        totalPoints: 1600,
      }),
    );

    fireEvent.click(screen.getByRole('button', { name: '符の詳細' }));

    expect(screen.getByText('七対子は固定25符です。')).toBeVisible();
    expect(screen.getByText('最終符')).toBeVisible();
    expect(screen.getAllByText('25符')).toHaveLength(2);
  });

  it('routes correction and restart actions without mutating unrelated session state', () => {
    const applicationStore = renderResult(calculation());
    const initialSession = applicationStore.getState().activeScoringSession;

    fireEvent.click(screen.getByRole('button', { name: '親子を修正' }));
    expect(screen.getByRole('heading', { name: '条件入力' })).toBeVisible();
    expect(applicationStore.getState().activeScoringSession).toBe(initialSession);
    expect(
      applicationStore.getState().activeScoringSession?.conditions.seatWind,
    ).toBe('east');
  });

  it('preserves conditions for recognition correction', () => {
    const applicationStore = renderResult(calculation());
    const initialSession = applicationStore.getState().activeScoringSession;

    fireEvent.click(screen.getByRole('button', { name: '認識結果を修正' }));
    expect(
      screen.getByRole('heading', { name: '認識結果を修正' }),
    ).toBeVisible();
    expect(applicationStore.getState().activeScoringSession).toBe(initialSession);
  });

  it('clears state for explicit new recognition', async () => {
    const applicationStore = renderResult(calculation());

    fireEvent.click(screen.getByRole('button', { name: 'もう一度判定' }));
    expect(screen.getByRole('heading', { name: '認識' })).toBeVisible();
    await waitFor(() =>
      expect(applicationStore.getState().activeScoringSession).toBeNull(),
    );
  });
});
