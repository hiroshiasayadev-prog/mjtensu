import { fireEvent, render, screen, within } from '@testing-library/react';
import {
  createCorrectionEditorService,
  createScoringSessionService,
  type ScoringSessionState,
} from '@/application';
import type {
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import {
  DEFAULT_RULE_PROFILE,
  type ScoringCalculation,
  type ScoringPreview,
  type ScoringService,
} from '@/scoring';
import {
  RecognitionCorrectionPageView,
  TileCorrectionEditor,
} from '@/ui';
import { describe, expect, it, vi } from 'vitest';

import { createFakeService } from './support';

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}

function tile(id: string, kind: TileKind, red = false): TileInstance {
  return { id: tileId(id), tile: { kind, red } };
}

function tiles(prefix: string, kinds: readonly TileKind[]): readonly TileInstance[] {
  return kinds.map((kind, index) => tile(`${prefix}-${index + 1}`, kind));
}

function structure(
  completedHand: readonly TileInstance[],
  overrides: Partial<RecognizedStructure> = {},
): RecognizedStructure {
  return {
    completedHand,
    meldGroups: [],
    doraIndicators: [],
    ...overrides,
  };
}

const closedWinningHand = tiles('closed', [
  '1m',
  '2m',
  '3m',
  '4m',
  '5m',
  '6m',
  '7m',
  '8m',
  '9m',
  '1p',
  '1p',
  '1p',
  '2p',
  '2p',
]);

const oneMeldStructure = structure(
  tiles('hand', [
    '1m',
    '2m',
    '3m',
    '4m',
    '5m',
    '6m',
    '7m',
    '8m',
    '9m',
    '1p',
    '1p',
  ]),
  {
    meldGroups: [
      {
        kind: 'chi',
        tiles: [tile('meld-1', '1s'), tile('meld-2', '2s'), tile('meld-3', '3s')],
      },
    ],
  },
);

const readyPreview: ScoringPreview = {
  kind: 'ready',
  yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
};

const calculation: ScoringCalculation = {
  yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
  dora: { dora: 0, akaDora: 0 },
  han: 1,
  fu: {
    kind: 'standard',
    base: 20,
    menzenRon: 0,
    tsumo: 2,
    melds: 0,
    pair: 0,
    wait: 0,
    rawTotal: 22,
    rounded: 30,
  },
  limit: null,
  winnerRole: 'dealer',
  winMethod: 'tsumo',
  payment: { kind: 'tsumo-dealer', eachOpponent: 500 },
  totalPoints: 1500,
};

function scoringService(
  preview: ScoringPreview = readyPreview,
  overrides: Partial<ScoringService> = {},
): ScoringService {
  return createFakeService<ScoringService>(
    {
      validateWinningStructure: () => ({ kind: 'valid' }),
      preview: () => preview,
      calculate: () => calculation,
    },
    overrides,
  );
}

function renderEditor(initialStructure: RecognizedStructure = structure(closedWinningHand)) {
  const service = createCorrectionEditorService(scoringService());
  const update = vi.spyOn(service, 'update');
  const onCommit = vi.fn();

  render(
    <TileCorrectionEditor
      initialStructure={initialStructure}
      onCommit={onCommit}
      service={service}
    />,
  );

  return { onCommit, service, update };
}

function resultOriginSession(
  preview: ScoringPreview,
  overrides: Partial<ScoringService> = {},
) {
  const scoring = scoringService(preview, overrides);
  const sessionService = createScoringSessionService(scoring);
  const created = sessionService.create(
    structure(closedWinningHand),
    DEFAULT_RULE_PROFILE,
  );

  return {
    correctionEditorService: createCorrectionEditorService(scoring),
    session: { ...created, latestResult: calculation } satisfies ScoringSessionState,
    sessionService,
  };
}

describe('TileCorrectionEditor', () => {
  it('opens insertion without a placeholder and emits add only after choosing a red five', () => {
    const { update } = renderEditor();

    fireEvent.click(screen.getByRole('button', { name: '手牌に追加' }));

    expect(update).not.toHaveBeenCalled();
    const selector = screen.getByRole('dialog', { name: '牌を選択' });
    fireEvent.click(within(selector).getByRole('button', { name: '赤5m' }));

    expect(update).toHaveBeenLastCalledWith(
      expect.anything(),
      {
        kind: 'add-tile',
        destination: { kind: 'completed-hand' },
        tile: { kind: '5m', red: true } satisfies TileIdentity,
        index: undefined,
      },
    );
  });

  it('emits replacement and deletion against the existing tile instance', () => {
    const { update } = renderEditor();

    fireEvent.click(screen.getByRole('button', { name: '手牌 1 1m' }));
    fireEvent.click(
      within(screen.getByRole('dialog', { name: '牌を選択' })).getByRole('button', {
        name: '4m',
      }),
    );

    expect(update).toHaveBeenLastCalledWith(
      expect.anything(),
      {
        kind: 'replace-tile',
        tileId: tileId('closed-1'),
        tile: { kind: '4m', red: false },
      },
    );

    fireEvent.click(screen.getByRole('button', { name: '手牌 1 4m' }));
    fireEvent.click(
      within(screen.getByRole('dialog', { name: '牌を選択' })).getByRole('button', {
        name: 'この牌を削除',
      }),
    );

    expect(update).toHaveBeenLastCalledWith(expect.anything(), {
      kind: 'remove-tile',
      tileId: tileId('closed-1'),
    });
  });

  it('supports local reorder and movement between semantic regions', () => {
    const { update } = renderEditor(oneMeldStructure);

    fireEvent.click(screen.getByRole('button', { name: '手牌 2 2m' }));
    fireEvent.click(
      within(screen.getByRole('dialog', { name: '牌を選択' })).getByRole('button', {
        name: '右へ',
      }),
    );

    expect(update).toHaveBeenLastCalledWith(expect.anything(), {
      kind: 'move-tile',
      tileId: tileId('hand-2'),
      destination: { kind: 'completed-hand' },
      index: 2,
    });

    fireEvent.click(screen.getByRole('button', { name: '手牌 1 1m' }));
    fireEvent.click(
      within(screen.getByRole('dialog', { name: '牌を選択' })).getByRole('button', {
        name: '副露 1へ移動',
      }),
    );

    expect(update.mock.calls.at(-1)?.[1]).toEqual(
      expect.objectContaining({
        kind: 'move-tile',
        tileId: tileId('hand-1'),
        destination: expect.objectContaining({ kind: 'meld' }),
      }),
    );
  });

  it('shows every four-equal kan openness explicitly and lets the user flip it', () => {
    const initialStructure = structure(
      tiles('kan-hand', [
        '1m',
        '2m',
        '3m',
        '4m',
        '5m',
        '6m',
        '7m',
        '8m',
        '9m',
        '1p',
        '1p',
      ]),
      {
        meldGroups: [
          {
            kind: 'concealed-kan',
            tiles: [
              tile('kan-1', '7s'),
              tile('kan-2', '7s'),
              tile('kan-3', '7s'),
              tile('kan-4', '7s'),
            ],
          },
        ],
      },
    );
    const { update } = renderEditor(initialStructure);

    fireEvent.click(
      screen.getByRole('button', { name: '副露 1 槓種別 暗槓' }),
    );

    expect(update.mock.calls.at(-1)?.[1]).toEqual(
      expect.objectContaining({ kind: 'toggle-kan-openness' }),
    );
    expect(
      screen.getByRole('button', { name: '副露 1 槓種別 明槓' }),
    ).toBeVisible();
  });

  it('keeps incomplete melds visibly editable and blocks commit with targeted text', () => {
    renderEditor();

    fireEvent.click(screen.getByRole('button', { name: '副露を追加' }));

    expect(screen.getByLabelText('副露 1')).toHaveAttribute('data-invalid', 'true');
    expect(screen.getByText('副露の牌構成を修正してください。')).toBeVisible();
    expect(screen.getByRole('button', { name: '副露 1 に牌を追加' })).toBeVisible();
    expect(screen.getByRole('button', { name: '修正を確定' })).toBeDisabled();
  });

  it('targets completed-hand count feedback and blocks the primary commit action', () => {
    renderEditor(structure(closedWinningHand.slice(0, 13)));

    expect(screen.getByLabelText('手牌修正')).toHaveAttribute('data-invalid', 'true');
    expect(screen.getByText('手牌の枚数が副露数と合っていません。')).toBeVisible();
    expect(screen.getByRole('button', { name: '修正を確定' })).toBeDisabled();
  });

  it('shows completed-hand tile validity feedback from scoring validation', () => {
    const service = createCorrectionEditorService(
      scoringService(readyPreview, {
        validateWinningStructure: () => ({
          kind: 'invalid-structure',
          issues: [{ kind: 'completed-hand-tile', tileIndex: 4 }],
        }),
      }),
    );

    render(
      <TileCorrectionEditor
        initialStructure={structure(closedWinningHand)}
        onCommit={vi.fn()}
        service={service}
      />,
    );

    expect(screen.getByLabelText('手牌修正')).toHaveAttribute('data-invalid', 'true');
    expect(screen.getByText('手牌に不正な牌があります。')).toBeVisible();
    expect(screen.getByRole('button', { name: '修正を確定' })).toBeDisabled();
  });

  it('shows whole-structure winning-shape feedback without blaming dora or conditions', () => {
    const service = createCorrectionEditorService(
      scoringService(readyPreview, {
        validateWinningStructure: () => ({ kind: 'not-winning-shape' }),
      }),
    );

    render(
      <TileCorrectionEditor
        initialStructure={structure(closedWinningHand)}
        onCommit={vi.fn()}
        service={service}
      />,
    );

    expect(
      screen.getByText('和了形として成立する牌姿に修正してください。'),
    ).toBeVisible();
    expect(screen.getByLabelText('ドラ表示牌修正')).toHaveAttribute(
      'data-invalid',
      'false',
    );
    expect(screen.getByRole('button', { name: '修正を確定' })).toBeDisabled();
  });
});

describe('RecognitionCorrectionPageView', () => {
  it('cancels without changing the old Result session', () => {
    const fixture = resultOriginSession({ kind: 'no-yaku' });
    const onCancel = vi.fn();
    const onSessionChange = vi.fn();

    render(
      <RecognitionCorrectionPageView
        {...fixture}
        onCancel={onCancel}
        onContinueToConditions={vi.fn()}
        onReturnToResult={vi.fn()}
        onSessionChange={onSessionChange}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'キャンセル' }));

    expect(onCancel).toHaveBeenCalledOnce();
    expect(onSessionChange).not.toHaveBeenCalled();
  });

  it('installs and recalculates a ready correction before returning to Result', () => {
    const fixture = resultOriginSession({
      kind: 'ready',
      yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
    });
    const onSessionChange = vi.fn();
    const onReturnToResult = vi.fn();
    const onContinueToConditions = vi.fn();

    render(
      <RecognitionCorrectionPageView
        {...fixture}
        onCancel={vi.fn()}
        onContinueToConditions={onContinueToConditions}
        onReturnToResult={onReturnToResult}
        onSessionChange={onSessionChange}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: '修正を確定' }));

    expect(onSessionChange).toHaveBeenCalledTimes(2);
    expect(onSessionChange.mock.calls[0]?.[0].latestResult).toBeNull();
    expect(onSessionChange.mock.calls[1]?.[0].latestResult).toBe(calculation);
    expect(onReturnToResult).toHaveBeenCalledOnce();
    expect(onContinueToConditions).not.toHaveBeenCalled();
  });

  it('keeps the confirmed corrected session when recalculation fails', () => {
    const fixture = resultOriginSession(
      {
        kind: 'ready',
        yaku: [{ kind: 'regular', id: 'menzen-tsumo', han: 1 }],
      },
      {
        calculate: () => {
          throw new Error('recalculation failed');
        },
      },
    );
    const onSessionChange = vi.fn();
    const onReturnToResult = vi.fn();
    const onContinueToConditions = vi.fn();

    render(
      <RecognitionCorrectionPageView
        {...fixture}
        onCancel={vi.fn()}
        onContinueToConditions={onContinueToConditions}
        onReturnToResult={onReturnToResult}
        onSessionChange={onSessionChange}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: '修正を確定' }));

    expect(onSessionChange).toHaveBeenCalledOnce();
    expect(onSessionChange.mock.calls[0]?.[0].latestResult).toBeNull();
    expect(onContinueToConditions).toHaveBeenCalledOnce();
    expect(onReturnToResult).not.toHaveBeenCalled();
  });

  it.each([
    { kind: 'no-yaku' } as const,
    { kind: 'incomplete', missing: ['seat-wind'] } as const,
    {
      kind: 'invalid-input',
      issues: [{ kind: 'contradictory-conditions' }],
    } as const,
  ])('keeps the confirmed corrected session and routes $kind to Conditions', (preview) => {
    const calculate = vi.fn(() => calculation);
    const fixture = resultOriginSession(preview, { calculate });
    const onSessionChange = vi.fn();
    const onReturnToResult = vi.fn();
    const onContinueToConditions = vi.fn();

    render(
      <RecognitionCorrectionPageView
        {...fixture}
        onCancel={vi.fn()}
        onContinueToConditions={onContinueToConditions}
        onReturnToResult={onReturnToResult}
        onSessionChange={onSessionChange}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: '修正を確定' }));

    expect(onSessionChange).toHaveBeenCalledOnce();
    expect(onSessionChange.mock.calls[0]?.[0].latestResult).toBeNull();
    expect(calculate).not.toHaveBeenCalled();
    expect(onContinueToConditions).toHaveBeenCalledOnce();
    expect(onReturnToResult).not.toHaveBeenCalled();
  });
});
