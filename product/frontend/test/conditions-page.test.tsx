import { fireEvent, render, screen } from '@testing-library/react';
import type { ComponentProps } from 'react';
import {
  createScoringSessionService,
  type ScoringSessionCalculation,
  type ScoringSessionState,
} from '@/application';
import type { RecognizedStructure, TileInstance, TileInstanceId } from '@/domain';
import {
  DEFAULT_RULE_PROFILE,
  type ScoringCalculation,
  type ScoringPreview,
  type ScoringService,
} from '@/scoring';
import { ConditionsPageView } from '@/ui/conditions-page';
import { describe, expect, it, vi } from 'vitest';

import { createFakeService } from './support';

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

function tileId(value: string): TileInstanceId {
  return value as TileInstanceId;
}

function tile(
  id: string,
  kind: TileInstance['tile']['kind'] = '5m',
  red = false,
): TileInstance {
  return {
    id: tileId(id),
    tile: { kind, red },
  };
}

function structure(
  completedHand: readonly TileInstance[],
  overrides: Partial<RecognizedStructure> = {},
): RecognizedStructure {
  return {
    completedHand,
    doraIndicators: [],
    meldGroups: [],
    ...overrides,
  };
}

function fakeScoringService(
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

interface RenderConditionsPageOptions {
  readonly preview?: ScoringPreview;
  readonly completedHand?: readonly TileInstance[];
  readonly initialFocus?: ComponentProps<typeof ConditionsPageView>['initialFocus'];
  readonly onCancel?: () => void;
  readonly onCalculationComplete?: (calculation: ScoringSessionCalculation) => void;
  readonly onSessionChange?: (state: ScoringSessionState) => void;
  readonly renderCorrectionEditor?: ComponentProps<
    typeof ConditionsPageView
  >['renderCorrectionEditor'];
  readonly scoringOverrides?: Partial<ScoringService>;
}

function renderConditionsPage(options: RenderConditionsPageOptions = {}) {
  const {
    preview = readyPreview,
    completedHand = [tile('left'), tile('first-five'), tile('second-five')],
    initialFocus,
    onCancel,
    onCalculationComplete = vi.fn(),
    onSessionChange = vi.fn(),
    renderCorrectionEditor,
    scoringOverrides = {},
  } = options;
  const sessionService = createScoringSessionService(
    fakeScoringService(preview, scoringOverrides),
  );
  const initialSession = sessionService.create(
    structure(completedHand, {
      doraIndicators: [tile('dora', '1z')],
      meldGroups: [
        {
          kind: 'chi',
          tiles: [tile('meld-1', '2m'), tile('meld-2', '3m'), tile('meld-3', '4m')],
        },
      ],
    }),
    DEFAULT_RULE_PROFILE,
  );

  render(
    <ConditionsPageView
      initialFocus={initialFocus}
      initialSession={initialSession}
      onCancel={onCancel}
      onCalculationComplete={onCalculationComplete}
      onSessionChange={onSessionChange}
      renderCorrectionEditor={renderCorrectionEditor}
      sessionService={sessionService}
    />,
  );

  return { initialSession, onCalculationComplete, onSessionChange, sessionService };
}

describe('ConditionsPageView', () => {
  it('selects duplicate tile kinds by completed-hand tile instance', () => {
    const onSessionChange = vi.fn();
    renderConditionsPage({ onSessionChange });

    const duplicateButtons = screen.getAllByRole('button', { name: /5萬/ });

    expect(duplicateButtons.at(-1)).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(duplicateButtons[1]);

    expect(onSessionChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ winningTileId: tileId('first-five') }),
    );
    expect(duplicateButtons[1]).toHaveAttribute('aria-pressed', 'true');
  });

  it('preserves red-five identity in the selectable completed hand', () => {
    renderConditionsPage({
      completedHand: [tile('ordinary-five', '5m'), tile('red-five', '5m', true)],
    });

    const ordinary = screen.getByRole('button', { name: '5萬 1' });
    const red = screen.getByRole('button', { name: '赤5萬 2 和了牌' });

    expect(ordinary).toBeVisible();
    expect(red).toBeVisible();
    expect(
      ordinary.querySelector('[data-tile-asset]')?.getAttribute('data-tile-asset'),
    ).toContain('5m.svg');
    expect(
      red.querySelector('[data-tile-asset]')?.getAttribute('data-tile-asset'),
    ).toContain('5m-red.svg');
  });

  it('renders hand, meld, and dora structure while keeping meld tiles out of winning selection', () => {
    renderConditionsPage();

    expect(screen.getByRole('group', { name: '和了牌選択' })).toBeVisible();
    expect(screen.getByLabelText('チー 1')).toBeVisible();
    expect(screen.getByLabelText('ドラ表示牌 1 東')).toBeVisible();
    expect(screen.queryByRole('button', { name: /2萬/ })).not.toBeInTheDocument();
  });

  it('uses policy-normalized state to clear dependent selections after ordinary condition changes', () => {
    renderConditionsPage();

    fireEvent.click(screen.getByText('その他の条件'));
    const rinshan = screen.getByLabelText('嶺上開花');
    fireEvent.click(rinshan);

    expect(rinshan).toBeChecked();

    fireEvent.click(screen.getByLabelText('ロン'));

    expect(screen.getByLabelText('嶺上開花')).not.toBeChecked();
    expect(screen.getByLabelText('嶺上開花')).toBeDisabled();
    expect(screen.getByLabelText('槍槓')).not.toBeDisabled();
  });

  it('enables ippatsu only through policy availability', () => {
    renderConditionsPage();

    expect(screen.getByLabelText('一発')).toBeDisabled();

    fireEvent.click(screen.getByLabelText('リーチ'));

    expect(screen.getByLabelText('一発')).not.toBeDisabled();

    fireEvent.click(screen.getByLabelText('なし'));

    expect(screen.getByLabelText('一発')).toBeDisabled();
    expect(screen.getByLabelText('一発')).not.toBeChecked();
  });

  it.each([
    {
      preview: { kind: 'invalid-winning-shape' } as const,
      text: '和了形として成立していません',
    },
    {
      preview: {
        kind: 'invalid-input',
        issues: [{ kind: 'contradictory-conditions' }],
      } as const,
      text: '入力の組み合わせを確認してください:矛盾する条件',
    },
    {
      preview: { kind: 'no-yaku' } as const,
      text: '役なし',
    },
    {
      preview: {
        kind: 'incomplete',
        missing: ['seat-wind'],
      } as const,
      text: '未入力: 自風',
    },
  ])('renders $preview.kind as a non-ready preview state', ({ preview, text }) => {
    const onCalculationComplete = vi.fn();

    renderConditionsPage({ onCalculationComplete, preview });

    expect(screen.getByRole('status')).toHaveTextContent(text);
    expect(screen.getByRole('button', { name: '計算する' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '計算する' }));
    expect(onCalculationComplete).not.toHaveBeenCalled();
  });

  it('shows awarded yaku and calculates only when preview is scoring-ready', () => {
    const onCalculationComplete = vi.fn();
    renderConditionsPage({ onCalculationComplete });

    expect(screen.getByRole('status')).toHaveTextContent('門前清自摸和 1翻');

    fireEvent.click(screen.getByRole('button', { name: '計算する' }));

    expect(onCalculationComplete).toHaveBeenCalledWith(
      expect.objectContaining({ result: calculation }),
    );
  });

  it('does not treat ready-without-yaku as calculation-ready', () => {
    const onCalculationComplete = vi.fn();
    renderConditionsPage({
      onCalculationComplete,
      preview: { kind: 'ready', yaku: [] },
    });

    expect(screen.getByRole('button', { name: '計算する' })).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '計算する' }));
    expect(onCalculationComplete).not.toHaveBeenCalled();
  });

  it('focuses and emphasizes seat wind when entered through the Result shortcut', () => {
    const onCancel = vi.fn();
    renderConditionsPage({ initialFocus: 'seatWind', onCancel });

    const seatWind = screen.getByRole('group', { name: '自風' });
    expect(seatWind).toHaveFocus();
    expect(seatWind).toHaveAttribute('data-edit-focus', 'true');

    fireEvent.click(screen.getByRole('button', { name: '戻る' }));
    expect(onCancel).toHaveBeenCalledOnce();
  });

  it('uses the mobile scoring shell without exposing tile instance ids or canonical tile codes', () => {
    renderConditionsPage();

    expect(screen.getByTestId('mobile-scoring-app-bar')).toHaveTextContent('条件入力');
    expect(screen.getByTestId('persistent-bottom-bar')).toBeVisible();
    expect(screen.getByTestId('mobile-scoring-app-bar')).toHaveAttribute(
      'data-safe-area-top',
      'true',
    );
    expect(screen.getByTestId('mobile-scoring-scroll-content')).toHaveAttribute(
      'data-bottom-clearance-px',
      '112',
    );
    expect(screen.getByTestId('persistent-bottom-bar')).toHaveAttribute(
      'data-safe-area-bottom',
      'true',
    );
    expect(document.body.textContent).not.toContain('first-five');
    expect(document.body.textContent).not.toContain('second-five');
    expect(document.body.textContent).not.toContain('1z');
    expect(screen.getByRole('heading', { name: '和了牌を選択' })).toBeVisible();
  });

  it('updates persistent yaku feedback and calculation availability after condition changes', () => {
    renderConditionsPage({
      preview: { kind: 'no-yaku' },
      scoringOverrides: {
        preview: (input) =>
          input.conditions.riichi === 'riichi'
            ? {
                kind: 'ready',
                yaku: [{ kind: 'regular', id: 'riichi', han: 1 }],
              }
            : { kind: 'no-yaku' },
      },
    });

    expect(screen.getByRole('status')).toHaveTextContent('役なし');
    expect(screen.getByRole('button', { name: '計算する' })).toBeDisabled();

    fireEvent.click(screen.getByLabelText('リーチ'));

    expect(screen.getByRole('status')).toHaveTextContent('立直 1翻');
    expect(screen.getByRole('button', { name: '計算する' })).toBeEnabled();
  });

  it('expands the full current-yaku list inside the persistent dock', () => {
    renderConditionsPage({
      preview: {
        kind: 'ready',
        yaku: [
          { kind: 'regular', id: 'riichi', han: 1 },
          { kind: 'regular', id: 'tanyao', han: 1 },
          { kind: 'regular', id: 'pinfu', han: 1 },
        ],
      },
    });

    expect(screen.queryByLabelText('現在の役一覧')).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'すべて表示' }));

    const list = screen.getByLabelText('現在の役一覧');
    expect(list).toHaveTextContent('立直 1翻');
    expect(list).toHaveTextContent('断么九 1翻');
    expect(list).toHaveTextContent('平和 1翻');
  });

  it('keeps structure correction secondary until explicitly opened and commits through session replacement', () => {
    const onSessionChange = vi.fn();
    const replacement = structure([tile('replacement-left'), tile('replacement-right')]);

    renderConditionsPage({
      onSessionChange,
      renderCorrectionEditor: ({ commitStructure }) => (
        <button onClick={() => commitStructure(replacement)} type="button">
          修正を確定
        </button>
      ),
    });

    expect(screen.queryByRole('button', { name: '修正を確定' })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: '牌を修正' }));
    fireEvent.click(screen.getByRole('button', { name: '修正を確定' }));

    expect(onSessionChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        structure: replacement,
        winningTileId: tileId('replacement-right'),
      }),
    );
  });
});
