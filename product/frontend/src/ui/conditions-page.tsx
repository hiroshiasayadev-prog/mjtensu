import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

import type {
  ScoringConditionAvailability,
  ScoringConditionKey,
  ScoringConditionPolicy,
  ScoringSessionCalculation,
  ScoringSessionService,
  ScoringSessionState,
} from '@/application';
import { scoringConditionPolicy } from '@/application';
import type { RecognizedStructure, TileInstance, TileInstanceId } from '@/domain';
import {
  getYakuDisplayName,
  type RiichiState,
  type ScoringConditionsDraft,
  type ScoringInputIssue,
  type ScoringPreview,
  type ScoringRequiredField,
  type Wind,
  type WinMethod,
} from '@/scoring';

import {
  MobileScoringPageShell,
  PersistentBottomBar,
} from './mobile-scoring-shell';
import { formatTileIdentity, TileFace } from './tile-presentation';

export interface CorrectionEditorSlotProps {
  readonly session: ScoringSessionState;
  readonly commitStructure: (structure: RecognizedStructure) => void;
}

export interface ConditionsPageViewProps {
  readonly initialSession: ScoringSessionState;
  readonly sessionService: Pick<
    ScoringSessionService,
    'update' | 'preview' | 'calculate'
  >;
  readonly conditionPolicy?: ScoringConditionPolicy;
  readonly renderCorrectionEditor?: (props: CorrectionEditorSlotProps) => ReactNode;
  readonly initialFocus?: 'seatWind';
  readonly onBack?: () => void;
  readonly onCancel?: () => void;
  readonly onCalculationComplete?: (calculation: ScoringSessionCalculation) => void;
  readonly onSessionChange?: (state: ScoringSessionState) => void;
}

const WIN_METHOD_OPTIONS = [
  { value: 'tsumo', label: 'ツモ' },
  { value: 'ron', label: 'ロン' },
] as const satisfies readonly { value: WinMethod; label: string }[];

const WIND_OPTIONS = [
  { value: 'east', label: '東' },
  { value: 'south', label: '南' },
  { value: 'west', label: '西' },
  { value: 'north', label: '北' },
] as const satisfies readonly { value: Wind; label: string }[];

const RIICHI_OPTIONS = [
  { value: 'none', label: 'なし' },
  { value: 'riichi', label: 'リーチ' },
  { value: 'double-riichi', label: 'ダブルリーチ' },
] as const satisfies readonly { value: RiichiState; label: string }[];

const SECONDARY_CONDITIONS = [
  { key: 'rinshan', label: '嶺上開花' },
  { key: 'chankan', label: '槍槓' },
  { key: 'haitei', label: '海底摸月' },
  { key: 'houtei', label: '河底撈魚' },
  { key: 'tenhou', label: '天和' },
  { key: 'chiihou', label: '地和' },
] as const satisfies readonly {
  key: Exclude<ScoringConditionKey, 'ippatsu'>;
  label: string;
}[];

const contentStyle: CSSProperties = {
  display: 'grid',
  gap: 12,
};

const cardStyle: CSSProperties = {
  display: 'grid',
  gap: 12,
  padding: 14,
  border: '1px solid #e0e4e8',
  borderRadius: 12,
  background: '#ffffff',
  boxShadow: '0 1px 3px rgba(20, 24, 32, 0.04)',
};

const cardHeadingStyle: CSSProperties = {
  margin: 0,
  fontSize: 17,
  lineHeight: 1.3,
};

const tileRowStyle: CSSProperties = {
  display: 'flex',
  gap: 1,
  alignItems: 'flex-end',
  overflowX: 'auto',
  padding: '4px 1px 6px',
  scrollbarWidth: 'thin',
};

const winningTileButtonStyle: CSSProperties = {
  display: 'grid',
  justifyItems: 'center',
  gap: 2,
  flex: '0 0 auto',
  padding: '4px 0 0',
  border: 0,
  borderRadius: 6,
  background: 'transparent',
  color: '#1b1d22',
  cursor: 'pointer',
  touchAction: 'manipulation',
};

const selectedWinningTileButtonStyle: CSSProperties = {
  ...winningTileButtonStyle,
  background: '#f1f8ff',
};

const supportRowStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  alignItems: 'center',
  gap: 6,
};

const supportBlockStyle: CSSProperties = {
  display: 'inline-flex',
  gap: 3,
  padding: 4,
  border: '1px solid #e0e4e8',
  borderRadius: 6,
  background: '#f8f9fa',
};

const secondaryActionStyle: CSSProperties = {
  minHeight: 36,
  padding: '0 10px',
  border: '1px solid #adb5bd',
  borderRadius: 8,
  background: '#ffffff',
  color: '#343a40',
  fontWeight: 700,
  cursor: 'pointer',
};

const fieldsetStyle: CSSProperties = {
  border: 0,
  margin: 0,
  padding: 0,
  display: 'grid',
  gap: 8,
};

const segmentedStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 8,
};

const optionLabelStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  minHeight: 40,
  padding: '0 12px',
  border: '1px solid #c7cdd8',
  borderRadius: 8,
  background: '#ffffff',
  cursor: 'pointer',
};

const checkedOptionLabelStyle: CSSProperties = {
  ...optionLabelStyle,
  border: '1px solid #1c7ed6',
  background: '#e7f5ff',
  fontWeight: 700,
};

const checkboxGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
  gap: 8,
  marginTop: 10,
};

const dockRowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'minmax(0, 1fr) auto',
  gap: 10,
  alignItems: 'center',
};

const calculateButtonStyle: CSSProperties = {
  minWidth: 104,
  minHeight: 46,
  padding: '0 16px',
  border: 0,
  borderRadius: 10,
  background: '#1971c2',
  color: '#ffffff',
  fontWeight: 800,
  fontSize: 15,
  cursor: 'pointer',
};

export function ConditionsPageView({
  initialSession,
  sessionService,
  conditionPolicy = scoringConditionPolicy,
  renderCorrectionEditor,
  initialFocus,
  onBack,
  onCancel,
  onCalculationComplete,
  onSessionChange,
}: ConditionsPageViewProps) {
  const [session, setSession] = useState(initialSession);
  const [correctionOpen, setCorrectionOpen] = useState(false);
  const availability = conditionPolicy.availability(session.conditions);
  const preview = sessionService.preview(session);
  const canCalculate = isScoringReady(session, preview);

  function replaceSession(next: ScoringSessionState) {
    setSession(next);
    onSessionChange?.(next);
  }

  function selectWinningTile(tileId: TileInstanceId) {
    replaceSession(
      sessionService.update(session, {
        kind: 'select-winning-tile',
        tileId,
      }),
    );
  }

  function updateConditions(nextConditions: ScoringConditionsDraft) {
    replaceSession(
      sessionService.update(session, {
        kind: 'replace-conditions',
        conditions: nextConditions,
      }),
    );
  }

  function commitStructure(structure: RecognizedStructure) {
    replaceSession(
      sessionService.update(session, {
        kind: 'replace-structure',
        structure,
      }),
    );
  }

  function calculate() {
    if (!canCalculate) {
      return;
    }

    const calculation = sessionService.calculate(session);
    replaceSession(calculation.state);
    onCalculationComplete?.(calculation);
  }

  return (
    <MobileScoringPageShell
      bottomBar={
        <ConditionsYakuDock
          canCalculate={canCalculate}
          onCalculate={calculate}
          preview={preview}
        />
      }
      onBack={onBack ?? onCancel}
      title="条件入力"
    >
      <div style={contentStyle}>
        <section aria-labelledby="winning-tile-selection" style={cardStyle}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: 8,
            }}
          >
            <h2 id="winning-tile-selection" style={cardHeadingStyle}>
              牌姿確認 / 和了牌選択
            </h2>
            {renderCorrectionEditor === undefined ? null : (
              <button
                aria-expanded={correctionOpen}
                onClick={() => setCorrectionOpen((open) => !open)}
                style={secondaryActionStyle}
                type="button"
              >
                {correctionOpen ? '修正を閉じる' : '牌を修正'}
              </button>
            )}
          </div>

          <CompletedHandTiles
            onSelect={selectWinningTile}
            tiles={session.structure.completedHand}
            winningTileId={session.winningTileId}
          />
          <SupportingStructure structure={session.structure} />

          {renderCorrectionEditor === undefined || !correctionOpen ? null : (
            <div
              aria-label="牌を修正"
              style={{
                paddingTop: 12,
                borderTop: '1px solid #e9ecef',
              }}
            >
              {renderCorrectionEditor({ session, commitStructure })}
            </div>
          )}
        </section>

        <ConditionControls
          availability={availability}
          conditions={session.conditions}
          initialFocus={initialFocus}
          onChange={updateConditions}
        />
      </div>
    </MobileScoringPageShell>
  );
}

function CompletedHandTiles({
  tiles,
  winningTileId,
  onSelect,
}: {
  readonly tiles: readonly TileInstance[];
  readonly winningTileId: TileInstanceId;
  readonly onSelect: (tileId: TileInstanceId) => void;
}) {
  return (
    <div aria-label="和了牌選択" role="group" style={tileRowStyle}>
      {tiles.map((tile, index) => {
        const selected = tile.id === winningTileId;
        const label = formatTileIdentity(tile.tile);

        return (
          <button
            aria-label={`${label} ${index + 1}${selected ? ' 和了牌' : ''}`}
            aria-pressed={selected}
            key={tile.id}
            onClick={() => onSelect(tile.id)}
            style={selected ? selectedWinningTileButtonStyle : winningTileButtonStyle}
            type="button"
          >
            <TileFace selected={selected} tile={tile.tile} />
            <span style={{ minHeight: 12, fontSize: 10, fontWeight: 700 }}>
              {selected ? '和了' : ''}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function SupportingStructure({
  structure,
}: {
  readonly structure: RecognizedStructure;
}) {
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      {structure.doraIndicators.length === 0 ? null : (
        <div style={supportRowStyle}>
          <strong style={{ fontSize: 13 }}>ドラ</strong>
          {structure.doraIndicators.map((tile, index) => (
            <span
              aria-label={`ドラ表示牌 ${index + 1} ${formatTileIdentity(tile.tile)}`}
              key={tile.id}
            >
              <TileFace compact tile={tile.tile} />
            </span>
          ))}
        </div>
      )}

      {structure.meldGroups.length === 0 ? null : (
        <div style={supportRowStyle}>
          <strong style={{ fontSize: 13 }}>副露</strong>
          {structure.meldGroups.map((meld, index) => (
            <div
              aria-label={`${meldKindLabel(meld.kind)} ${index + 1}`}
              key={`${meld.kind}-${index}`}
              style={supportBlockStyle}
            >
              <span style={{ alignSelf: 'center', fontSize: 9, fontWeight: 700 }}>
                {meldKindLabel(meld.kind)}
              </span>
              {meld.tiles.map((tile) => (
                <TileFace compact key={tile.id} tile={tile.tile} />
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ConditionControls({
  conditions,
  availability,
  initialFocus,
  onChange,
}: {
  readonly conditions: ScoringConditionsDraft;
  readonly availability: ScoringConditionAvailability;
  readonly initialFocus?: 'seatWind';
  readonly onChange: (conditions: ScoringConditionsDraft) => void;
}) {
  function patch(patchConditions: Partial<ScoringConditionsDraft>) {
    onChange({ ...conditions, ...patchConditions });
  }

  return (
    <>
      <section aria-labelledby="ordinary-conditions" style={cardStyle}>
        <h2 id="ordinary-conditions" style={cardHeadingStyle}>
          基本条件
        </h2>
        <RadioButtonSet
          label="和了方法"
          name="win-method"
          onChange={(winMethod) => patch({ winMethod })}
          options={WIN_METHOD_OPTIONS}
          value={conditions.winMethod}
        />
        <RadioButtonSet
          label="場風"
          name="round-wind"
          onChange={(roundWind) => patch({ roundWind })}
          options={WIND_OPTIONS}
          value={conditions.roundWind}
        />
        <RadioButtonSet
          focusOnMount={initialFocus === 'seatWind'}
          label="自風"
          name="seat-wind"
          onChange={(seatWind) => patch({ seatWind })}
          options={WIND_OPTIONS}
          value={conditions.seatWind}
        />
        <RadioButtonSet
          label="リーチ"
          name="riichi"
          onChange={(riichi) => patch({ riichi })}
          options={RIICHI_OPTIONS}
          value={conditions.riichi}
        />
        <CheckboxControl
          checked={conditions.ippatsu}
          disabled={!availability.ippatsu}
          label="一発"
          onChange={(ippatsu) => patch({ ippatsu })}
        />
      </section>

      <section aria-labelledby="secondary-conditions" style={cardStyle}>
        <details>
          <summary id="secondary-conditions" style={{ cursor: 'pointer', fontWeight: 800 }}>
            その他の条件
          </summary>
          <div style={checkboxGridStyle}>
            {SECONDARY_CONDITIONS.map(({ key, label }) => (
              <CheckboxControl
                checked={conditions[key]}
                disabled={!availability[key]}
                key={key}
                label={label}
                onChange={(selected) => patch({ [key]: selected })}
              />
            ))}
          </div>
        </details>
      </section>
    </>
  );
}

function RadioButtonSet<TValue extends string>({
  label,
  name,
  options,
  value,
  focusOnMount = false,
  onChange,
}: {
  readonly label: string;
  readonly name: string;
  readonly options: readonly { readonly value: TValue; readonly label: string }[];
  readonly value: TValue | null;
  readonly focusOnMount?: boolean;
  readonly onChange: (value: TValue) => void;
}) {
  const fieldsetRef = useRef<HTMLFieldSetElement>(null);

  useEffect(() => {
    if (focusOnMount) {
      fieldsetRef.current?.focus();
    }
  }, [focusOnMount]);

  return (
    <fieldset
      data-edit-focus={focusOnMount ? 'true' : undefined}
      ref={fieldsetRef}
      style={
        focusOnMount
          ? {
              ...fieldsetStyle,
              outline: '3px solid #74c0fc',
              outlineOffset: 2,
              borderRadius: 6,
              padding: 6,
            }
          : fieldsetStyle
      }
      tabIndex={focusOnMount ? -1 : undefined}
    >
      <legend>{label}</legend>
      <div style={segmentedStyle}>
        {options.map((option) => (
          <label
            key={option.value}
            style={
              option.value === value ? checkedOptionLabelStyle : optionLabelStyle
            }
          >
            <input
              checked={option.value === value}
              name={name}
              onChange={() => onChange(option.value)}
              style={{ marginRight: 6 }}
              type="radio"
            />
            {option.label}
          </label>
        ))}
      </div>
    </fieldset>
  );
}

function CheckboxControl({
  label,
  checked,
  disabled,
  onChange,
}: {
  readonly label: string;
  readonly checked: boolean;
  readonly disabled: boolean;
  readonly onChange: (checked: boolean) => void;
}) {
  return (
    <label
      style={{
        ...optionLabelStyle,
        ...(checked ? checkedOptionLabelStyle : {}),
        cursor: disabled ? 'not-allowed' : 'pointer',
        opacity: disabled ? 0.45 : 1,
      }}
    >
      <input
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.currentTarget.checked)}
        style={{ marginRight: 6 }}
        type="checkbox"
      />
      {label}
    </label>
  );
}

function ConditionsYakuDock({
  preview,
  canCalculate,
  onCalculate,
}: {
  readonly preview: ScoringPreview;
  readonly canCalculate: boolean;
  readonly onCalculate: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const readyYaku = preview.kind === 'ready' ? preview.yaku : [];
  const canExpand = readyYaku.length > 2;

  useEffect(() => {
    if (!canExpand) {
      setExpanded(false);
    }
  }, [canExpand]);

  return (
    <PersistentBottomBar ariaLabel="現在の役と計算">
      <div data-preview-state={preview.kind} style={{ display: 'grid', gap: 8 }}>
        {expanded && preview.kind === 'ready' ? (
          <div
            aria-label="現在の役一覧"
            style={{
              maxHeight: '32dvh',
              overflowY: 'auto',
              paddingBottom: 8,
              borderBottom: '1px solid #e9ecef',
            }}
          >
            <YakuEntries preview={preview} />
          </div>
        ) : null}

        <div style={dockRowStyle}>
          <div style={{ minWidth: 0 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <strong style={{ fontSize: 13 }}>現在の役</strong>
              {canExpand ? (
                <button
                  aria-expanded={expanded}
                  onClick={() => setExpanded((open) => !open)}
                  style={{
                    padding: 0,
                    border: 0,
                    background: 'transparent',
                    color: '#1971c2',
                    fontSize: 12,
                    fontWeight: 700,
                    cursor: 'pointer',
                  }}
                  type="button"
                >
                  {expanded ? '閉じる' : 'すべて表示'}
                </button>
              ) : null}
            </div>
            <div
              aria-live="polite"
              role="status"
              style={{
                marginTop: 3,
                overflow: 'hidden',
                color: preview.kind === 'ready' ? '#212529' : '#5c6770',
                fontSize: 13,
                fontWeight: 600,
                lineHeight: 1.35,
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
              }}
            >
              {previewSummary(preview)}
            </div>
          </div>

          <button
            disabled={!canCalculate}
            onClick={onCalculate}
            style={{
              ...calculateButtonStyle,
              cursor: canCalculate ? 'pointer' : 'not-allowed',
              opacity: canCalculate ? 1 : 0.45,
            }}
            type="button"
          >
            計算する
          </button>
        </div>
      </div>
    </PersistentBottomBar>
  );
}

function YakuEntries({ preview }: { readonly preview: Extract<ScoringPreview, { kind: 'ready' }> }) {
  return (
    <ul style={{ display: 'grid', gap: 6, margin: 0, paddingLeft: 20 }}>
      {preview.yaku.map((entry) => (
        <li key={entry.id}>
          {getYakuDisplayName(entry.id)}
          {entry.kind === 'regular' ? ` ${entry.han}翻` : ' 役満'}
        </li>
      ))}
    </ul>
  );
}

function previewSummary(preview: ScoringPreview): string {
  switch (preview.kind) {
    case 'incomplete':
      return `未入力: ${preview.missing.map(requiredFieldLabel).join('、')}`;
    case 'invalid-input':
      return `入力の組み合わせを確認してください:${preview.issues.map(inputIssueLabel).join('、')}`;
    case 'invalid-winning-shape':
      return '和了形として成立していません';
    case 'no-yaku':
      return '役なし';
    case 'ready': {
      if (preview.yaku.length === 0) {
        return '役なし';
      }
      const visible = preview.yaku.slice(0, 2).map((entry) =>
        `${getYakuDisplayName(entry.id)}${entry.kind === 'regular' ? ` ${entry.han}翻` : ' 役満'}`,
      );
      const remaining = preview.yaku.length - visible.length;
      return `${visible.join(' / ')}${remaining > 0 ? ` / ほか${remaining}件` : ''}`;
    }
  }
}

function isScoringReady(
  session: ScoringSessionState,
  preview: ScoringPreview,
): boolean {
  return (
    preview.kind === 'ready' &&
    preview.yaku.length > 0 &&
    session.conditions.winMethod !== null &&
    session.conditions.roundWind !== null &&
    session.conditions.seatWind !== null &&
    session.structure.completedHand.some((tile) => tile.id === session.winningTileId)
  );
}

function meldKindLabel(kind: RecognizedStructure['meldGroups'][number]['kind']): string {
  switch (kind) {
    case 'chi':
      return 'チー';
    case 'pon':
      return 'ポン';
    case 'open-kan':
      return '明槓';
    case 'concealed-kan':
      return '暗槓';
    case 'unresolved':
      return '未解決';
  }
}

function requiredFieldLabel(field: ScoringRequiredField): string {
  switch (field) {
    case 'win-method':
      return '和了方法';
    case 'round-wind':
      return '場風';
    case 'seat-wind':
      return '自風';
  }
}

function inputIssueLabel(issue: ScoringInputIssue): string {
  switch (issue.kind) {
    case 'winning-tile-not-in-completed-hand':
      return '和了牌';
    case 'unresolved-meld':
    case 'invalid-meld':
      return `副露 ${issue.meldIndex + 1}`;
    case 'invalid-structure':
      return '牌姿';
    case 'contradictory-conditions':
      return '矛盾する条件';
  }
}
