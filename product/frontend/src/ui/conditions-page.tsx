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
import type {
  RiichiState,
  ScoringConditionsDraft,
  ScoringInputIssue,
  ScoringPreview,
  ScoringRequiredField,
  Wind,
  WinMethod,
  YakuId,
} from '@/scoring';

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

const YAKU_LABELS: Record<YakuId, string> = {
  riichi: 'リーチ',
  'double-riichi': 'ダブルリーチ',
  ippatsu: '一発',
  'menzen-tsumo': '門前清自摸和',
  tanyao: '断么九',
  pinfu: '平和',
  iipeikou: '一盃口',
  'yakuhai-east': '役牌 東',
  'yakuhai-south': '役牌 南',
  'yakuhai-west': '役牌 西',
  'yakuhai-north': '役牌 北',
  'yakuhai-white': '役牌 白',
  'yakuhai-green': '役牌 發',
  'yakuhai-red': '役牌 中',
  'rinshan-kaihou': '嶺上開花',
  chankan: '槍槓',
  haitei: '海底摸月',
  houtei: '河底撈魚',
  toitoi: '対々和',
  'sanshoku-doujun': '三色同順',
  'sanshoku-doukou': '三色同刻',
  ittsu: '一気通貫',
  chiitoitsu: '七対子',
  chanta: '混全帯么九',
  sanankou: '三暗刻',
  sankantsu: '三槓子',
  honroutou: '混老頭',
  shousangen: '小三元',
  honitsu: '混一色',
  junchan: '純全帯么九',
  ryanpeikou: '二盃口',
  chinitsu: '清一色',
  tenhou: '天和',
  chiihou: '地和',
  'kokushi-musou': '国士無双',
  'kokushi-13-wait': '国士無双十三面待ち',
  suuankou: '四暗刻',
  'suuankou-tanki': '四暗刻単騎',
  daisangen: '大三元',
  shousuushii: '小四喜',
  daisuushii: '大四喜',
  tsuuiisou: '字一色',
  chinroutou: '清老頭',
  ryuuiisou: '緑一色',
  'chuuren-poutou': '九蓮宝燈',
  'junsei-chuuren-poutou': '純正九蓮宝燈',
  suukantsu: '四槓子',
};

const pageStyle: CSSProperties = {
  display: 'grid',
  gap: 24,
  maxWidth: 960,
  margin: '0 auto',
  padding: 16,
};

const sectionStyle: CSSProperties = {
  display: 'grid',
  gap: 12,
};

const tileRowStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 8,
  alignItems: 'center',
};

const tileButtonStyle: CSSProperties = {
  minWidth: 42,
  minHeight: 56,
  border: '1px solid #c7cdd8',
  borderRadius: 6,
  background: '#fffaf0',
  color: '#1b1d22',
  fontWeight: 700,
};

const selectedTileStyle: CSSProperties = {
  ...tileButtonStyle,
  border: '3px solid #1c7ed6',
  background: '#e7f5ff',
  transform: 'translateY(-4px)',
};

const supportTileStyle: CSSProperties = {
  ...tileButtonStyle,
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  minWidth: 34,
  minHeight: 46,
  fontSize: 12,
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
  minHeight: 36,
  padding: '0 12px',
  border: '1px solid #c7cdd8',
  borderRadius: 6,
  background: '#ffffff',
  cursor: 'pointer',
};

const checkedOptionLabelStyle: CSSProperties = {
  ...optionLabelStyle,
  borderColor: '#1c7ed6',
  background: '#e7f5ff',
  fontWeight: 700,
};

const checkboxGridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
  gap: 8,
};

const previewStyle: CSSProperties = {
  borderLeft: '4px solid #1c7ed6',
  padding: '8px 12px',
  background: '#f8f9fa',
};

export function ConditionsPageView({
  initialSession,
  sessionService,
  conditionPolicy = scoringConditionPolicy,
  renderCorrectionEditor,
  initialFocus,
  onCancel,
  onCalculationComplete,
  onSessionChange,
}: ConditionsPageViewProps) {
  const [session, setSession] = useState(initialSession);
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
    <main style={pageStyle}>
      <header>
        <h1>条件入力</h1>
        <p>現在の和了牌: {session.winningTileId}</p>
      </header>

      <section aria-labelledby="recognized-structure" style={sectionStyle}>
        <h2 id="recognized-structure">認識牌姿</h2>
        <CompletedHandTiles
          onSelect={selectWinningTile}
          tiles={session.structure.completedHand}
          winningTileId={session.winningTileId}
        />
        <SupportingStructure structure={session.structure} />
      </section>

      {renderCorrectionEditor === undefined ? null : (
        <section aria-labelledby="structure-correction" style={sectionStyle}>
          <h2 id="structure-correction">牌姿修正</h2>
          {renderCorrectionEditor({ session, commitStructure })}
        </section>
      )}

      <ConditionControls
        availability={availability}
        conditions={session.conditions}
        initialFocus={initialFocus}
        onChange={updateConditions}
      />

      <section aria-labelledby="scoring-preview" style={sectionStyle}>
        <h2 id="scoring-preview">現在の役</h2>
        <ScoringPreviewPanel preview={preview} />
      </section>

      <footer style={{ display: 'flex', gap: 8 }}>
        {onCancel === undefined ? null : (
          <button onClick={onCancel} type="button">
            キャンセル
          </button>
        )}
        <button disabled={!canCalculate} onClick={calculate} type="button">
          計算する
        </button>
      </footer>
    </main>
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

        return (
          <button
            aria-pressed={selected}
            key={tile.id}
            onClick={() => onSelect(tile.id)}
            style={selected ? selectedTileStyle : tileButtonStyle}
            type="button"
          >
            <span>{tileLabel(tile)}</span>
            <span style={{ display: 'block', fontSize: 10 }}>
              {selected ? '和了' : `${index + 1}`}
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
    <div style={{ display: 'grid', gap: 10 }}>
      {structure.meldGroups.length === 0 ? null : (
        <div>
          <strong>副露</strong>
          <div style={tileRowStyle}>
            {structure.meldGroups.map((meld, index) => (
              <div
                aria-label={`${meldKindLabel(meld.kind)} ${index + 1}`}
                key={`${meld.kind}-${index}`}
                style={{
                  display: 'inline-flex',
                  gap: 4,
                  padding: 6,
                  border: '1px solid #d8dee9',
                  borderRadius: 6,
                }}
              >
                {meld.tiles.map((tile) => (
                  <span key={tile.id} style={supportTileStyle}>
                    {tileLabel(tile)}
                  </span>
                ))}
              </div>
            ))}
          </div>
        </div>
      )}
      {structure.doraIndicators.length === 0 ? null : (
        <div>
          <strong>ドラ表示牌</strong>
          <div style={tileRowStyle}>
            {structure.doraIndicators.map((tile, index) => (
              <span
                aria-label={`ドラ表示牌 ${index + 1} ${tileLabel(tile)}`}
                key={tile.id}
                style={supportTileStyle}
              >
                {tileLabel(tile)}
              </span>
            ))}
          </div>
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
      <section aria-labelledby="ordinary-conditions" style={sectionStyle}>
        <h2 id="ordinary-conditions">基本条件</h2>
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

      <section aria-labelledby="secondary-conditions" style={sectionStyle}>
        <details>
          <summary id="secondary-conditions">その他の条件</summary>
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

function ScoringPreviewPanel({ preview }: { readonly preview: ScoringPreview }) {
  switch (preview.kind) {
    case 'incomplete':
      return (
        <div role="status" style={previewStyle}>
          未入力: {preview.missing.map(requiredFieldLabel).join('、')}
        </div>
      );

    case 'invalid-input':
      return (
        <div role="status" style={previewStyle}>
          入力の組み合わせを確認してください:
          {preview.issues.map(inputIssueLabel).join('、')}
        </div>
      );

    case 'invalid-winning-shape':
      return (
        <div role="status" style={previewStyle}>
          和了形として成立していません
        </div>
      );

    case 'no-yaku':
      return (
        <div role="status" style={previewStyle}>
          役なし
        </div>
      );

    case 'ready':
      return (
        <div role="status" style={previewStyle}>
          <ul>
            {preview.yaku.map((entry) => (
              <li key={entry.id}>
                {YAKU_LABELS[entry.id]}
                {entry.kind === 'regular' ? ` ${entry.han}翻` : ''}
              </li>
            ))}
          </ul>
        </div>
      );
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

function tileLabel(tile: TileInstance): string {
  return `${tile.tile.red ? '赤' : ''}${tile.tile.kind}`;
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
