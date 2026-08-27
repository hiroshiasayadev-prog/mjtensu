import { useEffect, type CSSProperties, type ReactNode } from 'react';

import type { RecognizedMeldGroup, RecognizedStructure, TileInstance } from '@/domain';
import {
  getYakuDisplayName,
  type FuCalculation,
  type LimitClassification,
  type ScoringCalculation,
  type ScoringPayment,
} from '@/scoring';

import {
  MobileScoringPageShell,
  PersistentBottomBar,
} from './mobile-scoring-shell';
import { formatTileIdentity, TileFace } from './tile-presentation';

export interface ResultPresentationProps {
  readonly structure: RecognizedStructure;
  readonly winningTileId: string;
  readonly calculation: ScoringCalculation;
  readonly fuDetailOpen: boolean;
  readonly onOpenFuDetail: () => void;
  readonly onCloseFuDetail: () => void;
  readonly onCorrectRecognition: () => void;
  readonly onCorrectConditions: () => void;
  readonly onFocusSeatWind: () => void;
  readonly onNewRecognition: () => void;
}

type HanBand = 'one' | 'two' | 'three-to-five' | 'six-plus' | 'yakuman';

const contentStyle: CSSProperties = {
  display: 'grid',
  gap: 12,
};

const cardStyle: CSSProperties = {
  display: 'grid',
  gap: 10,
  padding: 12,
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

const tileStripStyle: CSSProperties = {
  display: 'flex',
  gap: 3,
  alignItems: 'flex-end',
  minWidth: 0,
  overflowX: 'auto',
  padding: '2px 1px 4px',
  scrollbarWidth: 'thin',
};

const meldBlockStyle: CSSProperties = {
  display: 'inline-flex',
  gap: 2,
  flex: '0 0 auto',
  padding: 3,
  border: '1px solid #dee2e6',
  borderRadius: 6,
  background: '#f8f9fa',
};

const secondaryActionStyle: CSSProperties = {
  minHeight: 38,
  padding: '0 10px',
  border: '1px solid #adb5bd',
  borderRadius: 8,
  background: '#ffffff',
  color: '#343a40',
  fontSize: 13,
  fontWeight: 700,
  cursor: 'pointer',
};

const primaryActionStyle: CSSProperties = {
  minHeight: 42,
  padding: '0 14px',
  border: 0,
  borderRadius: 9,
  background: '#1971c2',
  color: '#ffffff',
  fontSize: 14,
  fontWeight: 800,
  cursor: 'pointer',
};

function formatPoints(points: number): string {
  return `${points.toLocaleString('ja-JP')}点`;
}

function formatLimit(limit: LimitClassification | null): string {
  if (limit === null) {
    return '通常';
  }

  switch (limit.kind) {
    case 'mangan':
      return limit.kiriage ? '切り上げ満貫' : '満貫';
    case 'haneman':
      return '跳満';
    case 'baiman':
      return '倍満';
    case 'sanbaiman':
      return '三倍満';
    case 'yakuman':
      return `${limit.counted ? '数え役満' : '役満'} ${limit.units}倍`;
  }
}

function formatPayment(payment: ScoringPayment): string {
  switch (payment.kind) {
    case 'ron':
      return `ロン 放銃者 ${formatPoints(payment.amount)}`;
    case 'tsumo-dealer':
      return `ツモ 親: 各家 ${formatPoints(payment.eachOpponent)}`;
    case 'tsumo-non-dealer':
      return `ツモ 子: 子 ${formatPoints(payment.nonDealerPays)} / 親 ${formatPoints(
        payment.dealerPays,
      )}`;
  }
}

function visibleFu(fu: FuCalculation): number {
  return fu.kind === 'chiitoitsu' ? fu.fixed : fu.rounded;
}

function winMethodLabel(calculation: ScoringCalculation): string {
  return calculation.winMethod === 'ron' ? 'ロン' : 'ツモ';
}

function hanBand(han: number): HanBand {
  if (han <= 1) {
    return 'one';
  }
  if (han === 2) {
    return 'two';
  }
  if (han <= 5) {
    return 'three-to-five';
  }
  return 'six-plus';
}

function bandAccentStyle(band: HanBand): CSSProperties {
  if (band === 'yakuman') {
    return {
      borderBottom: '4px solid transparent',
      borderImage:
        'linear-gradient(90deg, #e03131, #f08c00, #f2c94c, #2f9e44, #1971c2, #7048e8) 1',
    };
  }

  const accent = {
    one: '#1971c2',
    two: '#2b8a3e',
    'three-to-five': '#e6b800',
    'six-plus': '#c92a2a',
  }[band];

  return { borderBottom: `4px solid ${accent}` };
}

function ResultTile({
  tile,
  compact = false,
  winning = false,
}: {
  readonly tile: TileInstance;
  readonly compact?: boolean;
  readonly winning?: boolean;
}) {
  return (
    <span
      aria-label={`${formatTileIdentity(tile.tile)}${winning ? ' 和了牌' : ''}`}
      data-tile-id={tile.id}
      data-winning={winning ? 'true' : 'false'}
      style={{ display: 'inline-flex', flex: '0 0 auto' }}
    >
      <TileFace compact={compact} selected={winning} tile={tile.tile} />
    </span>
  );
}

function MeldTiles({ group }: { readonly group: RecognizedMeldGroup }) {
  return (
    <div aria-label={`${group.kind} meld`} style={meldBlockStyle}>
      {group.tiles.map((tile) => (
        <ResultTile compact key={tile.id} tile={tile} />
      ))}
    </div>
  );
}

function EvidenceTiles({
  calculation,
  structure,
  winningTileId,
}: {
  readonly calculation: ScoringCalculation;
  readonly structure: RecognizedStructure;
  readonly winningTileId: string;
}) {
  const winningTile = structure.completedHand.find((tile) => tile.id === winningTileId);
  const ordinaryTiles = structure.completedHand.filter((tile) => tile.id !== winningTileId);

  return (
    <section aria-labelledby="result-tiles-heading" style={cardStyle}>
      <h2 id="result-tiles-heading" style={cardHeadingStyle}>
        牌
      </h2>

      {structure.doraIndicators.length === 0 ? null : (
        <div
          aria-label="ドラ表示牌"
          style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}
        >
          <strong style={{ flex: '0 0 auto', fontSize: 13 }}>ドラ</strong>
          <div style={tileStripStyle}>
            {structure.doraIndicators.map((tile) => (
              <ResultTile compact key={tile.id} tile={tile} />
            ))}
          </div>
        </div>
      )}

      <div aria-label="完成手牌" style={{ display: 'grid', gap: 4 }}>
        <strong style={{ fontSize: 13 }}>手牌</strong>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: winningTile === undefined ? 'minmax(0, 1fr)' : 'minmax(0, 1fr) auto',
            gap: 8,
            alignItems: 'end',
          }}
        >
          <div style={tileStripStyle}>
            {ordinaryTiles.map((tile) => (
              <ResultTile key={tile.id} tile={tile} />
            ))}
          </div>

          {winningTile === undefined ? null : (
            <div
              aria-label="和了牌"
              style={{
                display: 'grid',
                justifyItems: 'center',
                gap: 3,
                paddingLeft: 7,
                borderLeft: '1px solid #e9ecef',
              }}
            >
              <ResultTile tile={winningTile} winning />
              <strong style={{ fontSize: 11, lineHeight: 1.1 }}>
                {winMethodLabel(calculation)}
              </strong>
            </div>
          )}
        </div>
      </div>

      {structure.meldGroups.length === 0 ? null : (
        <div aria-label="副露" style={{ display: 'grid', gap: 4 }}>
          <strong style={{ fontSize: 13 }}>副露</strong>
          <div
            style={{
              minWidth: 0,
              overflowX: 'auto',
              paddingBottom: 2,
            }}
          >
            <div
              style={{
                display: 'flex',
                gap: 5,
                width: 'max-content',
                minWidth: '100%',
                justifyContent: 'flex-end',
              }}
            >
              {structure.meldGroups.map((group, index) => (
                <MeldTiles group={group} key={`${group.kind}-${index}`} />
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

interface YakuDisplayEntry {
  readonly key: string;
  readonly name: string;
  readonly value: string;
  readonly band: HanBand;
}

function yakuDisplayEntries(calculation: ScoringCalculation): readonly YakuDisplayEntry[] {
  const entries: YakuDisplayEntry[] = calculation.yaku.map((entry) =>
    entry.kind === 'regular'
      ? {
          key: `regular-${entry.id}`,
          name: getYakuDisplayName(entry.id),
          value: `${entry.han}翻`,
          band: hanBand(entry.han),
        }
      : {
          key: `yakuman-${entry.id}`,
          name: getYakuDisplayName(entry.id),
          value: '役満',
          band: 'yakuman',
        },
  );

  if (calculation.dora.dora > 0) {
    entries.push({
      key: 'bonus-dora',
      name: 'ドラ',
      value: `${calculation.dora.dora}翻`,
      band: hanBand(calculation.dora.dora),
    });
  }
  if (calculation.dora.akaDora > 0) {
    entries.push({
      key: 'bonus-aka-dora',
      name: '赤ドラ',
      value: `${calculation.dora.akaDora}翻`,
      band: hanBand(calculation.dora.akaDora),
    });
  }

  return entries;
}

export function YakuList({
  calculation,
}: {
  readonly calculation: ScoringCalculation;
}) {
  return (
    <section aria-labelledby="result-yaku-heading" style={cardStyle}>
      <h2 id="result-yaku-heading" style={cardHeadingStyle}>
        役
      </h2>
      <div style={{ display: 'grid', gap: 5 }}>
        {yakuDisplayEntries(calculation).map((entry) => (
          <div
            data-han-band={entry.band}
            key={entry.key}
            style={{
              ...bandAccentStyle(entry.band),
              display: 'grid',
              gridTemplateColumns: 'minmax(0, 13rem) max-content',
              justifyContent: 'start',
              columnGap: 10,
              alignItems: 'end',
              maxWidth: '100%',
              padding: '3px 4px 5px',
            }}
          >
            <span
              style={{
                minWidth: 0,
                overflowWrap: 'anywhere',
                fontSize: 14,
                fontWeight: 650,
                lineHeight: 1.25,
              }}
            >
              {entry.name}
            </span>
            <strong style={{ fontSize: 13, lineHeight: 1.2, whiteSpace: 'nowrap' }}>
              {entry.value}
            </strong>
          </div>
        ))}
      </div>
    </section>
  );
}

function ScoreStat({
  label,
  value,
  action,
}: {
  readonly label: string;
  readonly value: string;
  readonly action?: ReactNode;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'baseline', gap: 5 }}>
      <span style={{ color: '#6c757d', fontSize: 11, fontWeight: 700 }}>{label}</span>
      <strong style={{ fontSize: 14 }}>{value}</strong>
      {action}
    </div>
  );
}

export function ScoreSummary({
  calculation,
  onFocusSeatWind,
  onOpenFuDetail,
}: {
  readonly calculation: ScoringCalculation;
  readonly onFocusSeatWind: () => void;
  readonly onOpenFuDetail: () => void;
}) {
  return (
    <section aria-labelledby="result-score-heading" style={cardStyle}>
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: 8,
        }}
      >
        <h2 id="result-score-heading" style={cardHeadingStyle}>
          点数
        </h2>
        <button
          aria-label="親子を修正"
          onClick={onFocusSeatWind}
          style={{ ...secondaryActionStyle, minHeight: 32, padding: '0 9px' }}
          type="button"
        >
          {calculation.winnerRole === 'dealer' ? '親' : '子'}
        </button>
      </div>

      <strong
        data-testid="result-primary-points"
        style={{
          fontSize: 'clamp(34px, 11vw, 48px)',
          fontWeight: 900,
          lineHeight: 1,
          letterSpacing: '-0.03em',
        }}
      >
        {formatPoints(calculation.totalPoints)}
      </strong>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 12px' }}>
        {calculation.fu === null ? null : (
          <ScoreStat
            action={
              calculation.limit?.kind === 'yakuman' ? undefined : (
                <button
                  onClick={onOpenFuDetail}
                  style={{
                    padding: 0,
                    border: 0,
                    background: 'transparent',
                    color: '#1971c2',
                    fontSize: 11,
                    fontWeight: 750,
                    textDecoration: 'underline',
                    cursor: 'pointer',
                  }}
                  type="button"
                >
                  符の詳細
                </button>
              )
            }
            label="符"
            value={`${visibleFu(calculation.fu)}符`}
          />
        )}
        {calculation.han === null ? null : (
          <ScoreStat label="翻" value={`${calculation.han}翻`} />
        )}
        <ScoreStat label="区分" value={formatLimit(calculation.limit)} />
      </div>

      <div
        aria-label="支払い"
        style={{
          paddingTop: 8,
          borderTop: '1px solid #e9ecef',
          fontSize: 14,
          fontWeight: 750,
          lineHeight: 1.35,
        }}
      >
        {formatPayment(calculation.payment)}
      </div>
    </section>
  );
}

function FuDetailDialog({
  fu,
  onClose,
  opened,
}: {
  readonly fu: FuCalculation | null;
  readonly onClose: () => void;
  readonly opened: boolean;
}) {
  useEffect(() => {
    if (!opened) {
      return undefined;
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        onClose();
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onClose, opened]);

  if (!opened || fu === null) {
    return null;
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 80,
        display: 'flex',
        alignItems: 'flex-end',
        justifyContent: 'center',
        background: 'rgba(20, 24, 32, 0.38)',
      }}
    >
      <section
        aria-label="符の詳細"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        style={{
          width: 'min(100%, 720px)',
          maxHeight: '75dvh',
          overflowY: 'auto',
          padding: '14px 16px calc(14px + env(safe-area-inset-bottom))',
          borderRadius: '16px 16px 0 0',
          background: '#ffffff',
          boxShadow: '0 -12px 36px rgba(20, 24, 32, 0.18)',
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 8,
            marginBottom: 12,
          }}
        >
          <h2 style={cardHeadingStyle}>符の詳細</h2>
          <button onClick={onClose} style={secondaryActionStyle} type="button">
            閉じる
          </button>
        </div>

        {fu.kind === 'standard' ? (
          <div style={{ display: 'grid', gap: 7 }}>
            <FuDetailRow label="副底" value={fu.base} />
            <FuDetailRow label="門前ロン" value={fu.menzenRon} />
            <FuDetailRow label="ツモ" value={fu.tsumo} />
            <FuDetailRow label="面子" value={fu.melds} />
            <FuDetailRow label="雀頭" value={fu.pair} />
            <FuDetailRow label="待ち" value={fu.wait} />
            <div style={{ borderTop: '1px solid #dee2e6', margin: '2px 0' }} />
            <FuDetailRow label="合計" value={fu.rawTotal} />
            <FuDetailRow emphasized label="切り上げ後" value={fu.rounded} />
          </div>
        ) : (
          <div style={{ display: 'grid', gap: 8 }}>
            <p style={{ margin: 0 }}>七対子は固定25符です。</p>
            <FuDetailRow emphasized label="最終符" value={fu.fixed} />
          </div>
        )}
      </section>
    </div>
  );
}

function FuDetailRow({
  label,
  value,
  emphasized = false,
}: {
  readonly label: string;
  readonly value: number;
  readonly emphasized?: boolean;
}) {
  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 1fr) auto',
        gap: 10,
        fontWeight: emphasized ? 800 : 500,
      }}
    >
      <span>{label}</span>
      <span>{value}符</span>
    </div>
  );
}

function ResultActions({
  onCorrectConditions,
  onCorrectRecognition,
  onNewRecognition,
}: {
  readonly onCorrectConditions: () => void;
  readonly onCorrectRecognition: () => void;
  readonly onNewRecognition: () => void;
}) {
  return (
    <PersistentBottomBar ariaLabel="結果の操作">
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
          gap: 7,
        }}
      >
        <button onClick={onCorrectRecognition} style={secondaryActionStyle} type="button">
          認識結果を修正
        </button>
        <button onClick={onCorrectConditions} style={secondaryActionStyle} type="button">
          条件を修正
        </button>
        <button
          onClick={onNewRecognition}
          style={{ ...primaryActionStyle, gridColumn: '1 / -1' }}
          type="button"
        >
          もう一度判定
        </button>
      </div>
    </PersistentBottomBar>
  );
}

export function ResultPresentation({
  calculation,
  fuDetailOpen,
  onCloseFuDetail,
  onCorrectConditions,
  onCorrectRecognition,
  onFocusSeatWind,
  onNewRecognition,
  onOpenFuDetail,
  structure,
  winningTileId,
}: ResultPresentationProps) {
  return (
    <>
      <MobileScoringPageShell
        bottomBar={
          <ResultActions
            onCorrectConditions={onCorrectConditions}
            onCorrectRecognition={onCorrectRecognition}
            onNewRecognition={onNewRecognition}
          />
        }
        bottomClearancePx={132}
        title="結果"
      >
        <div style={contentStyle}>
          <EvidenceTiles
            calculation={calculation}
            structure={structure}
            winningTileId={winningTileId}
          />
          <YakuList calculation={calculation} />
          <ScoreSummary
            calculation={calculation}
            onFocusSeatWind={onFocusSeatWind}
            onOpenFuDetail={onOpenFuDetail}
          />
        </div>
      </MobileScoringPageShell>

      <FuDetailDialog
        fu={calculation.fu}
        onClose={onCloseFuDetail}
        opened={fuDetailOpen}
      />
    </>
  );
}
