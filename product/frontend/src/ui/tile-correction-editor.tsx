import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';

import type {
  CorrectionCommand,
  CorrectionDestination,
  CorrectionDraft,
  CorrectionEditorService,
  CorrectionIssue,
} from '@/application';
import type {
  RecognizedStructure,
  TileIdentity,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';

import { formatTileIdentity, TileFace } from './tile-presentation';

export interface TileCorrectionEditorProps {
  readonly initialStructure: RecognizedStructure;
  readonly service: CorrectionEditorService;
  readonly onCommit: (structure: RecognizedStructure) => void;
  readonly primaryActionLabel?: string;
  readonly autoCommitValidChanges?: boolean;
}

type SelectorState =
  | {
      readonly kind: 'insert';
      readonly destination: CorrectionDestination;
      readonly index?: number;
    }
  | {
      readonly kind: 'edit';
      readonly tileId: TileInstanceId;
    };

type DraftTileLocation = {
  readonly destination: CorrectionDestination;
  readonly index: number;
};

const TILE_KEYBOARD_ROWS: readonly {
  readonly label: string;
  readonly tiles: readonly TileIdentity[];
}[] = [
  {
    label: '萬子',
    tiles: keyboardSuitRow('m'),
  },
  {
    label: '筒子',
    tiles: keyboardSuitRow('p'),
  },
  {
    label: '索子',
    tiles: keyboardSuitRow('s'),
  },
  {
    label: '字牌',
    tiles: ['1z', '2z', '3z', '4z', '5z', '6z', '7z'].map(
      (kind) => ({ kind: kind as TileKind, red: false }),
    ),
  },
];

const editorStyle: CSSProperties = {
  display: 'grid',
  gap: 16,
};

const structureStyle: CSSProperties = {
  display: 'grid',
  gap: 14,
  padding: 10,
  border: '1px solid transparent',
  borderRadius: 8,
};

const regionStyle: CSSProperties = {
  display: 'grid',
  gap: 8,
  padding: 10,
  border: '1px solid #d8dee9',
  borderRadius: 8,
  background: '#ffffff',
};

const invalidRegionStyle: CSSProperties = {
  ...regionStyle,
  border: '2px solid #c92a2a',
};

const tileRowStyle: CSSProperties = {
  display: 'flex',
  flexWrap: 'wrap',
  gap: 6,
  alignItems: 'center',
};

const tileButtonStyle: CSSProperties = {
  minWidth: 44,
  minHeight: 56,
  padding: '4px 6px',
  border: '1px solid #adb5bd',
  borderRadius: 6,
  background: '#fffaf0',
  color: '#1b1d22',
  fontWeight: 700,
  cursor: 'pointer',
};

const addButtonStyle: CSSProperties = {
  minWidth: 44,
  minHeight: 44,
  border: '1px dashed #868e96',
  borderRadius: 6,
  background: '#f8f9fa',
  cursor: 'pointer',
};

const actionButtonStyle: CSSProperties = {
  minHeight: 38,
  padding: '0 12px',
  border: '1px solid #adb5bd',
  borderRadius: 6,
  background: '#ffffff',
  cursor: 'pointer',
};

const issueStyle: CSSProperties = {
  margin: 0,
  color: '#a61e1e',
  fontSize: 14,
  fontWeight: 600,
};

const selectorBackdropStyle: CSSProperties = {
  position: 'fixed',
  inset: 0,
  zIndex: 100,
  display: 'flex',
  alignItems: 'flex-end',
  background: 'rgba(0, 0, 0, 0.4)',
};

const selectorStyle: CSSProperties = {
  width: '100%',
  maxHeight: '78vh',
  overflowY: 'auto',
  display: 'grid',
  gap: 14,
  padding: 16,
  borderRadius: '14px 14px 0 0',
  background: '#ffffff',
};

const tileKeyboardStyle: CSSProperties = {
  display: 'grid',
  gap: 10,
};

const tileKeyboardRowStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(10, minmax(0, 1fr))',
  gap: 3,
  alignItems: 'center',
};

const selectorTileButtonStyle: CSSProperties = {
  minWidth: 0,
  minHeight: 0,
  padding: 0,
  border: 0,
  borderRadius: 4,
  background: 'transparent',
  cursor: 'pointer',
  display: 'grid',
  placeItems: 'center',
  touchAction: 'manipulation',
};

export function TileCorrectionEditor({
  initialStructure,
  service,
  onCommit,
  primaryActionLabel = '修正を確定',
  autoCommitValidChanges = false,
}: TileCorrectionEditorProps) {
  const [draft, setDraft] = useState(() => service.create(initialStructure));
  const [selector, setSelector] = useState<SelectorState | null>(null);
  const previousInputs = useRef({ initialStructure, service });

  useEffect(() => {
    if (
      previousInputs.current.initialStructure === initialStructure &&
      previousInputs.current.service === service
    ) {
      return;
    }

    previousInputs.current = { initialStructure, service };
    setDraft(service.create(initialStructure));
    setSelector(null);
  }, [initialStructure, service]);

  const validation = useMemo(() => service.validate(draft), [draft, service]);
  const completedHandIssues = validation.issues.filter(
    (issue) => issue.target.kind === 'completed-hand',
  );
  const winningStructureIssues = validation.issues.filter(
    (issue) => issue.target.kind === 'winning-structure',
  );

  function dispatch(command: CorrectionCommand) {
    const next = service.update(draft, command);
    setDraft(next);

    if (autoCommitValidChanges) {
      const result = service.commit(next);
      if (result.kind === 'valid') {
        onCommit(result.structure);
      }
    }
  }

  function commit() {
    const result = service.commit(draft);
    if (result.kind === 'valid') {
      onCommit(result.structure);
    }
  }

  return (
    <div style={editorStyle}>
      <div
        data-invalid={winningStructureIssues.length > 0 ? 'true' : 'false'}
        style={
          winningStructureIssues.length > 0
            ? { ...structureStyle, border: '2px solid #c92a2a' }
            : structureStyle
        }
      >
        <CorrectionRegion
          issues={completedHandIssues}
          label="手牌"
          onAdd={() =>
            setSelector({
              kind: 'insert',
              destination: { kind: 'completed-hand' },
            })
          }
        >
          <TileRow
            label="手牌"
            onEdit={(tileId) => setSelector({ kind: 'edit', tileId })}
            tiles={draft.completedHand}
          />
        </CorrectionRegion>

        <section aria-labelledby="correction-melds" style={{ display: 'grid', gap: 8 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
            <strong id="correction-melds">副露</strong>
            <button
              onClick={() => dispatch({ kind: 'add-meld-group' })}
              style={actionButtonStyle}
              type="button"
            >
              副露を追加
            </button>
          </div>

          {draft.meldGroups.length === 0 ? (
            <span style={{ color: '#6c757d', fontSize: 14 }}>副露なし</span>
          ) : null}

          {draft.meldGroups.map((group, index) => {
            const issues = validation.issues.filter(
              (issue) =>
                issue.target.kind === 'meld' &&
                issue.target.groupId === group.id,
            );
            const number = index + 1;

            return (
              <div
                aria-label={`副露 ${number}`}
                data-invalid={issues.length > 0 ? 'true' : 'false'}
                key={group.id}
                style={issues.length > 0 ? invalidRegionStyle : regionStyle}
              >
                <div
                  style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    justifyContent: 'space-between',
                    gap: 8,
                    alignItems: 'center',
                  }}
                >
                  <strong>副露 {number}</strong>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {group.kanOpenness === null ? null : (
                      <button
                        aria-label={`副露 ${number} 槓種別 ${kanOpennessLabel(group.kanOpenness)}`}
                        onClick={() =>
                          dispatch({
                            kind: 'toggle-kan-openness',
                            groupId: group.id,
                          })
                        }
                        style={actionButtonStyle}
                        type="button"
                      >
                        {kanOpennessLabel(group.kanOpenness)}
                      </button>
                    )}
                    <button
                      onClick={() =>
                        dispatch({ kind: 'remove-meld-group', groupId: group.id })
                      }
                      style={actionButtonStyle}
                      type="button"
                    >
                      副露 {number} を削除
                    </button>
                  </div>
                </div>

                <div style={tileRowStyle}>
                  <TileRow
                    label={`副露 ${number}`}
                    onEdit={(tileId) => setSelector({ kind: 'edit', tileId })}
                    tiles={group.tiles}
                  />
                  <button
                    aria-label={`副露 ${number} に牌を追加`}
                    onClick={() =>
                      setSelector({
                        kind: 'insert',
                        destination: { kind: 'meld', groupId: group.id },
                      })
                    }
                    style={addButtonStyle}
                    type="button"
                  >
                    ＋
                  </button>
                </div>

                <IssueMessages issues={issues} />
              </div>
            );
          })}
        </section>

        <IssueMessages issues={winningStructureIssues} />
      </div>

      <CorrectionRegion
        issues={[]}
        label="ドラ表示牌"
        onAdd={() =>
          setSelector({
            kind: 'insert',
            destination: { kind: 'dora-indicators' },
          })
        }
      >
        <TileRow
          label="ドラ表示牌"
          onEdit={(tileId) => setSelector({ kind: 'edit', tileId })}
          tiles={draft.doraIndicators}
        />
      </CorrectionRegion>

      {autoCommitValidChanges ? null : (
        <button
          disabled={!validation.canCommit}
          onClick={commit}
          style={{
            ...actionButtonStyle,
            minHeight: 46,
            fontWeight: 700,
            cursor: validation.canCommit ? 'pointer' : 'not-allowed',
            opacity: validation.canCommit ? 1 : 0.5,
          }}
          type="button"
        >
          {primaryActionLabel}
        </button>
      )}

      {selector === null ? null : (
        <TileSelector
          draft={draft}
          onClose={() => setSelector(null)}
          onCommand={(command) => {
            dispatch(command);
            setSelector(null);
          }}
          selector={selector}
        />
      )}
    </div>
  );
}

function CorrectionRegion({
  label,
  issues,
  onAdd,
  children,
}: {
  readonly label: string;
  readonly issues: readonly CorrectionIssue[];
  readonly onAdd: () => void;
  readonly children: ReactNode;
}) {
  return (
    <section
      aria-label={`${label}修正`}
      data-invalid={issues.length > 0 ? 'true' : 'false'}
      style={issues.length > 0 ? invalidRegionStyle : regionStyle}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
        <strong>{label}</strong>
        <button
          aria-label={`${label}に追加`}
          onClick={onAdd}
          style={addButtonStyle}
          type="button"
        >
          ＋
        </button>
      </div>
      {children}
      <IssueMessages issues={issues} />
    </section>
  );
}

function TileRow({
  label,
  tiles,
  onEdit,
}: {
  readonly label: string;
  readonly tiles: readonly TileInstance[];
  readonly onEdit: (tileId: TileInstanceId) => void;
}) {
  return (
    <div style={tileRowStyle}>
      {tiles.map((tile, index) => (
        <button
          aria-label={`${label} ${index + 1} ${formatTileIdentity(tile.tile)}`}
          key={tile.id}
          onClick={() => onEdit(tile.id)}
          style={tileButtonStyle}
          type="button"
        >
          <TileFace size="keyboard" tile={tile.tile} />
        </button>
      ))}
    </div>
  );
}

function TileSelector({
  draft,
  selector,
  onCommand,
  onClose,
}: {
  readonly draft: CorrectionDraft;
  readonly selector: SelectorState;
  readonly onCommand: (command: CorrectionCommand) => void;
  readonly onClose: () => void;
}) {
  const location =
    selector.kind === 'edit' ? findDraftTileLocation(draft, selector.tileId) : null;
  const moveDestinations = selector.kind === 'edit' && location !== null
    ? allDestinations(draft).filter(
        (destination) => !sameDestination(destination, location.destination),
      )
    : [];

  function choose(tile: TileIdentity) {
    if (selector.kind === 'insert') {
      onCommand({
        kind: 'add-tile',
        destination: selector.destination,
        tile,
        index: selector.index,
      });
      return;
    }

    onCommand({ kind: 'replace-tile', tileId: selector.tileId, tile });
  }

  return (
    <div onClick={onClose} style={selectorBackdropStyle}>
      <section
        aria-label="牌を選択"
        aria-modal="true"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        style={selectorStyle}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          <strong>{selector.kind === 'insert' ? '牌を追加' : '牌を修正'}</strong>
          <button onClick={onClose} style={actionButtonStyle} type="button">
            閉じる
          </button>
        </div>

        <div aria-label="牌キーボード" style={tileKeyboardStyle}>
          {TILE_KEYBOARD_ROWS.map((row) => (
            <div key={row.label} style={{ display: 'grid', gap: 4 }}>
              <span style={{ color: '#6c757d', fontSize: 12, fontWeight: 700 }}>
                {row.label}
              </span>
              <div aria-label={row.label} role="group" style={tileKeyboardRowStyle}>
                {row.tiles.map((tile) => (
                  <button
                    aria-label={formatTileIdentity(tile)}
                    key={`${tile.kind}-${tile.red ? 'red' : 'normal'}`}
                    onClick={() => choose(tile)}
                    style={selectorTileButtonStyle}
                    type="button"
                  >
                    <TileFace size="keyboard" tile={tile} />
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {selector.kind === 'edit' && location !== null ? (
          <div style={{ display: 'grid', gap: 8 }}>
            <strong>並び・所属を修正</strong>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
              <button
                disabled={location.index === 0}
                onClick={() =>
                  onCommand({
                    kind: 'move-tile',
                    tileId: selector.tileId,
                    destination: location.destination,
                    index: location.index - 1,
                  })
                }
                style={actionButtonStyle}
                type="button"
              >
                左へ
              </button>
              <button
                disabled={
                  location.index >=
                  destinationTiles(draft, location.destination).length - 1
                }
                onClick={() =>
                  onCommand({
                    kind: 'move-tile',
                    tileId: selector.tileId,
                    destination: location.destination,
                    index: location.index + 1,
                  })
                }
                style={actionButtonStyle}
                type="button"
              >
                右へ
              </button>
              {moveDestinations.map((destination) => (
                <button
                  key={destinationKey(destination)}
                  onClick={() =>
                    onCommand({
                      kind: 'move-tile',
                      tileId: selector.tileId,
                      destination,
                      index: destinationTiles(draft, destination).length,
                    })
                  }
                  style={actionButtonStyle}
                  type="button"
                >
                  {destinationLabel(draft, destination)}へ移動
                </button>
              ))}
            </div>

            <button
              onClick={() =>
                onCommand({ kind: 'remove-tile', tileId: selector.tileId })
              }
              style={{ ...actionButtonStyle, borderColor: '#c92a2a', color: '#a61e1e' }}
              type="button"
            >
              この牌を削除
            </button>
          </div>
        ) : null}
      </section>
    </div>
  );
}

function IssueMessages({ issues }: { readonly issues: readonly CorrectionIssue[] }) {
  if (issues.length === 0) {
    return null;
  }

  return (
    <div style={{ display: 'grid', gap: 4 }}>
      {issues.map((issue, index) => (
        <p key={`${issue.kind}-${index}`} role="status" style={issueStyle}>
          {correctionIssueMessage(issue)}
        </p>
      ))}
    </div>
  );
}

function correctionIssueMessage(issue: CorrectionIssue): string {
  switch (issue.kind) {
    case 'completed-hand-count':
      return '手牌の枚数が副露数と合っていません。';
    case 'invalid-completed-hand-tile':
      return '手牌に不正な牌があります。';
    case 'invalid-meld':
      return '副露の牌構成を修正してください。';
    case 'not-winning-shape':
      return '和了形として成立する牌姿に修正してください。';
  }
}

function kanOpennessLabel(openness: 'open' | 'concealed'): string {
  return openness === 'open' ? '明槓' : '暗槓';
}

function keyboardSuitRow(suit: 'm' | 'p' | 's'): readonly TileIdentity[] {
  return [
    ...[1, 2, 3, 4, 5].map(
      (rank) => ({ kind: `${rank}${suit}` as TileKind, red: false }),
    ),
    { kind: `5${suit}` as TileKind, red: true },
    ...[6, 7, 8, 9].map(
      (rank) => ({ kind: `${rank}${suit}` as TileKind, red: false }),
    ),
  ];
}

function findDraftTileLocation(
  draft: CorrectionDraft,
  tileId: TileInstanceId,
): DraftTileLocation | null {
  const completedHandIndex = draft.completedHand.findIndex((tile) => tile.id === tileId);
  if (completedHandIndex !== -1) {
    return {
      destination: { kind: 'completed-hand' },
      index: completedHandIndex,
    };
  }

  const doraIndex = draft.doraIndicators.findIndex((tile) => tile.id === tileId);
  if (doraIndex !== -1) {
    return {
      destination: { kind: 'dora-indicators' },
      index: doraIndex,
    };
  }

  for (const group of draft.meldGroups) {
    const tileIndex = group.tiles.findIndex((tile) => tile.id === tileId);
    if (tileIndex !== -1) {
      return {
        destination: { kind: 'meld', groupId: group.id },
        index: tileIndex,
      };
    }
  }

  return null;
}

function allDestinations(draft: CorrectionDraft): readonly CorrectionDestination[] {
  return [
    { kind: 'completed-hand' },
    { kind: 'dora-indicators' },
    ...draft.meldGroups.map(
      (group): CorrectionDestination => ({ kind: 'meld', groupId: group.id }),
    ),
  ];
}

function destinationTiles(
  draft: CorrectionDraft,
  destination: CorrectionDestination,
): readonly TileInstance[] {
  switch (destination.kind) {
    case 'completed-hand':
      return draft.completedHand;
    case 'dora-indicators':
      return draft.doraIndicators;
    case 'meld': {
      const group = draft.meldGroups.find((candidate) => candidate.id === destination.groupId);
      return group?.tiles ?? [];
    }
  }
}

function sameDestination(
  left: CorrectionDestination,
  right: CorrectionDestination,
): boolean {
  if (left.kind !== right.kind) {
    return false;
  }
  return left.kind !== 'meld' ||
    (right.kind === 'meld' && left.groupId === right.groupId);
}

function destinationKey(destination: CorrectionDestination): string {
  return destination.kind === 'meld'
    ? `meld-${destination.groupId}`
    : destination.kind;
}

function destinationLabel(
  draft: CorrectionDraft,
  destination: CorrectionDestination,
): string {
  switch (destination.kind) {
    case 'completed-hand':
      return '手牌';
    case 'dora-indicators':
      return 'ドラ表示牌';
    case 'meld': {
      const index = draft.meldGroups.findIndex((group) => group.id === destination.groupId);
      return `副露 ${index + 1}`;
    }
  }
}
