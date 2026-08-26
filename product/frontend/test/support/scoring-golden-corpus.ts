import type {
  TileIdentity,
  TileInstance,
  TileInstanceId,
  TileKind,
} from '@/domain';
import type {
  ScoringCalculation,
  ScoringConditions,
  ScoringInput,
  ScoringMeld,
  ScoringRuleProfile,
  YakuEntry,
} from '@/scoring';

export type GoldenTileToken = TileKind | 'red5m' | 'red5p' | 'red5s';

export interface ScoringGoldenMeldV1 {
  readonly kind: ScoringMeld['kind'];
  readonly tiles: readonly GoldenTileToken[];
}

export interface ScoringGoldenInputV1 {
  readonly completedHand: readonly GoldenTileToken[];
  readonly melds: readonly ScoringGoldenMeldV1[];
  readonly doraIndicators: readonly GoldenTileToken[];
  readonly winningTileIndex: number;
  readonly conditions: ScoringConditions;
}

export type ScoringGoldenExpectedV1 =
  | {
      readonly status: 'scored';
      readonly calculation: ScoringCalculation;
    }
  | {
      readonly status: 'not-winning-shape';
    }
  | {
      readonly status: 'no-yaku';
    };

export interface ScoringGoldenCaseV1<TCoverage extends string = string> {
  readonly id: string;
  readonly description: string;
  readonly coverage: readonly TCoverage[];
  readonly ruleProfileId: string;
  readonly input: ScoringGoldenInputV1;
  readonly expected: ScoringGoldenExpectedV1;
}

export interface ScoringGoldenCorpusV1<TCoverage extends string = string> {
  readonly schemaVersion: 1;
  readonly corpusId: string;
  readonly ruleProfiles: Readonly<Record<string, ScoringRuleProfile>>;
  readonly cases: readonly ScoringGoldenCaseV1<TCoverage>[];
}

export interface MaterializedScoringGoldenCaseV1<TCoverage extends string = string> {
  readonly id: string;
  readonly description: string;
  readonly coverage: readonly TCoverage[];
  readonly input: ScoringInput;
  readonly ruleProfile: ScoringRuleProfile;
  readonly expected: ScoringGoldenExpectedV1;
}

const RED_TOKEN_MAP: Readonly<Record<'red5m' | 'red5p' | 'red5s', TileKind>> = {
  red5m: '5m',
  red5p: '5p',
  red5s: '5s',
};

function tileIdentity(token: GoldenTileToken): TileIdentity {
  if (token in RED_TOKEN_MAP) {
    return {
      kind: RED_TOKEN_MAP[token as keyof typeof RED_TOKEN_MAP],
      red: true,
    };
  }

  return { kind: token as TileKind, red: false };
}

function tileInstance(
  caseId: string,
  index: number,
  token: GoldenTileToken,
): TileInstance {
  return {
    id: `golden:${caseId}:hand:${index}` as TileInstanceId,
    tile: tileIdentity(token),
  };
}

function materializeMeld(meld: ScoringGoldenMeldV1): ScoringMeld {
  const tiles = meld.tiles.map(tileIdentity);

  switch (meld.kind) {
    case 'chi':
    case 'pon':
      return {
        kind: meld.kind,
        tiles: tiles as [TileIdentity, TileIdentity, TileIdentity],
      };
    case 'open-kan':
    case 'concealed-kan':
      return {
        kind: meld.kind,
        tiles: tiles as [
          TileIdentity,
          TileIdentity,
          TileIdentity,
          TileIdentity,
        ],
      };
  }
}

export function materializeScoringGoldenCaseV1<TCoverage extends string>(
  corpus: ScoringGoldenCorpusV1<TCoverage>,
  goldenCase: ScoringGoldenCaseV1<TCoverage>,
): MaterializedScoringGoldenCaseV1<TCoverage> {
  const ruleProfile = corpus.ruleProfiles[goldenCase.ruleProfileId];
  if (ruleProfile === undefined) {
    throw new Error(
      `Golden case ${goldenCase.id} references unknown rule profile ${goldenCase.ruleProfileId}`,
    );
  }

  const completedHand = goldenCase.input.completedHand.map((token, index) =>
    tileInstance(goldenCase.id, index, token),
  );
  const winningTile = completedHand[goldenCase.input.winningTileIndex];
  if (winningTile === undefined) {
    throw new Error(
      `Golden case ${goldenCase.id} has invalid winningTileIndex ${goldenCase.input.winningTileIndex}`,
    );
  }

  return {
    id: goldenCase.id,
    description: goldenCase.description,
    coverage: goldenCase.coverage,
    ruleProfile,
    input: {
      completedHand,
      melds: goldenCase.input.melds.map(materializeMeld),
      doraIndicators: goldenCase.input.doraIndicators.map(tileIdentity),
      winningTileId: winningTile.id,
      conditions: goldenCase.input.conditions,
    },
    expected: goldenCase.expected,
  };
}

function normalizedKind(token: GoldenTileToken): TileKind {
  return tileIdentity(token).kind;
}

function validateMeld(caseId: string, meld: ScoringGoldenMeldV1, meldIndex: number): string[] {
  const errors: string[] = [];
  const expectedLength =
    meld.kind === 'open-kan' || meld.kind === 'concealed-kan' ? 4 : 3;

  if (meld.tiles.length !== expectedLength) {
    errors.push(
      `${caseId}: meld ${meldIndex} (${meld.kind}) has ${meld.tiles.length} tiles; expected ${expectedLength}`,
    );
    return errors;
  }

  const kinds = meld.tiles.map(normalizedKind);
  if (meld.kind === 'pon' || meld.kind === 'open-kan' || meld.kind === 'concealed-kan') {
    if (!kinds.every((kind) => kind === kinds[0])) {
      errors.push(`${caseId}: meld ${meldIndex} (${meld.kind}) is not one tile kind`);
    }
    return errors;
  }

  const suited = kinds.map((kind) => /^([1-9])([mps])$/.exec(kind));
  if (suited.some((match) => match === null)) {
    errors.push(`${caseId}: meld ${meldIndex} chi contains an honor tile`);
    return errors;
  }

  const matches = suited as RegExpExecArray[];
  const suit = matches[0]?.[2];
  const ranks = matches.map((match) => Number(match[1])).sort((a, b) => a - b);
  if (
    !matches.every((match) => match[2] === suit) ||
    ranks[1] !== (ranks[0] ?? 0) + 1 ||
    ranks[2] !== (ranks[1] ?? 0) + 1
  ) {
    errors.push(`${caseId}: meld ${meldIndex} chi is not a suited consecutive sequence`);
  }

  return errors;
}

function allCaseTokens(goldenCase: ScoringGoldenCaseV1): GoldenTileToken[] {
  return [
    ...goldenCase.input.completedHand,
    ...goldenCase.input.melds.flatMap((meld) => meld.tiles),
  ];
}

function validateTileMultiplicity(goldenCase: ScoringGoldenCaseV1): string[] {
  const counts = new Map<TileKind, number>();
  for (const token of allCaseTokens(goldenCase)) {
    const kind = normalizedKind(token);
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
  }

  return [...counts.entries()]
    .filter(([, count]) => count > 4)
    .map(
      ([kind, count]) =>
        `${goldenCase.id}: physical tile multiplicity ${kind}=${count} exceeds four`,
    );
}

function validateLogicalTileCount(goldenCase: ScoringGoldenCaseV1): string[] {
  const meldTileCount = goldenCase.input.melds.reduce(
    (sum, meld) => sum + meld.tiles.length,
    0,
  );
  const kanCount = goldenCase.input.melds.filter(
    (meld) => meld.kind === 'open-kan' || meld.kind === 'concealed-kan',
  ).length;
  const physicalCount = goldenCase.input.completedHand.length + meldTileCount;
  const expectedPhysicalCount = 14 + kanCount;

  return physicalCount === expectedPhysicalCount
    ? []
    : [
        `${goldenCase.id}: logical scoring structure has ${physicalCount} physical tiles; expected ${expectedPhysicalCount}`,
      ];
}

function validateExpectedYaku(caseId: string, yaku: readonly YakuEntry[]): string[] {
  const keys = yaku.map((entry) => `${entry.kind}:${entry.id}`);
  const duplicates = keys.filter((key, index) => keys.indexOf(key) !== index);
  return duplicates.length === 0
    ? []
    : [`${caseId}: expected normalized yaku contains duplicate IDs: ${duplicates.join(', ')}`];
}

function validateScoredExpectation(goldenCase: ScoringGoldenCaseV1): string[] {
  if (goldenCase.expected.status !== 'scored') {
    return [];
  }

  const errors = validateExpectedYaku(
    goldenCase.id,
    goldenCase.expected.calculation.yaku,
  );
  const calculation = goldenCase.expected.calculation;
  const isDealer = goldenCase.input.conditions.seatWind === 'east';

  if (calculation.winnerRole !== (isDealer ? 'dealer' : 'non-dealer')) {
    errors.push(`${goldenCase.id}: expected winnerRole disagrees with seat wind`);
  }
  if (calculation.winMethod !== goldenCase.input.conditions.winMethod) {
    errors.push(`${goldenCase.id}: expected winMethod disagrees with fixture input`);
  }

  if (calculation.winMethod === 'ron' && calculation.payment.kind !== 'ron') {
    errors.push(`${goldenCase.id}: ron input must use ron expected payment`);
  }
  if (
    calculation.winMethod === 'tsumo' &&
    isDealer &&
    calculation.payment.kind !== 'tsumo-dealer'
  ) {
    errors.push(`${goldenCase.id}: dealer tsumo input must use tsumo-dealer payment`);
  }
  if (
    calculation.winMethod === 'tsumo' &&
    !isDealer &&
    calculation.payment.kind !== 'tsumo-non-dealer'
  ) {
    errors.push(`${goldenCase.id}: non-dealer tsumo input must use tsumo-non-dealer payment`);
  }

  const actualYakuman = calculation.limit?.kind === 'yakuman' && !calculation.limit.counted;
  if (actualYakuman && calculation.han !== null) {
    errors.push(`${goldenCase.id}: actual yakuman expectation must use han=null`);
  }
  if (calculation.limit?.kind === 'yakuman' && calculation.fu !== null) {
    errors.push(`${goldenCase.id}: yakuman-class expectation must use fu=null`);
  }

  return errors;
}

export function validateScoringGoldenCorpusV1<TCoverage extends string>(
  corpus: ScoringGoldenCorpusV1<TCoverage>,
  requiredCoverage: readonly TCoverage[],
): string[] {
  const errors: string[] = [];
  const caseIds = new Set<string>();
  const observedCoverage = new Set<TCoverage>();
  const requiredCoverageSet = new Set(requiredCoverage);

  if (corpus.schemaVersion !== 1) {
    errors.push(`Unsupported scoring golden corpus schemaVersion ${String(corpus.schemaVersion)}`);
  }

  for (const goldenCase of corpus.cases) {
    if (caseIds.has(goldenCase.id)) {
      errors.push(`Duplicate scoring golden case id ${goldenCase.id}`);
    }
    caseIds.add(goldenCase.id);

    if (corpus.ruleProfiles[goldenCase.ruleProfileId] === undefined) {
      errors.push(
        `${goldenCase.id}: unknown rule profile ${goldenCase.ruleProfileId}`,
      );
    }

    if (
      !Number.isInteger(goldenCase.input.winningTileIndex) ||
      goldenCase.input.winningTileIndex < 0 ||
      goldenCase.input.winningTileIndex >= goldenCase.input.completedHand.length
    ) {
      errors.push(
        `${goldenCase.id}: winningTileIndex ${goldenCase.input.winningTileIndex} is outside completedHand`,
      );
    }

    for (const [meldIndex, meld] of goldenCase.input.melds.entries()) {
      errors.push(...validateMeld(goldenCase.id, meld, meldIndex));
    }
    errors.push(...validateLogicalTileCount(goldenCase));
    errors.push(...validateTileMultiplicity(goldenCase));
    errors.push(...validateScoredExpectation(goldenCase));

    for (const coverage of goldenCase.coverage) {
      if (!requiredCoverageSet.has(coverage)) {
        errors.push(`${goldenCase.id}: unknown coverage id ${coverage}`);
      }
      observedCoverage.add(coverage);
    }
  }

  for (const coverage of requiredCoverage) {
    if (!observedCoverage.has(coverage)) {
      errors.push(`Missing required scoring golden coverage: ${coverage}`);
    }
  }

  return errors;
}
