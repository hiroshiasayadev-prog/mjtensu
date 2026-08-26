import {
  Badge,
  Box,
  Button,
  Divider,
  Group,
  SimpleGrid,
  Stack,
  Text,
  Title,
} from '@mantine/core';

import type { RecognizedMeldGroup, RecognizedStructure, TileInstance } from '@/domain';
import {
  getYakuDisplayName,
  type FuCalculation,
  type LimitClassification,
  type ScoringCalculation,
  type ScoringPayment,
} from '@/scoring';

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

const HONOR_LABELS = {
  '1z': '東',
  '2z': '南',
  '3z': '西',
  '4z': '北',
  '5z': '白',
  '6z': '發',
  '7z': '中',
} as const;

const SUIT_LABELS = {
  m: '萬',
  p: '筒',
  s: '索',
} as const;

function formatPoints(points: number): string {
  return `${points.toLocaleString('ja-JP')}点`;
}

function formatTile(tile: TileInstance): string {
  const { kind, red } = tile.tile;
  if (kind.endsWith('z')) {
    return HONOR_LABELS[kind as keyof typeof HONOR_LABELS];
  }

  const number = kind.at(0);
  const suit = kind.at(1);
  const suitLabel =
    suit === 'm' || suit === 'p' || suit === 's' ? SUIT_LABELS[suit] : '';

  return `${red ? '赤' : ''}${number}${suitLabel}`;
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

function TileBadge({
  compact = false,
  selected,
  tile,
}: {
  readonly compact?: boolean;
  readonly selected: boolean;
  readonly tile: TileInstance;
}) {
  return (
    <Box
      aria-label={`${formatTile(tile)}${selected ? ' 和了牌' : ''}`}
      bg={selected ? 'yellow.1' : 'gray.0'}
      c={tile.tile.red ? 'red.8' : 'dark.8'}
      data-tile-id={tile.id}
      data-winning={selected ? 'true' : 'false'}
      fw={700}
      lh={1}
      px={compact ? 7 : 9}
      py={compact ? 7 : 10}
      style={{
        border: selected
          ? '2px solid var(--mantine-color-yellow-7)'
          : '1px solid var(--mantine-color-gray-4)',
        borderRadius: 4,
        minWidth: compact ? 34 : 42,
        textAlign: 'center',
      }}
    >
      <Text component="span" fz={compact ? 'sm' : 'md'} inherit>
        {formatTile(tile)}
      </Text>
    </Box>
  );
}

function TileLine({
  compact = false,
  tiles,
  winningTileId,
}: {
  readonly compact?: boolean;
  readonly tiles: readonly TileInstance[];
  readonly winningTileId: string;
}) {
  return (
    <Group gap={compact ? 4 : 6} role="list">
      {tiles.map((tile) => (
        <Box component="span" key={tile.id} role="listitem">
          <TileBadge
            compact={compact}
            selected={tile.id === winningTileId}
            tile={tile}
          />
        </Box>
      ))}
    </Group>
  );
}

function MeldTiles({
  group,
  winningTileId,
}: {
  readonly group: RecognizedMeldGroup;
  readonly winningTileId: string;
}) {
  return (
    <Box
      aria-label={`${group.kind} meld`}
      px="xs"
      py={6}
      style={{
        border: '1px solid var(--mantine-color-gray-3)',
        borderRadius: 4,
      }}
    >
      <TileLine compact tiles={group.tiles} winningTileId={winningTileId} />
    </Box>
  );
}

function EvidenceTiles({
  structure,
  winningTileId,
}: {
  readonly structure: RecognizedStructure;
  readonly winningTileId: string;
}) {
  return (
    <Stack gap="sm">
      <Stack gap={6}>
        <Text fw={700}>手牌</Text>
        <TileLine tiles={structure.completedHand} winningTileId={winningTileId} />
      </Stack>

      {structure.meldGroups.length > 0 && (
        <Stack gap={6}>
          <Text fw={700}>副露</Text>
          <Group gap="xs">
            {structure.meldGroups.map((group, index) => (
              <MeldTiles
                group={group}
                key={`${group.kind}-${index}`}
                winningTileId={winningTileId}
              />
            ))}
          </Group>
        </Stack>
      )}

      <Stack gap={6}>
        <Text fw={700}>ドラ表示牌</Text>
        {structure.doraIndicators.length > 0 ? (
          <TileLine compact tiles={structure.doraIndicators} winningTileId="" />
        ) : (
          <Text c="dimmed">なし</Text>
        )}
      </Stack>
    </Stack>
  );
}

export function YakuList({
  calculation,
}: {
  readonly calculation: ScoringCalculation;
}) {
  return (
    <Stack gap="xs">
      <Title order={2} size="h3">
        役
      </Title>
      <Stack gap={6}>
        {calculation.yaku.map((entry) => (
          <Group justify="space-between" key={`${entry.kind}-${entry.id}`}>
            <Text>{getYakuDisplayName(entry.id)}</Text>
            {entry.kind === 'regular' ? (
              <Badge variant="light">{entry.han}翻</Badge>
            ) : (
              <Badge color="red" variant="light">
                役満
              </Badge>
            )}
          </Group>
        ))}
        {calculation.dora.dora > 0 && (
          <Group justify="space-between">
            <Text>ドラ</Text>
            <Badge variant="light">{calculation.dora.dora}翻</Badge>
          </Group>
        )}
        {calculation.dora.akaDora > 0 && (
          <Group justify="space-between">
            <Text>赤ドラ</Text>
            <Badge variant="light">{calculation.dora.akaDora}翻</Badge>
          </Group>
        )}
      </Stack>
    </Stack>
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
    <Stack gap="sm">
      <Group align="center" justify="space-between">
        <Title order={2} size="h3">
          点数
        </Title>
        <Button
          aria-label="親子を修正"
          onClick={onFocusSeatWind}
          size="xs"
          variant="default"
        >
          {calculation.winnerRole === 'dealer' ? '親' : '子'}
        </Button>
      </Group>

      <Text fw={800} fz={40} lh={1.1}>
        {formatPoints(calculation.totalPoints)}
      </Text>

      <Group gap="xs">
        {calculation.fu !== null && <Badge>{visibleFu(calculation.fu)}符</Badge>}
        {calculation.han !== null && <Badge>{calculation.han}翻</Badge>}
        <Badge color={calculation.limit === null ? 'gray' : 'orange'}>
          {formatLimit(calculation.limit)}
        </Badge>
        {calculation.fu !== null && (
          <Button onClick={onOpenFuDetail} size="compact-sm" variant="subtle">
            符の詳細
          </Button>
        )}
      </Group>

      <Text fw={700}>{formatPayment(calculation.payment)}</Text>
    </Stack>
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
  if (!opened || fu === null) {
    return null;
  }

  return (
    <Box
      aria-label="符の詳細"
      p="md"
      role="dialog"
      style={{
        border: '1px solid var(--mantine-color-gray-3)',
        borderRadius: 4,
      }}
    >
      <Group justify="space-between" mb="sm">
        <Title order={2} size="h3">
          符の詳細
        </Title>
        <Button onClick={onClose} size="compact-sm" variant="subtle">
          閉じる
        </Button>
      </Group>
      {fu?.kind === 'standard' && (
        <Stack gap="xs">
          <Group justify="space-between">
            <Text>副底</Text>
            <Text>{fu.base}符</Text>
          </Group>
          <Group justify="space-between">
            <Text>門前ロン</Text>
            <Text>{fu.menzenRon}符</Text>
          </Group>
          <Group justify="space-between">
            <Text>ツモ</Text>
            <Text>{fu.tsumo}符</Text>
          </Group>
          <Group justify="space-between">
            <Text>面子</Text>
            <Text>{fu.melds}符</Text>
          </Group>
          <Group justify="space-between">
            <Text>雀頭</Text>
            <Text>{fu.pair}符</Text>
          </Group>
          <Group justify="space-between">
            <Text>待ち</Text>
            <Text>{fu.wait}符</Text>
          </Group>
          <Divider />
          <Group justify="space-between">
            <Text>合計</Text>
            <Text>{fu.rawTotal}符</Text>
          </Group>
          <Group justify="space-between">
            <Text fw={700}>切り上げ後</Text>
            <Text fw={700}>{fu.rounded}符</Text>
          </Group>
        </Stack>
      )}

      {fu?.kind === 'chiitoitsu' && (
        <Stack gap="xs">
          <Text>七対子は固定25符です。</Text>
          <Group justify="space-between">
            <Text fw={700}>最終符</Text>
            <Text fw={700}>{fu.fixed}符</Text>
          </Group>
        </Stack>
      )}
    </Box>
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
    <Stack gap="lg" py="xl">
      <Title order={1}>結果</Title>

      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
        <EvidenceTiles structure={structure} winningTileId={winningTileId} />
        <Stack gap="lg">
          <YakuList calculation={calculation} />
          <ScoreSummary
            calculation={calculation}
            onFocusSeatWind={onFocusSeatWind}
            onOpenFuDetail={onOpenFuDetail}
          />
        </Stack>
      </SimpleGrid>

      <Group gap="sm">
        <Button onClick={onCorrectRecognition} variant="default">
          認識結果を修正
        </Button>
        <Button onClick={onCorrectConditions} variant="default">
          条件を修正
        </Button>
        <Button onClick={onNewRecognition}>もう一度判定</Button>
      </Group>

      <FuDetailDialog
        fu={calculation.fu}
        onClose={onCloseFuDetail}
        opened={fuDetailOpen}
      />
    </Stack>
  );
}
