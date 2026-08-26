import {
  AppShell,
  Button,
  Container,
  Group,
  List,
  Stack,
  Text,
  Title,
} from '@mantine/core';
import { useEffect, useState, type ReactNode } from 'react';
import {
  Link as RouterLink,
  Navigate,
  Outlet,
  useLocation,
  useNavigate,
} from 'react-router-dom';

import { createScoringSessionService } from '@/application';
import type { ScoringService } from '@/scoring';

import { useApplicationStore } from './application-state';
import { ConditionsPageView } from './conditions-page';
import {
  appRoutePaths,
  navigateAfterCalculation,
  navigateToConditionCorrection,
  navigateToHelp,
  navigateToNewRecognition,
  navigateToRecognitionCorrection,
  navigateToTop,
} from './navigation';
import { ResultPresentation } from './result-presentation';

const deferredScoringService: ScoringService = {
  validateWinningStructure: () => ({ kind: 'valid' }),
  preview: () => ({ kind: 'no-yaku' }),
  calculate: () => {
    throw new Error('Scoring service is not connected.');
  },
};

const conditionsPageSessionService =
  createScoringSessionService(deferredScoringService);

export function ProductionShell() {
  return (
    <AppShell header={{ height: 60 }} padding="md">
      <AppShell.Header>
        <Group h="100%" px="md" justify="space-between">
          <Button
            component={RouterLink}
            to={appRoutePaths.top}
            variant="transparent"
            px={0}
            size="compact-md"
          >
            mjtensu
          </Button>
        </Group>
      </AppShell.Header>
      <AppShell.Main>
        <Container size="sm">
          <Outlet />
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}

export function RequireActiveScoringSession({
  children,
}: {
  readonly children: ReactNode;
}) {
  const hasActiveScoringSession = useApplicationStore(
    (state) => state.activeScoringSession !== null,
  );

  if (!hasActiveScoringSession) {
    return <Navigate to={appRoutePaths.top} replace />;
  }

  return children;
}

export function TopPage() {
  const navigate = useNavigate();

  return (
    <Stack gap="xl" py="xl">
      <Stack gap="xs">
        <Title order={1}>mjtensu</Title>
        <Text c="dimmed">麻雀の和了形をカメラで読み取り、点数計算へ進みます。</Text>
      </Stack>

      <Stack gap="sm">
        <Button
          size="xl"
          onClick={() => navigateToNewRecognition(navigate)}
        >
          判定する
        </Button>
        <Button variant="subtle" onClick={() => navigateToHelp(navigate)}>
          使い方
        </Button>
      </Stack>
    </Stack>
  );
}

export function HelpPage() {
  const navigate = useNavigate();

  return (
    <Stack gap="lg" py="xl">
      <Stack gap="xs">
        <Title order={1}>使い方</Title>
        <Text c="dimmed">
          認識前に、牌を固定の撮影領域へ収めてください。
        </Text>
      </Stack>

      <List spacing="xs">
        <List.Item>認識中はPWAを横向きで使います。</List.Item>
        <List.Item>
          数えるドラ表示牌は、カンドラやリーチ時の裏ドラを含めて左上のドラ領域に置きます。
        </List.Item>
        <List.Item>完成した手牌は左下の手牌領域に置きます。</List.Item>
        <List.Item>
          副露は右側の正方形領域に、横並びのグループごとに重ねて置きます。
        </List.Item>
        <List.Item>すべての牌を見えている撮影領域の内側に収めます。</List.Item>
        <List.Item>
          シャッター操作は不要で、同じ牌構成が安定して見えると自動で進みます。
        </List.Item>
        <List.Item>
          和了牌を手牌列の右端に置くと初期選択が合いやすくなります。選択は条件入力で変更できます。
        </List.Item>
        <List.Item>
          ロン/ツモ、場風、自風、リーチ状態などは認識後に入力します。
        </List.Item>
        <List.Item>
          認識ミスは計算前の条件入力、または計算後の結果画面から修正できます。
        </List.Item>
      </List>

      <Button variant="light" onClick={() => navigateToTop(navigate)}>
        トップへ戻る
      </Button>
    </Stack>
  );
}

export function RecognitionPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const beginNewRecognitionAttempt = useApplicationStore(
    (state) => state.beginNewRecognitionAttempt,
  );

  useEffect(() => {
    if (shouldClearSessionForRecognition(location.state)) {
      const timeout = window.setTimeout(beginNewRecognitionAttempt, 0);

      return () => window.clearTimeout(timeout);
    }

    return undefined;
  }, [beginNewRecognitionAttempt, location.state]);

  return (
    <Stack gap="md" py="xl">
      <Title order={1}>認識</Title>
      <Text>
        カメラ認識ページの境界です。安定認識後は履歴を置換して条件入力へ進みます。
      </Text>
      <Button variant="light" onClick={() => navigateToTop(navigate)}>
        トップへ戻る
      </Button>
    </Stack>
  );
}

export function ConditionsPage() {
  const activeScoringSession = useApplicationStore(
    (state) => state.activeScoringSession,
  );
  const installScoringSession = useApplicationStore(
    (state) => state.installScoringSession,
  );
  const navigate = useNavigate();

  if (activeScoringSession === null) {
    return null;
  }

  return (
    <ConditionsPageView
      initialSession={activeScoringSession}
      onCalculationComplete={() => navigateAfterCalculation(navigate)}
      onSessionChange={installScoringSession}
      sessionService={conditionsPageSessionService}
    />
  );
}

export function ResultPage() {
  const activeScoringSession = useApplicationStore(
    (state) => state.activeScoringSession,
  );
  const navigate = useNavigate();
  const [fuDetailOpen, setFuDetailOpen] = useState(false);

  if (activeScoringSession?.latestResult === null) {
    return (
      <Stack gap="md" py="xl">
        <Title order={1}>結果</Title>
        <Text>現在の和了牌: {activeScoringSession.winningTileId}</Text>
        <Text c="dimmed">計算結果がまだありません。</Text>
        <Button
          variant="light"
          onClick={() => navigateToConditionCorrection(navigate)}
        >
          条件入力へ戻る
        </Button>
      </Stack>
    );
  }

  if (activeScoringSession === null) {
    return null;
  }

  return (
    <ResultPresentation
      calculation={activeScoringSession.latestResult}
      fuDetailOpen={fuDetailOpen}
      onCloseFuDetail={() => setFuDetailOpen(false)}
      onCorrectConditions={() => navigateToConditionCorrection(navigate)}
      onCorrectRecognition={() => navigateToRecognitionCorrection(navigate)}
      onFocusSeatWind={() => navigateToConditionCorrection(navigate, 'seatWind')}
      onNewRecognition={() => navigateToNewRecognition(navigate)}
      onOpenFuDetail={() => setFuDetailOpen(true)}
      structure={activeScoringSession.structure}
      winningTileId={activeScoringSession.winningTileId}
    />
  );
}

function shouldClearSessionForRecognition(state: unknown): boolean {
  return (
    typeof state === 'object' &&
    state !== null &&
    'clearActiveScoringSession' in state &&
    state.clearActiveScoringSession === true
  );
}
