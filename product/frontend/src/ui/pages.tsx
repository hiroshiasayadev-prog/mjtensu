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
import { useState, type ReactNode } from 'react';
import {
  Link as RouterLink,
  Navigate,
  Outlet,
  useLocation,
  useNavigate,
} from 'react-router-dom';

import {
  selectHasActiveScoringSession,
  type ScoringSessionCalculation,
  type ScoringSessionService,
} from '@/application';

import { useApplicationStore } from './application-state';
import { ConditionsPageView } from './conditions-page';
import {
  appRoutePaths,
  navigateAfterCalculation,
  navigateAfterConditionCorrectionCancelled,
  navigateAfterRecognitionCorrectionCancelled,
  navigateAfterRecognitionCorrectionNeedsConditions,
  navigateAfterRecognitionCorrectionScored,
  navigateToConditionCorrection,
  navigateToHelp,
  navigateToNewRecognition,
  navigateToRecognitionCorrection,
  navigateToTop,
  navigateToUnscoredConditions,
  readConditionsNavigationState,
} from './navigation';
import { RecognitionCorrectionPageView } from './recognition-correction-page';
import { ResultPresentation } from './result-presentation';
import { useScoringFlowServices } from './scoring-flow-services';
import { TileCorrectionEditor } from './tile-correction-editor';

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
    selectHasActiveScoringSession,
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

export function ConditionsPage() {
  const scoringFlowServices = useScoringFlowServices();
  const scoringSession = useStoreBackedScoringSessionService();
  const activeScoringSession = useApplicationStore(
    (state) => state.activeScoringSession,
  );
  const location = useLocation();
  const navigate = useNavigate();
  const navigationState = readConditionsNavigationState(location.state);
  const mode = navigationState.fromResultConditionCorrection
    ? 'result-correction'
    : navigationState.fromConfirmedRecognitionCorrection
      ? 'recognition-repair'
      : 'initial';

  if (activeScoringSession === null) {
    return null;
  }

  if (scoringFlowServices === null) {
    return <ScoringFlowUnavailableState title="条件入力" />;
  }

  const initialScoringSession = activeScoringSession;

  function commitResultConditionCorrection(
    calculation: ScoringSessionCalculation,
  ) {
    scoringSession.update(initialScoringSession, {
      kind: 'select-winning-tile',
      tileId: calculation.state.winningTileId,
    });
    scoringSession.update(initialScoringSession, {
      kind: 'replace-conditions',
      conditions: calculation.state.conditions,
    });
    scoringSession.calculate(initialScoringSession);
    navigateAfterCalculation(navigate);
  }

  return (
    <ConditionsPageView
      initialFocus={navigationState.focus}
      initialSession={activeScoringSession}
      onCalculationComplete={
        mode === 'result-correction'
          ? commitResultConditionCorrection
          : () => navigateAfterCalculation(navigate)
      }
      onCancel={
        mode === 'result-correction'
          ? () => navigateAfterConditionCorrectionCancelled(navigate)
          : undefined
      }
      renderCorrectionEditor={
        mode === 'result-correction'
          ? undefined
          : ({ session, commitStructure }) => (
              <TileCorrectionEditor
                autoCommitValidChanges
                initialStructure={session.structure}
                onCommit={commitStructure}
                service={scoringFlowServices.correctionEditor}
              />
            )
      }
      sessionService={
        mode === 'result-correction'
          ? scoringFlowServices.scoringSession
          : scoringSession
      }
    />
  );
}

export function RecognitionCorrectionPage() {
  const scoringFlowServices = useScoringFlowServices();
  const scoringSession = useStoreBackedScoringSessionService();
  const activeScoringSession = useApplicationStore(
    (state) => state.activeScoringSession,
  );
  const navigate = useNavigate();

  if (activeScoringSession === null) {
    return null;
  }

  if (scoringFlowServices === null) {
    return <ScoringFlowUnavailableState title="認識結果を修正" />;
  }

  return (
    <RecognitionCorrectionPageView
      correctionEditorService={scoringFlowServices.correctionEditor}
      onCancel={() => navigateAfterRecognitionCorrectionCancelled(navigate)}
      onContinueToConditions={() =>
        navigateAfterRecognitionCorrectionNeedsConditions(navigate)
      }
      onReturnToResult={() => navigateAfterRecognitionCorrectionScored(navigate)}
      session={activeScoringSession}
      sessionService={scoringSession}
    />
  );
}

type StoreBackedScoringSessionService = Pick<
  ScoringSessionService,
  'update' | 'preview' | 'calculate'
>;

function useStoreBackedScoringSessionService(): StoreBackedScoringSessionService {
  const updateScoringSession = useApplicationStore(
    (state) => state.updateScoringSession,
  );
  const previewScoringSession = useApplicationStore(
    (state) => state.previewScoringSession,
  );
  const calculateScoringSession = useApplicationStore(
    (state) => state.calculateScoringSession,
  );

  return {
    update: (_session, command) => updateScoringSession(command),
    preview: () => previewScoringSession(),
    calculate: () => calculateScoringSession(),
  };
}

function ScoringFlowUnavailableState({ title }: { readonly title: string }) {
  return (
    <Stack gap="md" py="xl">
      <Title order={1}>{title}</Title>
      <Text role="alert">点数計算サービスを利用できません。</Text>
    </Stack>
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
          onClick={() => navigateToUnscoredConditions(navigate)}
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
