import type {
  CorrectionEditorService,
  ScoringSessionService,
  ScoringSessionState,
} from '@/application';
import type { RecognizedStructure } from '@/domain';

import { TileCorrectionEditor } from './tile-correction-editor';

export interface RecognitionCorrectionPageViewProps {
  readonly session: ScoringSessionState;
  readonly correctionEditorService: CorrectionEditorService;
  readonly sessionService: ScoringSessionService;
  readonly onCancel: () => void;
  readonly onSessionChange: (session: ScoringSessionState) => void;
  readonly onReturnToResult: () => void;
  readonly onContinueToConditions: () => void;
}

export function RecognitionCorrectionPageView({
  session,
  correctionEditorService,
  sessionService,
  onCancel,
  onSessionChange,
  onReturnToResult,
  onContinueToConditions,
}: RecognitionCorrectionPageViewProps) {
  function commitStructure(structure: RecognizedStructure) {
    const correctedSession = sessionService.update(session, {
      kind: 'replace-structure',
      structure,
    });

    // Once correction is confirmed, the old Result is stale. Install the
    // corrected session before deciding whether scoring can finish immediately.
    onSessionChange(correctedSession);

    let preview: ReturnType<ScoringSessionService['preview']>;
    try {
      preview = sessionService.preview(correctedSession);
    } catch {
      onContinueToConditions();
      return;
    }

    if (preview.kind !== 'ready' || preview.yaku.length === 0) {
      onContinueToConditions();
      return;
    }

    try {
      const calculation = sessionService.calculate(correctedSession);
      onSessionChange(calculation.state);
      onReturnToResult();
    } catch {
      // A failed recalculation must not restore the pre-correction Result.
      onContinueToConditions();
    }
  }

  return (
    <main
      style={{
        display: 'grid',
        gap: 18,
        maxWidth: 960,
        margin: '0 auto',
        padding: 16,
      }}
    >
      <header style={{ display: 'grid', gap: 6 }}>
        <h1 style={{ margin: 0 }}>認識結果を修正</h1>
        <p style={{ margin: 0 }}>現在の和了牌: {session.winningTileId}</p>
      </header>

      <TileCorrectionEditor
        initialStructure={session.structure}
        onCommit={commitStructure}
        primaryActionLabel="修正を確定"
        service={correctionEditorService}
      />

      <button
        onClick={onCancel}
        style={{
          minHeight: 42,
          border: '1px solid #adb5bd',
          borderRadius: 6,
          background: '#ffffff',
          cursor: 'pointer',
        }}
        type="button"
      >
        キャンセル
      </button>
    </main>
  );
}
