export const appRoutePaths = {
  top: '/',
  debug: '/debug',
  recognition: '/recognition',
  recognitionCorrection: '/recognition/correction',
  conditions: '/conditions',
  result: '/result',
  help: '/help',
} as const;

export type AppRouteName = keyof typeof appRoutePaths;
export type AppRoutePath = (typeof appRoutePaths)[AppRouteName];

export interface AppNavigationOptions {
  readonly flushSync?: boolean;
  readonly replace?: boolean;
  readonly state?: unknown;
}

export interface ConditionsNavigationState {
  readonly fromResultConditionCorrection?: true;
  readonly fromConfirmedRecognitionCorrection?: true;
  readonly focus?: 'seatWind';
}

export type ConditionsNavigationMode =
  | 'initial'
  | 'result-correction'
  | 'recognition-repair';

export type AppNavigate = (
  destination: AppRoutePath,
  options?: AppNavigationOptions,
) => void;

export function readConditionsNavigationState(
  state: unknown,
): ConditionsNavigationState {
  if (typeof state !== 'object' || state === null) {
    return {};
  }

  const candidate = state as Record<string, unknown>;
  return {
    ...(candidate.fromResultConditionCorrection === true
      ? { fromResultConditionCorrection: true }
      : {}),
    ...(candidate.fromConfirmedRecognitionCorrection === true
      ? { fromConfirmedRecognitionCorrection: true }
      : {}),
    ...(candidate.focus === 'seatWind' ? { focus: 'seatWind' } : {}),
  };
}

export function navigateToTop(navigate: AppNavigate): void {
  navigate(appRoutePaths.top);
}

export function navigateToHelp(navigate: AppNavigate): void {
  navigate(appRoutePaths.help);
}

export function navigateToNewRecognition(navigate: AppNavigate): void {
  navigate(appRoutePaths.recognition, {
    state: { clearActiveScoringSession: true },
  });
}

export function navigateAfterRecognitionConfirmed(navigate: AppNavigate): void {
  navigate(appRoutePaths.conditions, { replace: true });
}

export function navigateAfterCalculation(navigate: AppNavigate): void {
  navigate(appRoutePaths.result);
}

export function navigateToConditionCorrection(
  navigate: AppNavigate,
  focus?: 'seatWind',
): void {
  const state: ConditionsNavigationState = {
    fromResultConditionCorrection: true,
    ...(focus === undefined ? {} : { focus }),
  };
  navigate(appRoutePaths.conditions, { state });
}

export function navigateToUnscoredConditions(navigate: AppNavigate): void {
  navigate(appRoutePaths.conditions);
}

export function navigateAfterConditionCorrectionCancelled(
  navigate: AppNavigate,
): void {
  navigate(appRoutePaths.result, { replace: true });
}

export function navigateBackFromConditions(
  navigate: AppNavigate,
  mode: ConditionsNavigationMode,
): void {
  if (mode === 'result-correction') {
    navigateAfterConditionCorrectionCancelled(navigate);
    return;
  }

  navigate(appRoutePaths.top, { replace: true });
}

export function navigateToRecognitionCorrection(navigate: AppNavigate): void {
  navigate(appRoutePaths.recognitionCorrection);
}

export function navigateAfterRecognitionCorrectionCancelled(
  navigate: AppNavigate,
): void {
  navigate(appRoutePaths.result, { replace: true });
}

export function navigateAfterRecognitionCorrectionScored(
  navigate: AppNavigate,
): void {
  navigate(appRoutePaths.result, { replace: true });
}

export function navigateAfterRecognitionCorrectionNeedsConditions(
  navigate: AppNavigate,
): void {
  navigate(appRoutePaths.conditions, {
    replace: true,
    state: { fromConfirmedRecognitionCorrection: true },
  });
}
