export const appRoutePaths = {
  top: '/',
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

export type AppNavigate = (
  destination: AppRoutePath,
  options?: AppNavigationOptions,
) => void;

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
  navigate(
    appRoutePaths.conditions,
    focus === undefined ? undefined : { state: { focus } },
  );
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
