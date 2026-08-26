import { describe, expect, it } from 'vitest';

import {
  appRoutePaths,
  navigateAfterCalculation,
  navigateAfterRecognitionConfirmed,
  navigateToConditionCorrection,
  navigateToHelp,
  navigateToNewRecognition,
  navigateToRecognitionCorrection,
  navigateToTop,
  type AppNavigate,
} from '@/ui';

function navigationRecorder() {
  const calls: {
    readonly destination: string;
    readonly options: unknown;
  }[] = [];
  const navigate: AppNavigate = (destination, options) => {
    calls.push({ destination, options });
  };

  return { calls, navigate };
}

describe('route history helpers', () => {
  it('uses normal navigation for user-entered pages', () => {
    const { calls, navigate } = navigationRecorder();

    navigateToHelp(navigate);
    navigateToTop(navigate);
    navigateAfterCalculation(navigate);
    navigateToConditionCorrection(navigate);
    navigateToConditionCorrection(navigate, 'seatWind');
    navigateToRecognitionCorrection(navigate);

    expect(calls).toEqual([
      { destination: appRoutePaths.help, options: undefined },
      { destination: appRoutePaths.top, options: undefined },
      { destination: appRoutePaths.result, options: undefined },
      { destination: appRoutePaths.conditions, options: undefined },
      {
        destination: appRoutePaths.conditions,
        options: { state: { focus: 'seatWind' } },
      },
      {
        destination: appRoutePaths.recognition,
        options: { state: { mode: 'correction' } },
      },
    ]);
  });

  it('replaces Recognition when stable recognition commits to Conditions', () => {
    const { calls, navigate } = navigationRecorder();

    navigateAfterRecognitionConfirmed(navigate);

    expect(calls).toEqual([
      {
        destination: appRoutePaths.conditions,
        options: { replace: true },
      },
    ]);
  });

  it('enters a new recognition attempt with an explicit reset intent', () => {
    const { calls, navigate } = navigationRecorder();

    navigateToNewRecognition(navigate);

    expect(calls).toEqual([
      {
        destination: appRoutePaths.recognition,
        options: { state: { clearActiveScoringSession: true } },
      },
    ]);
  });
});
