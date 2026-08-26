import { describe, expect, it } from 'vitest';

import {
  appRoutePaths,
  navigateAfterCalculation,
  navigateAfterConditionCorrectionCancelled,
  navigateAfterRecognitionConfirmed,
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
    navigateToUnscoredConditions(navigate);
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
        options: { state: { fromResultConditionCorrection: true } },
      },
      {
        destination: appRoutePaths.conditions,
        options: {
          state: { fromResultConditionCorrection: true, focus: 'seatWind' },
        },
      },
      {
        destination: appRoutePaths.recognitionCorrection,
        options: undefined,
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

  it('encodes Result-origin correction outcomes without stale Result restoration', () => {
    const { calls, navigate } = navigationRecorder();

    navigateAfterConditionCorrectionCancelled(navigate);
    navigateAfterRecognitionCorrectionCancelled(navigate);
    navigateAfterRecognitionCorrectionScored(navigate);
    navigateAfterRecognitionCorrectionNeedsConditions(navigate);

    expect(calls).toEqual([
      { destination: appRoutePaths.result, options: { replace: true } },
      { destination: appRoutePaths.result, options: { replace: true } },
      { destination: appRoutePaths.result, options: { replace: true } },
      {
        destination: appRoutePaths.conditions,
        options: {
          replace: true,
          state: { fromConfirmedRecognitionCorrection: true },
        },
      },
    ]);
  });

  it('reads only supported Conditions navigation state fields', () => {
    expect(
      readConditionsNavigationState({
        fromResultConditionCorrection: true,
        fromConfirmedRecognitionCorrection: false,
        focus: 'seatWind',
        ignored: 'value',
      }),
    ).toEqual({
      fromResultConditionCorrection: true,
      focus: 'seatWind',
    });
    expect(readConditionsNavigationState('invalid')).toEqual({});
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
