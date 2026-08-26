import {
  assignSemanticRegion,
  compositeRectToSource,
  validateCaptureRegions,
} from './fixed-composite';
import {
  decodeNanoDetOutput,
  type NanoDetDecodeOptions,
} from './nanodet';
import { suppressDetectorDuplicates } from './duplicate-suppression';
import type {
  CaptureRegions,
  RegionDetection,
  TensorOutput,
} from './types';

interface DetectorPostprocessConfiguration extends NanoDetDecodeOptions {
  readonly duplicateOverlapThreshold: number;
}

const PRODUCTION_CONFIGURATION: DetectorPostprocessConfiguration = Object.freeze({
  confidenceThreshold: 0.35,
  nmsIouThreshold: 0.6,
  maximumDetections: 200,
  duplicateOverlapThreshold: 0.8,
});

export interface NanoDetDetectionPostprocessor {
  readonly process: (
    output: TensorOutput,
    captureRegions: CaptureRegions,
  ) => readonly RegionDetection[];
}

export function createProductionNanoDetPostprocessor(): NanoDetDetectionPostprocessor {
  return createNanoDetPostprocessor(PRODUCTION_CONFIGURATION);
}

export function createNanoDetPostprocessor(
  configuration: DetectorPostprocessConfiguration,
): NanoDetDetectionPostprocessor {
  const fixedConfiguration = Object.freeze({ ...configuration });
  return {
    process(output, captureRegions) {
      validateCaptureRegions(captureRegions);
      const assigned: RegionDetection[] = [];
      for (const detection of decodeNanoDetOutput(output, fixedConfiguration)) {
        const region = assignSemanticRegion(detection.box, captureRegions);
        if (region === null) {
          continue;
        }
        assigned.push({
          ...detection,
          region,
          sourceBox: compositeRectToSource(detection.box, region, captureRegions),
        });
      }
      return suppressDetectorDuplicates(
        assigned,
        fixedConfiguration.duplicateOverlapThreshold,
      );
    },
  };
}
