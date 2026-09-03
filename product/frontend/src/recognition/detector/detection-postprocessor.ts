import {
  assignSemanticRegion,
  assignSemanticRegionByCenter,
  compositeOrientedRectToSource,
  compositeRectToSource,
  validateCaptureRegions,
} from './fixed-composite';
import {
  decodeNanoDetOutput,
  type NanoDetDecodeOptions,
} from './nanodet';
import { suppressDetectorDuplicates } from './duplicate-suppression';
import {
  decodeRotatedFcosOutputs,
  orientedRectToAabb,
  type RotatedFcosDecodeOptions,
} from './rotated-fcos';
import type {
  CaptureRegions,
  RegionDetection,
  TensorOutput,
} from './types';

interface DetectorPostprocessConfiguration extends NanoDetDecodeOptions {
  readonly duplicateOverlapThreshold: number;
}

type DetectorOutput = TensorOutput | readonly TensorOutput[];

const PRODUCTION_CONFIGURATION: DetectorPostprocessConfiguration = Object.freeze({
  confidenceThreshold: 0.35,
  nmsIouThreshold: 0.6,
  maximumDetections: 200,
  duplicateOverlapThreshold: 0.8,
});

const ROTATED_FCOS_PRODUCTION_CONFIGURATION: RotatedFcosDecodeOptions = Object.freeze({
  confidenceThreshold: 0.3,
  nmsIouThreshold: 0.45,
  maximumDetections: 64,
});

export interface DetectorDetectionPostprocessor {
  readonly process: (
    output: DetectorOutput,
    captureRegions: CaptureRegions,
  ) => readonly RegionDetection[];
}

export type NanoDetDetectionPostprocessor = DetectorDetectionPostprocessor;

export function createProductionNanoDetPostprocessor(): NanoDetDetectionPostprocessor {
  return createNanoDetPostprocessor(PRODUCTION_CONFIGURATION);
}

export function createProductionRotatedFcosPostprocessor(): DetectorDetectionPostprocessor {
  return createRotatedFcosPostprocessor(ROTATED_FCOS_PRODUCTION_CONFIGURATION);
}

export function createNanoDetPostprocessor(
  configuration: DetectorPostprocessConfiguration,
): NanoDetDetectionPostprocessor {
  const fixedConfiguration = Object.freeze({ ...configuration });
  return {
    process(output, captureRegions) {
      validateCaptureRegions(captureRegions);
      const tensor = requireSingleOutput(output, 'NanoDet');
      const assigned: RegionDetection[] = [];
      for (const detection of decodeNanoDetOutput(tensor, fixedConfiguration)) {
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

export function createRotatedFcosPostprocessor(
  configuration: RotatedFcosDecodeOptions,
): DetectorDetectionPostprocessor {
  const fixedConfiguration = Object.freeze({ ...configuration });
  return {
    process(output, captureRegions) {
      validateCaptureRegions(captureRegions);
      const outputs = Array.isArray(output) ? output : [output];
      const assigned: RegionDetection[] = [];
      for (const detection of decodeRotatedFcosOutputs(outputs, fixedConfiguration)) {
        const region = assignSemanticRegionByCenter(
          detection.orientedBox.cx,
          detection.orientedBox.cy,
          captureRegions,
        );
        if (region === null) {
          continue;
        }
        const sourceOrientedBox = compositeOrientedRectToSource(
          detection.orientedBox,
          region,
          captureRegions,
        );
        assigned.push({
          id: detection.id,
          detectionIndex: detection.detectionIndex,
          classIndex: detection.classIndex,
          confidence: detection.confidence,
          box: orientedRectToAabb(detection.orientedBox),
          region,
          sourceBox: orientedRectToAabb(sourceOrientedBox),
          orientedBox: detection.orientedBox,
          sourceOrientedBox,
        });
      }
      return assigned;
    },
  };
}

function requireSingleOutput(output: DetectorOutput, label: string): TensorOutput {
  if ('dims' in output) {
    return output;
  }
  const tensor = output[0];
  if (output.length !== 1 || tensor === undefined) {
    throw new Error(`${label} requires exactly one detector output tensor.`);
  }
  return tensor;
}
