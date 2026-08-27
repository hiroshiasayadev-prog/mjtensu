import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import type { TileKind } from '@/domain';
import {
  BASE_CLASSIFIER_LABELS,
  baseClassifierLabelToTile,
  mapBaseClassifierLogits,
  type BaseClassifierLabel,
} from '@/recognition/classifier/labels';
import {
  makeBaseClassifierBatchTensor,
  makeBaseClassifierTensor,
  preprocessGrayClassifierCrop,
  preprocessRgbClassifierCrop,
  type ClassifierCropImage,
  type ClassifierTensor,
} from '@/recognition/classifier/preprocessing';
import {
  createC8ClassifierRuntime,
  type ClassifierSession,
} from '@/recognition/classifier/runtime';
import { describe, expect, it } from 'vitest';

const imageSize = 4;
const grayCrop = {
  width: 1,
  height: 4,
  channels: 1,
  data: new Uint8Array([10, 20, 30, 40]),
} as const satisfies ClassifierCropImage;
const rgbCrop = {
  width: 1,
  height: 4,
  channels: 3,
  data: new Uint8Array([
    10, 20, 30,
    30, 40, 50,
    50, 60, 70,
    70, 80, 90,
  ]),
} as const satisfies ClassifierCropImage;

describe('C8 classifier crop preprocessing', () => {
  it('letterboxes grayscale crops with the source border median fill', () => {
    expect([...preprocessGrayClassifierCrop(grayCrop, imageSize)]).toEqual([
      25, 10, 25, 25,
      25, 20, 25, 25,
      25, 30, 25, 25,
      25, 40, 25, 25,
    ]);
  });

  it('letterboxes RGB crops with per-channel border median fill', () => {
    expect([...preprocessRgbClassifierCrop(rgbCrop, imageSize)]).toEqual([
      40, 50, 60, 10, 20, 30, 40, 50, 60, 40, 50, 60,
      40, 50, 60, 30, 40, 50, 40, 50, 60, 40, 50, 60,
      40, 50, 60, 50, 60, 70, 40, 50, 60, 40, 50, 60,
      40, 50, 60, 70, 80, 90, 40, 50, 60, 40, 50, 60,
    ]);
  });

  it('normalizes grayscale input into NCHW float32 tensors', () => {
    const tensor = makeBaseClassifierTensor(
      grayCrop,
      { mean: [0.5], std: [0.25] },
      imageSize,
    );

    expect(tensor.shape).toEqual([1, 1, imageSize, imageSize]);
    expect(tensor.data[0]).toBeCloseTo((25 / 255 - 0.5) / 0.25);
    expect(tensor.data[1]).toBeCloseTo((10 / 255 - 0.5) / 0.25);
  });

  it('packs multiple crops into one [N,C,H,W] tensor without changing per-crop normalization', () => {
    const tensor = makeBaseClassifierBatchTensor(
      [grayCrop, grayCrop],
      { mean: [0.5], std: [0.25] },
      imageSize,
    );

    expect(tensor.shape).toEqual([2, 1, imageSize, imageSize]);
    expect(tensor.data).toHaveLength(2 * imageSize * imageSize);
    expect(tensor.data[0]).toBeCloseTo((25 / 255 - 0.5) / 0.25);
    expect(tensor.data[imageSize * imageSize]).toBeCloseTo(
      (25 / 255 - 0.5) / 0.25,
    );
  });
});

describe('C8 base classifier mapping', () => {
  it('maps every class label exhaustively and keeps invalid unresolved', () => {
    const mappedKinds = new Set<TileKind>();

    for (const label of BASE_CLASSIFIER_LABELS) {
      const result = baseClassifierLabelToTile(label);
      if (label === 'invalid') {
        expect(result).toEqual({ kind: 'invalid' });
      } else {
        expect(result.kind).toBe('tile');
        if (result.kind === 'tile') {
          expect(result.tile.red).toBe(false);
          mappedKinds.add(result.tile.kind);
        }
      }
    }

    expect(BASE_CLASSIFIER_LABELS).toHaveLength(35);
    expect(mappedKinds.size).toBe(34);
  });

  it('selects the maximum base logit and maps honor labels to canonical z kinds', () => {
    expect(mapBaseClassifierLogits(logitsFor('east'))).toEqual({
      kind: 'tile',
      tile: { kind: '1z', red: false },
    });
    expect(mapBaseClassifierLogits(logitsFor('invalid'))).toEqual({
      kind: 'invalid',
    });
  });
});

describe('C8 classifier runtime red-five refinement', () => {
  it('classifies a representative batch with one base inference and one selective red-five inference', async () => {
    const base = new FakeSession(
      concatenateLogits(
        logitsFor('1m'),
        logitsFor('invalid'),
        logitsFor('5p'),
        logitsFor('5s'),
      ),
    );
    const redFive = new FakeSession(new Float32Array([1, 0, 0, 1]));
    const runtime = createC8ClassifierRuntime({
      baseClassifier: base,
      redFiveClassifier: redFive,
      baseNormalization: { mean: [0], std: [1] },
      redFiveNormalization: { mean: [0, 0, 0], std: [1, 1, 1] },
      imageSize,
    });

    await expect(
      runtime.classifyBatch([rgbCrop, rgbCrop, rgbCrop, rgbCrop]),
    ).resolves.toEqual([
      { kind: 'tile', tile: { kind: '1m', red: false } },
      { kind: 'invalid' },
      { kind: 'tile', tile: { kind: '5p', red: false } },
      { kind: 'tile', tile: { kind: '5s', red: true } },
    ]);
    expect(base.inputs).toHaveLength(1);
    expect(base.inputs[0]?.shape).toEqual([4, 1, imageSize, imageSize]);
    expect(redFive.inputs).toHaveLength(1);
    expect(redFive.inputs[0]?.shape).toEqual([2, 3, imageSize, imageSize]);
    expect(runtime.getLastBatchTiming()).toMatchObject({
      candidateCount: 4,
      redFiveCandidateCount: 2,
    });
  });

  it('skips red-five inference entirely when a batch has no five candidates', async () => {
    const base = new FakeSession(
      concatenateLogits(logitsFor('4m'), logitsFor('invalid'), logitsFor('9s')),
    );
    const redFive = new FakeSession([0, 1]);
    const runtime = createC8ClassifierRuntime({
      baseClassifier: base,
      redFiveClassifier: redFive,
      baseNormalization: { mean: [0], std: [1] },
      redFiveNormalization: { mean: [0, 0, 0], std: [1, 1, 1] },
      imageSize,
    });

    await expect(runtime.classifyBatch([rgbCrop, rgbCrop, rgbCrop])).resolves.toEqual([
      { kind: 'tile', tile: { kind: '4m', red: false } },
      { kind: 'invalid' },
      { kind: 'tile', tile: { kind: '9s', red: false } },
    ]);
    expect(base.inputs).toHaveLength(1);
    expect(redFive.inputs).toHaveLength(0);
  });

  it('does not invoke the red-five specialist for non-five and invalid base results', async () => {
    const base = new FakeSession(logitsFor('4m'), logitsFor('invalid'));
    const redFive = new FakeSession([0, 1]);
    const runtime = createC8ClassifierRuntime({
      baseClassifier: base,
      redFiveClassifier: redFive,
      baseNormalization: { mean: [0], std: [1] },
      redFiveNormalization: { mean: [0, 0, 0], std: [1, 1, 1] },
      imageSize,
    });

    await expect(runtime.classify(rgbCrop)).resolves.toEqual({
      kind: 'tile',
      tile: { kind: '4m', red: false },
    });
    await expect(runtime.classify(rgbCrop)).resolves.toEqual({ kind: 'invalid' });
    expect(redFive.inputs).toHaveLength(0);
  });

  it.each([
    ['5m', [0, 1], true],
    ['5p', [1, 0], false],
    ['5s', [0, 1], true],
  ] as const)(
    'refines base %s into ordinary-versus-red identity',
    async (baseLabel, redFiveLogits, expectedRed) => {
      const redFive = new FakeSession(redFiveLogits);
      const runtime = createC8ClassifierRuntime({
        baseClassifier: new FakeSession(logitsFor(baseLabel)),
        redFiveClassifier: redFive,
        baseNormalization: { mean: [0], std: [1] },
        redFiveNormalization: { mean: [0, 0, 0], std: [1, 1, 1] },
        imageSize,
      });

      await expect(runtime.classify(rgbCrop)).resolves.toEqual({
        kind: 'tile',
        tile: { kind: baseLabel, red: expectedRed },
      });
      expect(redFive.inputs).toHaveLength(1);
      expect(redFive.inputs[0]?.shape).toEqual([1, 3, imageSize, imageSize]);
    },
  );

  it('keeps ORT-specific values out of the Recognition public entry point', () => {
    const publicEntry = readFileSync(
      resolve(process.cwd(), 'src/recognition/index.ts'),
      'utf8',
    );

    expect(publicEntry).not.toMatch(/onnxruntime|InferenceSession|ORT/);
  });
});

function logitsFor(label: BaseClassifierLabel): Float32Array {
  const logits = new Float32Array(BASE_CLASSIFIER_LABELS.length);
  logits.fill(-1);
  logits[BASE_CLASSIFIER_LABELS.indexOf(label)] = 10;
  return logits;
}

function concatenateLogits(...parts: readonly Float32Array[]): Float32Array {
  const output = new Float32Array(parts.reduce((sum, part) => sum + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    output.set(part, offset);
    offset += part.length;
  }
  return output;
}

class FakeSession implements ClassifierSession {
  readonly inputs: ClassifierTensor[] = [];
  private outputIndex = 0;

  constructor(...outputs: readonly ArrayLike<number>[]) {
    this.outputs = outputs;
  }

  private readonly outputs: readonly ArrayLike<number>[];

  async run(input: ClassifierTensor): Promise<ArrayLike<number>> {
    this.inputs.push(input);
    const output = this.outputs[Math.min(this.outputIndex, this.outputs.length - 1)];
    this.outputIndex += 1;
    if (output === undefined) {
      throw new Error('Fake session has no configured outputs');
    }
    return output;
  }
}
