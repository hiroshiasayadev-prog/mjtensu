import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';

import { describe, expect, it } from 'vitest';

import {
  analyzeSourceText,
  checkArchitecture,
} from '../scripts/check-architecture-imports';

const srcRoot = resolve(process.cwd(), 'src');

function analyze(importerRelativePath: string, sourceText: string) {
  return analyzeSourceText({
    filePath: resolve(srcRoot, importerRelativePath),
    sourceText,
    srcRoot,
  });
}

describe('architecture import boundaries', () => {
  it('accepts public cross-module entry points and same-module private imports', () => {
    const violations = analyze(
      'ui/example.ts',
      [
        "import type { ApplicationPort } from '@/application';",
        "import type { Tile } from '@/domain';",
        "import { localHelper } from '@/ui/internal/local-helper';",
      ].join('\n'),
    );

    expect(violations).toEqual([]);
  });

  it('rejects alias and relative cross-module deep imports', () => {
    const violations = analyze(
      'ui/example.ts',
      [
        "import { first } from '@/application/internal/first';",
        "import { second } from '../recognition/internal/second';",
      ].join('\n'),
    );

    expect(violations.map(({ rule }) => rule)).toEqual([
      'cross-feature-public-entry',
      'cross-feature-public-entry',
    ]);
  });

  it('covers re-exports and dynamic imports without treating comments or strings as imports', () => {
    const violations = analyze(
      'ui/example.ts',
      [
        "// import { ignored } from '@/recognition/internal/comment';",
        "const text = \"import { ignored } from '@/scoring/internal/string'\";",
        "export { value } from '@/application/internal/re-export';",
        "const modulePromise = import('@/recognition/internal/dynamic');",
      ].join('\n'),
    );

    expect(violations.map(({ specifier }) => specifier)).toEqual([
      '@/application/internal/re-export',
      '@/recognition/internal/dynamic',
    ]);
  });

  it('rejects direct onnxruntime-web imports from UI and Application', () => {
    const uiViolations = analyze('ui/example.ts', "import * as ort from 'onnxruntime-web';");
    const applicationViolations = analyze(
      'application/example.ts',
      "import 'onnxruntime-web/webgpu';",
    );

    expect(uiViolations).toHaveLength(1);
    expect(uiViolations[0]?.rule).toBe('ui-application-no-onnxruntime-web');
    expect(applicationViolations).toHaveLength(1);
    expect(applicationViolations[0]?.rule).toBe('ui-application-no-onnxruntime-web');
  });

  it('rejects direct concrete Agari WASM imports from UI and Application', () => {
    const uiViolations = analyze(
      'ui/example.ts',
      "import { score_hand_v1 } from '@/scoring/infra/agari-wasm/agari_wasm';",
    );
    const applicationViolations = analyze(
      'application/example.ts',
      "import '../../../../external/agari/web/src/lib/wasm/agari_wasm.js';",
    );

    expect(uiViolations).toHaveLength(1);
    expect(uiViolations[0]?.rule).toBe('ui-application-no-concrete-agari-wasm');
    expect(applicationViolations).toHaveLength(1);
    expect(applicationViolations[0]?.rule).toBe(
      'ui-application-no-concrete-agari-wasm',
    );
  });

  it('rejects Recognition imports of UI even through the UI public entry point', () => {
    const violations = analyze(
      'recognition/example.ts',
      "import { RecognitionOverlay } from '@/ui';",
    );

    expect(violations).toHaveLength(1);
    expect(violations[0]?.rule).toBe('recognition-no-ui');
  });

  it('accepts recognition-owned onnxruntime-web imports', () => {
    const violations = analyze(
      'recognition/infra/onnx-runtime.ts',
      "import * as ort from 'onnxruntime-web';",
    );

    expect(violations).toEqual([]);
  });

  it('keeps the current production source tree free of architecture violations', () => {
    const result = checkArchitecture({ srcRoot });

    expect(result.filesChecked).toBeGreaterThan(0);
    expect(result.violations).toEqual([]);
  });

  it('exits non-zero for a fixture-backed production-code violation', () => {
    const scriptPath = resolve(process.cwd(), 'scripts/check-architecture-imports.ts');
    const fixtureSrcRoot = resolve(
      process.cwd(),
      'fixtures/architecture-invalid/src',
    );
    const result = spawnSync(process.execPath, [scriptPath, '--src', fixtureSrcRoot], {
      encoding: 'utf8',
    });

    expect(result.status).toBe(1);
    expect(result.stderr).toContain('[cross-feature-public-entry]');
    expect(result.stderr).toContain('ui/deep-import.ts');
  });
});
