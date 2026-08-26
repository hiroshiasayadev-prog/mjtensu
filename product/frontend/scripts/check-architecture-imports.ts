import { readdirSync, readFileSync } from 'node:fs';
import { dirname, extname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createScanner, SyntaxKind } from 'typescript/unstable/ast';

const TOP_LEVEL_MODULES = new Set([
  'app',
  'application',
  'camera',
  'domain',
  'recognition',
  'scoring',
  'ui',
]);

const SOURCE_EXTENSIONS = new Set(['.ts', '.tsx']);

const ALLOWED_CROSS_MODULE_DEPENDENCIES: Readonly<Record<string, ReadonlySet<string>>> = {
  app: new Set(['application', 'camera', 'domain', 'recognition', 'scoring', 'ui']),
  application: new Set(['camera', 'domain', 'recognition', 'scoring']),
  camera: new Set(['domain']),
  domain: new Set(),
  recognition: new Set(['domain']),
  scoring: new Set(['domain']),
  ui: new Set(['application', 'camera', 'domain', 'recognition', 'scoring']),
};

const ZUSTAND_RUNTIME_RESOURCE_IDENTIFIERS = new Set([
  'InferenceSession',
  'MediaStream',
  'MediaStreamTrack',
  'MediaDevices',
  'HTMLVideoElement',
  'VideoFrame',
  'ImageBitmap',
  'CameraService',
  'RecognitionModelAssets',
  'RecognitionRuntime',
  'ScoringService',
  'ScoringSessionService',
]);

type ArchitectureRule =
  | 'cross-feature-public-entry'
  | 'top-level-dependency-direction'
  | 'recognition-no-ui'
  | 'ui-application-no-onnxruntime-web'
  | 'ui-application-no-concrete-agari-wasm'
  | 'zustand-no-runtime-resource-state';

export interface ArchitectureViolation {
  readonly rule: ArchitectureRule;
  readonly filePath: string;
  readonly line: number;
  readonly column: number;
  readonly specifier: string;
  readonly message: string;
}

interface ImportReference {
  readonly specifier: string;
  readonly position: number;
}

interface InternalTarget {
  readonly moduleName: string;
  readonly remainder: string;
}

export interface AnalyzeSourceOptions {
  readonly filePath: string;
  readonly sourceText: string;
  readonly srcRoot: string;
}

export interface CheckArchitectureOptions {
  readonly srcRoot: string;
}

export interface CheckArchitectureResult {
  readonly filesChecked: number;
  readonly violations: readonly ArchitectureViolation[];
}

function normalizeSlashes(value: string): string {
  return value.replaceAll('\\', '/');
}

function isInside(root: string, target: string): boolean {
  const relativePath = normalizeSlashes(relative(root, target));
  return relativePath === '' || (!relativePath.startsWith('../') && relativePath !== '..');
}

function moduleForFile(filePath: string, srcRoot: string): string | null {
  if (!isInside(srcRoot, filePath)) {
    return null;
  }

  const [firstSegment] = normalizeSlashes(relative(srcRoot, filePath)).split('/');
  return firstSegment !== undefined && TOP_LEVEL_MODULES.has(firstSegment)
    ? firstSegment
    : null;
}

function internalTargetForImport(
  importerPath: string,
  specifier: string,
  srcRoot: string,
): InternalTarget | null {
  let targetRelativePath: string;

  if (specifier.startsWith('@/')) {
    targetRelativePath = specifier.slice(2);
  } else if (specifier.startsWith('.')) {
    const absoluteTarget = resolve(dirname(importerPath), specifier);
    if (!isInside(srcRoot, absoluteTarget)) {
      return null;
    }
    targetRelativePath = normalizeSlashes(relative(srcRoot, absoluteTarget));
  } else {
    return null;
  }

  const [moduleName, ...rest] = normalizeSlashes(targetRelativePath).split('/');
  if (moduleName === undefined || !TOP_LEVEL_MODULES.has(moduleName)) {
    return null;
  }

  return {
    moduleName,
    remainder: rest.join('/'),
  };
}

function isPublicEntryTarget(remainder: string): boolean {
  return (
    remainder === '' ||
    remainder === 'index' ||
    /^index\.(?:ts|tsx|js|jsx|mts|cts)$/.test(remainder)
  );
}

function collectImportReferences(sourceText: string): ImportReference[] {
  const scanner = createScanner(true, undefined, sourceText);
  const references: ImportReference[] = [];
  const seenPositions = new Set<number>();

  const recordCurrentStringLiteral = (): void => {
    const position = scanner.getTokenStart();
    if (seenPositions.has(position)) {
      return;
    }
    seenPositions.add(position);
    references.push({ specifier: scanner.getTokenValue(), position });
  };

  const scanImportTail = (): void => {
    let token = scanner.scan();

    if (token === SyntaxKind.OpenParenToken) {
      token = scanner.scan();
      if (token === SyntaxKind.StringLiteral) {
        recordCurrentStringLiteral();
      }
      return;
    }

    if (token === SyntaxKind.StringLiteral) {
      recordCurrentStringLiteral();
      return;
    }

    while (token !== SyntaxKind.EndOfFile && token !== SyntaxKind.SemicolonToken) {
      if (token === SyntaxKind.FromKeyword) {
        token = scanner.scan();
        if (token === SyntaxKind.StringLiteral) {
          recordCurrentStringLiteral();
        }
        return;
      }
      token = scanner.scan();
    }
  };

  const scanExportTail = (): void => {
    let token = scanner.scan();
    let braceDepth = 0;

    while (token !== SyntaxKind.EndOfFile && token !== SyntaxKind.SemicolonToken) {
      if (token === SyntaxKind.OpenBraceToken) {
        braceDepth += 1;
      } else if (token === SyntaxKind.CloseBraceToken) {
        braceDepth = Math.max(0, braceDepth - 1);
      } else if (token === SyntaxKind.FromKeyword && braceDepth === 0) {
        token = scanner.scan();
        if (token === SyntaxKind.StringLiteral) {
          recordCurrentStringLiteral();
        }
        return;
      }

      token = scanner.scan();
      if (
        braceDepth === 0 &&
        token !== SyntaxKind.FromKeyword &&
        scanner.hasPrecedingLineBreak()
      ) {
        return;
      }
    }
  };

  const scanRequireTail = (): void => {
    let token = scanner.scan();
    if (token !== SyntaxKind.OpenParenToken) {
      return;
    }
    token = scanner.scan();
    if (token === SyntaxKind.StringLiteral) {
      recordCurrentStringLiteral();
    }
  };

  let token = scanner.scan();
  while (token !== SyntaxKind.EndOfFile) {
    if (token === SyntaxKind.ImportKeyword) {
      scanImportTail();
    } else if (token === SyntaxKind.ExportKeyword) {
      scanExportTail();
    } else if (
      token === SyntaxKind.RequireKeyword ||
      (token === SyntaxKind.Identifier && scanner.getTokenValue() === 'require')
    ) {
      scanRequireTail();
    }
    token = scanner.scan();
  }

  return references.sort((left, right) => left.position - right.position);
}

function lineAndColumnAt(sourceText: string, position: number): { line: number; column: number } {
  const prefix = sourceText.slice(0, position);
  const lines = prefix.split(/\r?\n/);
  return {
    line: lines.length,
    column: (lines.at(-1)?.length ?? 0) + 1,
  };
}

function makeViolation(
  sourceText: string,
  filePath: string,
  reference: ImportReference,
  rule: ArchitectureRule,
  message: string,
): ArchitectureViolation {
  const { line, column } = lineAndColumnAt(sourceText, reference.position);
  return {
    rule,
    filePath,
    line,
    column,
    specifier: reference.specifier,
    message,
  };
}

function isOnnxRuntimeImport(specifier: string): boolean {
  return specifier === 'onnxruntime-web' || specifier.startsWith('onnxruntime-web/');
}

function isConcreteAgariWasmImport(specifier: string): boolean {
  const normalized = normalizeSlashes(specifier).toLowerCase();
  return normalized.includes('agari') && normalized.includes('wasm');
}

function isZustandImport(specifier: string): boolean {
  return specifier === 'zustand' || specifier.startsWith('zustand/');
}

function isAllowedCrossModuleDependency(importerModule: string, targetModule: string): boolean {
  return ALLOWED_CROSS_MODULE_DEPENDENCIES[importerModule]?.has(targetModule) ?? false;
}

function findRuntimeResourceReference(
  sourceText: string,
): { identifier: string; position: number } | null {
  const scanner = createScanner(true, undefined, sourceText);
  let token = scanner.scan();

  while (token !== SyntaxKind.EndOfFile) {
    if (token === SyntaxKind.Identifier) {
      const identifier = scanner.getTokenValue();
      if (ZUSTAND_RUNTIME_RESOURCE_IDENTIFIERS.has(identifier)) {
        return { identifier, position: scanner.getTokenStart() };
      }
    }
    token = scanner.scan();
  }

  return null;
}

export function analyzeSourceText(options: AnalyzeSourceOptions): ArchitectureViolation[] {
  const filePath = resolve(options.filePath);
  const srcRoot = resolve(options.srcRoot);
  const importerModule = moduleForFile(filePath, srcRoot);
  const violations: ArchitectureViolation[] = [];
  const importReferences = collectImportReferences(options.sourceText);

  for (const reference of importReferences) {
    const { specifier } = reference;

    if (importerModule === 'ui' || importerModule === 'application') {
      if (isOnnxRuntimeImport(specifier)) {
        violations.push(
          makeViolation(
            options.sourceText,
            filePath,
            reference,
            'ui-application-no-onnxruntime-web',
            `${importerModule} must consume recognition contracts instead of importing onnxruntime-web directly.`,
          ),
        );
        continue;
      }

      if (isConcreteAgariWasmImport(specifier)) {
        violations.push(
          makeViolation(
            options.sourceText,
            filePath,
            reference,
            'ui-application-no-concrete-agari-wasm',
            `${importerModule} must consume the scoring public API instead of concrete Agari WASM bindings.`,
          ),
        );
        continue;
      }
    }

    const target = internalTargetForImport(filePath, specifier, srcRoot);
    if (!target || target.moduleName === importerModule) {
      continue;
    }

    if (importerModule === 'recognition' && target.moduleName === 'ui') {
      violations.push(
        makeViolation(
          options.sourceText,
          filePath,
          reference,
          'recognition-no-ui',
          'recognition must not import UI code.',
        ),
      );
      continue;
    }

    if (
      importerModule !== null &&
      !isAllowedCrossModuleDependency(importerModule, target.moduleName)
    ) {
      violations.push(
        makeViolation(
          options.sourceText,
          filePath,
          reference,
          'top-level-dependency-direction',
          `${importerModule} must not depend on ${target.moduleName}.`,
        ),
      );
      continue;
    }

    if (!isPublicEntryTarget(target.remainder)) {
      violations.push(
        makeViolation(
          options.sourceText,
          filePath,
          reference,
          'cross-feature-public-entry',
          `Cross-module import into ${target.moduleName} must use its public index entry point.`,
        ),
      );
    }
  }

  if (importerModule === 'application' && importReferences.some(({ specifier }) => isZustandImport(specifier))) {
    const runtimeResource = findRuntimeResourceReference(options.sourceText);
    if (runtimeResource !== null) {
      violations.push(
        makeViolation(
          options.sourceText,
          filePath,
          { specifier: runtimeResource.identifier, position: runtimeResource.position },
          'zustand-no-runtime-resource-state',
          `Application Zustand state must not own app-lifetime runtime/resource type ${runtimeResource.identifier}.`,
        ),
      );
    }
  }

  return violations;
}

function listSourceFiles(directory: string): string[] {
  const files: string[] = [];

  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const fullPath = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...listSourceFiles(fullPath));
      continue;
    }

    if (
      entry.isFile() &&
      SOURCE_EXTENSIONS.has(extname(entry.name)) &&
      !entry.name.endsWith('.d.ts')
    ) {
      files.push(fullPath);
    }
  }

  return files.sort((left, right) => left.localeCompare(right));
}

export function checkArchitecture(options: CheckArchitectureOptions): CheckArchitectureResult {
  const srcRoot = resolve(options.srcRoot);
  const sourceFiles = listSourceFiles(srcRoot);
  const violations = sourceFiles.flatMap((filePath) =>
    analyzeSourceText({
      filePath,
      sourceText: readFileSync(filePath, 'utf8'),
      srcRoot,
    }),
  );

  return {
    filesChecked: sourceFiles.length,
    violations,
  };
}

function formatViolation(violation: ArchitectureViolation, srcRoot: string): string {
  const displayPath = normalizeSlashes(relative(srcRoot, violation.filePath));
  return `${displayPath}:${violation.line}:${violation.column} [${violation.rule}] ${violation.message} Import: ${JSON.stringify(violation.specifier)}`;
}

function parseSrcRoot(args: readonly string[], defaultSrcRoot: string): string {
  if (args.length === 0) {
    return defaultSrcRoot;
  }

  if (args.length === 2 && args[0] === '--src' && args[1] !== undefined) {
    return resolve(args[1]);
  }

  throw new Error('Usage: node scripts/check-architecture-imports.ts [--src <source-root>]');
}

function runCli(): number {
  const scriptDirectory = dirname(fileURLToPath(import.meta.url));
  let srcRoot: string;

  try {
    srcRoot = parseSrcRoot(process.argv.slice(2), resolve(scriptDirectory, '../src'));
  } catch (error) {
    console.error(error instanceof Error ? error.message : String(error));
    return 2;
  }

  const result = checkArchitecture({ srcRoot });
  if (result.violations.length > 0) {
    console.error(`Architecture import boundaries: ${result.violations.length} violation(s).`);
    for (const violation of result.violations) {
      console.error(formatViolation(violation, srcRoot));
    }
    return 1;
  }

  console.log(`Architecture import boundaries: OK (${result.filesChecked} source files checked).`);
  return 0;
}

const invokedAsScript =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;

if (invokedAsScript) {
  process.exitCode = runCli();
}
