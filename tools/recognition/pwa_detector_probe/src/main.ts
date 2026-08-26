import { registerSW } from 'virtual:pwa-register';
import './styles.css';
import {
  Detection,
  INPUT_SIZE,
  centeredSquareCrop,
  createInputBuffer,
  decodeNanoDetOutput,
  preprocessVideoFrame,
} from './nanodet';
import {
  ProviderChoice,
  RuntimeHandle,
  initializeRuntime,
  providerChoiceFromLocation,
} from './runtime';
import { ProviderRunSummary, TelemetrySeries } from './telemetry';

const MODEL_URL = new URL('models/nanodet-plus-m-320.onnx', document.baseURI).href;
const DETECTOR_INTERVAL_MS = 100;
const RUN_DURATION_MS = 60_000;
const STORAGE_KEY = 'mjtensu.nanodet-provider-runs.v2';

const app = document.querySelector<HTMLDivElement>('#app');
if (app === null) {
  throw new Error('Missing #app root element.');
}

app.innerHTML = `
  <main class="app-shell">
    <section class="preview-area" aria-label="Rear camera preview">
      <div id="preview-stage" class="preview-stage">
        <video id="camera-video" class="camera-video" autoplay muted playsinline></video>
        <canvas id="overlay-canvas" class="overlay-canvas" aria-hidden="true"></canvas>
        <div class="evaluation-label">CENTER 1:1 · 320 × 320 INPUT</div>
      </div>
      <div class="top-status">
        <span id="runtime-badge" class="badge badge-waiting">runtime: loading</span>
        <span id="camera-badge" class="badge badge-waiting">camera: stopped</span>
        <span id="run-badge" class="badge badge-waiting">60s run: idle</span>
      </div>
      <div id="fatal-error" class="fatal-error hidden" role="alert"></div>
    </section>

    <aside class="telemetry-panel">
      <header class="panel-header">
        <div>
          <p class="eyebrow">NanoDet-Plus-m · 320 × 320</p>
          <h1>iPhone PWA detector probe</h1>
        </div>
        <button id="panel-toggle" class="icon-button" type="button" aria-label="Toggle telemetry panel">≡</button>
      </header>

      <div id="panel-body" class="panel-body">
        <section class="control-grid">
          <button id="start-camera" class="primary-button" type="button">Start rear camera</button>
          <button id="stop-camera" type="button" disabled>Stop camera</button>
          <button id="start-run" type="button" disabled>Run 60s measurement</button>
        </section>

        <section class="provider-controls" aria-label="Execution provider controls">
          <label for="provider-choice">Provider test</label>
          <div class="inline-control">
            <select id="provider-choice">
              <option value="auto">Auto fallback</option>
              <option value="webgl">WebGL</option>
              <option value="wasm-simd">WASM SIMD · 1 thread</option>
              <option value="wasm-threaded">WASM SIMD · multi-threaded</option>
            </select>
            <button id="reload-provider" type="button">Reload</button>
          </div>
          <p id="provider-detail" class="detail-line">initializing model…</p>
        </section>

        <section class="threshold-control">
          <div class="label-row">
            <label for="confidence-threshold">Confidence threshold</label>
            <output id="confidence-value" for="confidence-threshold">0.05</output>
          </div>
          <input id="confidence-threshold" type="range" min="0.01" max="0.90" step="0.01" value="0.05" />
        </section>

        <dl class="metric-grid">
          <div><dt>Selected provider</dt><dd id="metric-provider">—</dd></div>
          <div><dt>Camera frame</dt><dd id="metric-camera">—</dd></div>
          <div><dt>Preprocess</dt><dd id="metric-preprocess">—</dd></div>
          <div><dt>Inference</dt><dd id="metric-inference">—</dd></div>
          <div><dt>Decode / NMS</dt><dd id="metric-decode">—</dd></div>
          <div><dt>End-to-end</dt><dd id="metric-e2e">—</dd></div>
          <div><dt>Rolling median</dt><dd id="metric-median">—</dd></div>
          <div><dt>Rolling p95</dt><dd id="metric-p95">—</dd></div>
          <div><dt>Detector rate</dt><dd id="metric-hz">—</dd></div>
          <div><dt>Detections</dt><dd id="metric-detections">0</dd></div>
          <div><dt>Dropped frames</dt><dd id="metric-dropped">0</dd></div>
          <div><dt>Samples</dt><dd id="metric-samples">0</dd></div>
        </dl>

        <section class="run-summary">
          <h2>Latest 60s result</h2>
          <p id="run-summary-text">No completed run.</p>
          <p class="detail-line">Browser APIs do not expose device temperature. Record physical heat and display dimming manually.</p>
        </section>

        <section class="comparison-section">
          <div class="section-heading-row">
            <h2>Provider comparison</h2>
            <button id="clear-results" class="text-button" type="button">Clear</button>
          </div>
          <div class="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Provider</th>
                  <th>Median</th>
                  <th>p95</th>
                  <th>Hz</th>
                  <th>Slowdown</th>
                  <th>Dropped</th>
                </tr>
              </thead>
              <tbody id="comparison-body">
                <tr><td colspan="6">No completed provider runs.</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <details class="runtime-details">
          <summary>Runtime diagnostics</summary>
          <pre id="runtime-diagnostics"></pre>
        </details>
      </div>
    </aside>

    <canvas id="processing-canvas" class="processing-canvas" width="${INPUT_SIZE}" height="${INPUT_SIZE}" aria-hidden="true"></canvas>
  </main>
`;

const elements = {
  stage: requiredElement<HTMLElement>('preview-stage'),
  video: requiredElement<HTMLVideoElement>('camera-video'),
  overlay: requiredElement<HTMLCanvasElement>('overlay-canvas'),
  processingCanvas: requiredElement<HTMLCanvasElement>('processing-canvas'),
  runtimeBadge: requiredElement<HTMLElement>('runtime-badge'),
  cameraBadge: requiredElement<HTMLElement>('camera-badge'),
  runBadge: requiredElement<HTMLElement>('run-badge'),
  fatalError: requiredElement<HTMLElement>('fatal-error'),
  panelToggle: requiredElement<HTMLButtonElement>('panel-toggle'),
  panelBody: requiredElement<HTMLElement>('panel-body'),
  startCamera: requiredElement<HTMLButtonElement>('start-camera'),
  stopCamera: requiredElement<HTMLButtonElement>('stop-camera'),
  startRun: requiredElement<HTMLButtonElement>('start-run'),
  providerChoice: requiredElement<HTMLSelectElement>('provider-choice'),
  reloadProvider: requiredElement<HTMLButtonElement>('reload-provider'),
  providerDetail: requiredElement<HTMLElement>('provider-detail'),
  confidenceThreshold: requiredElement<HTMLInputElement>('confidence-threshold'),
  confidenceValue: requiredElement<HTMLOutputElement>('confidence-value'),
  metricProvider: requiredElement<HTMLElement>('metric-provider'),
  metricCamera: requiredElement<HTMLElement>('metric-camera'),
  metricPreprocess: requiredElement<HTMLElement>('metric-preprocess'),
  metricInference: requiredElement<HTMLElement>('metric-inference'),
  metricDecode: requiredElement<HTMLElement>('metric-decode'),
  metricE2e: requiredElement<HTMLElement>('metric-e2e'),
  metricMedian: requiredElement<HTMLElement>('metric-median'),
  metricP95: requiredElement<HTMLElement>('metric-p95'),
  metricHz: requiredElement<HTMLElement>('metric-hz'),
  metricDetections: requiredElement<HTMLElement>('metric-detections'),
  metricDropped: requiredElement<HTMLElement>('metric-dropped'),
  metricSamples: requiredElement<HTMLElement>('metric-samples'),
  runSummaryText: requiredElement<HTMLElement>('run-summary-text'),
  comparisonBody: requiredElement<HTMLTableSectionElement>('comparison-body'),
  clearResults: requiredElement<HTMLButtonElement>('clear-results'),
  runtimeDiagnostics: requiredElement<HTMLElement>('runtime-diagnostics'),
};

const processingContext = requireCanvas2dContext(
  elements.processingCanvas.getContext('2d', {
    alpha: false,
    willReadFrequently: true,
  }),
  'processing canvas',
);
const overlayContext = requireCanvas2dContext(
  elements.overlay.getContext('2d'),
  'overlay canvas',
);

const telemetry = new TelemetrySeries();
const inputBuffer = createInputBuffer();
const providerChoice = providerChoiceFromLocation();
let runtime: RuntimeHandle | null = null;
let cameraStream: MediaStream | null = null;
let detectorTimer: number | null = null;
let inferenceBusy = false;
let latestFrameRequested = false;
let droppedFrameCount = 0;
let latestDetections: Detection[] = [];
let runStartedAt: number | null = null;
let runTimeout: number | null = null;
let latestRunSummary: ProviderRunSummary | null = null;
let lastConsoleReportAt = 0;

initializePage();
void initializeModel();

function initializePage(): void {
  elements.providerChoice.value = providerChoice;
  elements.video.playsInline = true;
  elements.video.muted = true;

  elements.startCamera.addEventListener('click', () => void startCamera());
  elements.stopCamera.addEventListener('click', stopCamera);
  elements.startRun.addEventListener('click', startMeasurementRun);
  elements.reloadProvider.addEventListener('click', reloadWithSelectedProvider);
  elements.clearResults.addEventListener('click', clearStoredResults);
  elements.panelToggle.addEventListener('click', () => {
    elements.panelBody.classList.toggle('collapsed');
  });
  elements.confidenceThreshold.addEventListener('input', () => {
    elements.confidenceValue.value = Number(elements.confidenceThreshold.value).toFixed(2);
  });

  window.addEventListener('resize', drawOverlay);
  window.addEventListener('orientationchange', drawOverlay);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      drawOverlay();
    }
  });
  window.addEventListener('beforeunload', stopCamera);

  if (import.meta.env.PROD) {
    registerSW({
      immediate: true,
      onOfflineReady: () => console.info('[pwa] offline cache ready'),
      onNeedRefresh: () => console.info('[pwa] update available; reload to activate'),
      onRegisterError: (error) => console.error('[pwa] service worker registration failed', error),
    });
  } else if ('serviceWorker' in navigator) {
    void navigator.serviceWorker.getRegistrations().then((registrations) =>
      Promise.all(registrations.map((registration) => registration.unregister())),
    );
  }

  renderComparisonTable();
  updateTelemetryUi();
  window.setInterval(updateTelemetryUi, 250);
}

async function initializeModel(): Promise<void> {
  setBadge(elements.runtimeBadge, 'runtime: loading', 'waiting');
  elements.providerDetail.textContent = `choice=${providerChoice}; crossOriginIsolated=${window.crossOriginIsolated}`;
  try {
    runtime = await initializeRuntime(providerChoice, MODEL_URL);
    await warmUpRuntime(runtime);
    setBadge(elements.runtimeBadge, `runtime: ${runtime.provider}`, 'ready');
    elements.metricProvider.textContent = runtime.provider;
    elements.providerDetail.textContent = formatProviderDetail(runtime);
    elements.runtimeDiagnostics.textContent = JSON.stringify(
      {
        requestedProvider: providerChoice,
        selectedProvider: runtime.provider,
        wasmThreads: runtime.wasmThreads,
        wasmProxy: runtime.wasmProxy,
        crossOriginIsolated: window.crossOriginIsolated,
        hardwareConcurrency: navigator.hardwareConcurrency,
        userAgent: navigator.userAgent,
        modelUrl: MODEL_URL,
        inputNames: runtime.session.inputNames,
        outputNames: runtime.session.outputNames,
        fallbackFailures: runtime.initializationFailures,
      },
      null,
      2,
    );
    elements.startCamera.disabled = false;
    elements.startRun.disabled = cameraStream === null;
  } catch (error) {
    showFatalError(`Model initialization failed: ${errorMessage(error)}`);
    setBadge(elements.runtimeBadge, 'runtime: failed', 'error');
    console.error('[model-init-failed]', error);
  }
}

async function warmUpRuntime(handle: RuntimeHandle): Promise<void> {
  const inputName = handle.session.inputNames[0];
  if (inputName === undefined) {
    throw new Error('The ONNX model has no input name.');
  }
  const zeros = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let pass = 0; pass < 2; pass += 1) {
    const tensor = handle.createFloat32Tensor(zeros, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    await handle.session.run({ [inputName]: tensor });
  }
}

async function startCamera(): Promise<void> {
  hideFatalError();
  if (runtime === null) {
    showFatalError('The model is still loading.');
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    showFatalError('getUserMedia is unavailable. Open this page in a secure HTTPS context.');
    return;
  }

  stopCamera();
  elements.startCamera.disabled = true;
  setBadge(elements.cameraBadge, 'camera: requesting', 'waiting');

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { exact: 'environment' },
        width: { ideal: 1920 },
        height: { ideal: 1080 },
      },
    });
    elements.video.srcObject = cameraStream;
    await waitForVideoMetadata(elements.video);
    await elements.video.play();

    const track = cameraStream.getVideoTracks()[0];
    const settings = track?.getSettings();
    setBadge(elements.cameraBadge, 'camera: running', 'ready');
    elements.metricCamera.textContent = formatCameraFrame(elements.video);
    const crop = centeredSquareCrop(elements.video.videoWidth, elements.video.videoHeight);
    elements.runtimeDiagnostics.textContent = appendDiagnosticSection(
      elements.runtimeDiagnostics.textContent ?? '{}',
      'camera',
      {
        settings: settings ?? {},
        videoFrame: [elements.video.videoWidth, elements.video.videoHeight],
        evaluationCrop: {
          mode: 'center-square',
          x: crop.x,
          y: crop.y,
          size: crop.size,
          modelInput: [INPUT_SIZE, INPUT_SIZE],
        },
      },
    );
    elements.stopCamera.disabled = false;
    elements.startRun.disabled = false;
    startDetectorScheduler();
    drawOverlay();
  } catch (error) {
    cameraStream = null;
    setBadge(elements.cameraBadge, 'camera: failed', 'error');
    showFatalError(`Camera startup failed: ${errorMessage(error)}`);
    elements.startCamera.disabled = false;
    elements.stopCamera.disabled = true;
    elements.startRun.disabled = true;
    console.error('[camera-start-failed]', error);
  }
}

function stopCamera(): void {
  if (detectorTimer !== null) {
    window.clearInterval(detectorTimer);
    detectorTimer = null;
  }
  if (runTimeout !== null) {
    window.clearTimeout(runTimeout);
    runTimeout = null;
  }
  runStartedAt = null;
  latestFrameRequested = false;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  elements.video.srcObject = null;
  latestDetections = [];
  drawOverlay();
  setBadge(elements.cameraBadge, 'camera: stopped', 'waiting');
  setBadge(elements.runBadge, '60s run: idle', 'waiting');
  elements.startCamera.disabled = runtime === null;
  elements.stopCamera.disabled = true;
  elements.startRun.disabled = true;
}

function startDetectorScheduler(): void {
  if (detectorTimer !== null) {
    window.clearInterval(detectorTimer);
  }
  detectorTimer = window.setInterval(() => {
    if (runtime === null || cameraStream === null || elements.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }
    if (inferenceBusy) {
      droppedFrameCount += 1;
      latestFrameRequested = true;
      return;
    }
    void detectCurrentFrame();
  }, DETECTOR_INTERVAL_MS);
}

async function detectCurrentFrame(): Promise<void> {
  const activeRuntime = runtime;
  const activeStream = cameraStream;
  if (activeRuntime === null || activeStream === null || inferenceBusy) {
    return;
  }

  inferenceBusy = true;
  latestFrameRequested = false;
  const endToEndStartedAt = performance.now();
  try {
    const preprocessStartedAt = performance.now();
    const inputData = preprocessVideoFrame(elements.video, processingContext, inputBuffer);
    const input = activeRuntime.createFloat32Tensor(inputData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    const preprocessMs = performance.now() - preprocessStartedAt;

    const inputName = activeRuntime.session.inputNames[0];
    if (inputName === undefined) {
      throw new Error('The ONNX session has no input name.');
    }

    const inferenceStartedAt = performance.now();
    const outputs = await activeRuntime.session.run({ [inputName]: input });
    const inferenceMs = performance.now() - inferenceStartedAt;

    const outputName = activeRuntime.session.outputNames[0];
    const output = outputName === undefined ? undefined : outputs[outputName];
    if (output === undefined) {
      throw new Error(`The ONNX session did not return output ${outputName ?? '(missing output name)'}.`);
    }

    const decodeStartedAt = performance.now();
    const threshold = Number(elements.confidenceThreshold.value);
    const detections = decodeNanoDetOutput(output, threshold);
    const decodeMs = performance.now() - decodeStartedAt;
    const completedAt = performance.now();

    if (cameraStream !== activeStream) {
      return;
    }
    latestDetections = detections;
    telemetry.add({
      preprocessMs,
      inferenceMs,
      decodeMs,
      endToEndMs: completedAt - endToEndStartedAt,
      completedAt,
      detectionCount: latestDetections.length,
    });
    drawOverlay();
  } catch (error) {
    showFatalError(`Detection failed: ${errorMessage(error)}`);
    console.error('[detection-failed]', error);
    if (detectorTimer !== null) {
      window.clearInterval(detectorTimer);
      detectorTimer = null;
    }
  } finally {
    inferenceBusy = false;
    if (
      latestFrameRequested &&
      detectorTimer !== null &&
      runtime !== null &&
      cameraStream !== null
    ) {
      latestFrameRequested = false;
      void detectCurrentFrame();
    }
  }
}

function drawOverlay(): void {
  const stageRect = elements.stage.getBoundingClientRect();
  const width = Math.max(1, stageRect.width);
  const height = Math.max(1, stageRect.height);
  const devicePixelRatio = window.devicePixelRatio || 1;
  const backingWidth = Math.round(width * devicePixelRatio);
  const backingHeight = Math.round(height * devicePixelRatio);

  if (elements.overlay.width !== backingWidth || elements.overlay.height !== backingHeight) {
    elements.overlay.width = backingWidth;
    elements.overlay.height = backingHeight;
  }

  overlayContext.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  overlayContext.clearRect(0, 0, width, height);

  if (elements.video.videoWidth <= 0 || elements.video.videoHeight <= 0) {
    return;
  }

  overlayContext.lineWidth = Math.max(2, width / 420);
  overlayContext.font = `${Math.max(12, width / 42)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  overlayContext.textBaseline = 'top';

  for (const detection of latestDetections) {
    const x = (detection.x1 / INPUT_SIZE) * width;
    const y = (detection.y1 / INPUT_SIZE) * height;
    const boxWidth = ((detection.x2 - detection.x1) / INPUT_SIZE) * width;
    const boxHeight = ((detection.y2 - detection.y1) / INPUT_SIZE) * height;
    const label = detection.score.toFixed(2);
    const textMetrics = overlayContext.measureText(label);
    const labelWidth = textMetrics.width + 10;
    const labelHeight = Math.max(18, width / 34);
    const labelY = Math.max(0, y - labelHeight);

    overlayContext.strokeStyle = '#57e389';
    overlayContext.strokeRect(x, y, boxWidth, boxHeight);
    overlayContext.fillStyle = 'rgba(9, 11, 16, 0.82)';
    overlayContext.fillRect(x, labelY, labelWidth, labelHeight);
    overlayContext.fillStyle = '#b9fbcf';
    overlayContext.fillText(label, x + 5, labelY + 2);
  }
}

function startMeasurementRun(): void {
  if (runtime === null || cameraStream === null || runStartedAt !== null) {
    return;
  }

  telemetry.clear();
  droppedFrameCount = 0;
  latestRunSummary = null;
  runStartedAt = performance.now();
  elements.startRun.disabled = true;
  setBadge(elements.runBadge, '60s run: 60.0s', 'active');
  runTimeout = window.setTimeout(finishMeasurementRun, RUN_DURATION_MS);
}

function finishMeasurementRun(): void {
  if (runtime === null || runStartedAt === null) {
    return;
  }

  const endedAt = performance.now();
  latestRunSummary = telemetry.summarizeRun(
    runtime.provider,
    runStartedAt,
    endedAt,
    droppedFrameCount,
    DETECTOR_INTERVAL_MS,
  );
  storeRunSummary(latestRunSummary);
  runStartedAt = null;
  runTimeout = null;
  setBadge(elements.runBadge, '60s run: complete', 'ready');
  elements.startRun.disabled = cameraStream === null;
  renderRunSummary();
  renderComparisonTable();
  console.info('[60s-provider-run]', latestRunSummary);
}

function updateTelemetryUi(): void {
  const now = performance.now();
  const snapshot = telemetry.snapshot(now);
  const latest = snapshot.latest;

  elements.metricProvider.textContent = runtime?.provider ?? '—';
  elements.metricCamera.textContent = formatCameraFrame(elements.video);
  elements.metricPreprocess.textContent = formatMilliseconds(latest?.preprocessMs);
  elements.metricInference.textContent = formatMilliseconds(latest?.inferenceMs);
  elements.metricDecode.textContent = formatMilliseconds(latest?.decodeMs);
  elements.metricE2e.textContent = formatMilliseconds(latest?.endToEndMs);
  elements.metricMedian.textContent = formatMilliseconds(snapshot.rollingMedianMs || undefined);
  elements.metricP95.textContent = formatMilliseconds(snapshot.rollingP95Ms || undefined);
  elements.metricHz.textContent = snapshot.effectiveHz > 0 ? `${snapshot.effectiveHz.toFixed(2)} Hz` : '—';
  elements.metricDetections.textContent = String(latest?.detectionCount ?? latestDetections.length);
  const inferredDroppedFrames =
    runStartedAt === null
      ? droppedFrameCount
      : Math.max(
          droppedFrameCount,
          Math.floor((now - runStartedAt) / DETECTOR_INTERVAL_MS) - snapshot.sampleCount,
        );
  elements.metricDropped.textContent = String(Math.max(0, inferredDroppedFrames));
  elements.metricSamples.textContent = String(snapshot.sampleCount);

  if (runStartedAt !== null) {
    const remainingMs = Math.max(0, RUN_DURATION_MS - (now - runStartedAt));
    setBadge(elements.runBadge, `60s run: ${(remainingMs / 1000).toFixed(1)}s`, 'active');
  }

  if (now - lastConsoleReportAt >= 1000 && latest !== null) {
    lastConsoleReportAt = now;
    const crop =
      elements.video.videoWidth > 0 && elements.video.videoHeight > 0
        ? centeredSquareCrop(elements.video.videoWidth, elements.video.videoHeight)
        : null;
    console.info('[detector-telemetry]', {
      provider: runtime?.provider,
      cameraFrame: [elements.video.videoWidth, elements.video.videoHeight],
      evaluationCrop: crop,
      preprocessMs: latest.preprocessMs,
      inferenceMs: latest.inferenceMs,
      decodeNmsMs: latest.decodeMs,
      endToEndMs: latest.endToEndMs,
      rollingMedianMs: snapshot.rollingMedianMs,
      rollingP95Ms: snapshot.rollingP95Ms,
      effectiveDetectorHz: snapshot.effectiveHz,
      detectionCount: latest.detectionCount,
      droppedFrameCount,
      samples: snapshot.sampleCount,
    });
  }
}

function renderRunSummary(): void {
  if (latestRunSummary === null) {
    elements.runSummaryText.textContent = 'No completed run.';
    return;
  }
  const summary = latestRunSummary;
  elements.runSummaryText.textContent = [
    `${summary.provider}: ${summary.effectiveHz.toFixed(2)} Hz`,
    `pre median/p95 ${summary.medianPreprocessMs.toFixed(1)}/${summary.p95PreprocessMs.toFixed(1)} ms`,
    `infer ${summary.medianInferenceMs.toFixed(1)}/${summary.p95InferenceMs.toFixed(1)} ms`,
    `decode ${summary.medianDecodeMs.toFixed(1)}/${summary.p95DecodeMs.toFixed(1)} ms`,
    `e2e ${summary.medianEndToEndMs.toFixed(1)}/${summary.p95EndToEndMs.toFixed(1)} ms`,
    `e2e first→last ${summary.firstTenSecondsMedianMs.toFixed(1)}→${summary.lastTenSecondsMedianMs.toFixed(1)} ms (${formatSignedPercent(summary.slowdownPercent)})`,
    `infer first→last ${summary.firstTenSecondsInferenceMedianMs.toFixed(1)}→${summary.lastTenSecondsInferenceMedianMs.toFixed(1)} ms (${formatSignedPercent(summary.inferenceSlowdownPercent)})`,
    `completed ${summary.sampleCount}/${summary.expectedDetectorRequests}`,
    `dropped ${summary.droppedFrames}`,
  ].join(' · ');
}

function renderComparisonTable(): void {
  const summaries = loadStoredSummaries();
  if (summaries.length === 0) {
    elements.comparisonBody.innerHTML = '<tr><td colspan="6">No completed provider runs.</td></tr>';
    return;
  }

  elements.comparisonBody.replaceChildren(
    ...summaries.map((summary) => {
      const row = document.createElement('tr');
      appendCell(row, summary.provider);
      appendCell(row, `${summary.medianEndToEndMs.toFixed(1)} ms`);
      appendCell(row, `${summary.p95EndToEndMs.toFixed(1)} ms`);
      appendCell(row, summary.effectiveHz.toFixed(2));
      appendCell(row, formatSignedPercent(summary.slowdownPercent));
      appendCell(row, String(summary.droppedFrames));
      return row;
    }),
  );
}

function storeRunSummary(summary: ProviderRunSummary): void {
  const summaries = loadStoredSummaries().filter((candidate) => candidate.provider !== summary.provider);
  summaries.push(summary);
  summaries.sort((left, right) => providerOrder(left.provider) - providerOrder(right.provider));
  localStorage.setItem(STORAGE_KEY, JSON.stringify(summaries));
}

function loadStoredSummaries(): ProviderRunSummary[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw === null) {
      return [];
    }
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as ProviderRunSummary[]) : [];
  } catch (error) {
    console.warn('[stored-run-read-failed]', error);
    return [];
  }
}

function clearStoredResults(): void {
  localStorage.removeItem(STORAGE_KEY);
  latestRunSummary = null;
  renderRunSummary();
  renderComparisonTable();
}

function reloadWithSelectedProvider(): void {
  const selected = elements.providerChoice.value as ProviderChoice;
  const url = new URL(window.location.href);
  if (selected === 'auto') {
    url.searchParams.delete('provider');
  } else {
    url.searchParams.set('provider', selected);
  }
  window.location.assign(url);
}

function formatProviderDetail(handle: RuntimeHandle): string {
  const threadDetail =
    handle.provider === 'webgl'
      ? 'GPU path'
      : `${handle.wasmThreads} WASM thread(s), proxy=${handle.wasmProxy}`;
  const failures =
    handle.initializationFailures.length === 0
      ? 'no fallback failures'
      : `fallback: ${handle.initializationFailures.join(' | ')}`;
  return `${threadDetail}; crossOriginIsolated=${window.crossOriginIsolated}; ${failures}`;
}

function setBadge(element: HTMLElement, text: string, state: 'waiting' | 'ready' | 'active' | 'error'): void {
  element.textContent = text;
  element.className = `badge badge-${state}`;
}

function showFatalError(message: string): void {
  elements.fatalError.textContent = message;
  elements.fatalError.classList.remove('hidden');
}

function hideFatalError(): void {
  elements.fatalError.textContent = '';
  elements.fatalError.classList.add('hidden');
}

function waitForVideoMetadata(video: HTMLVideoElement): Promise<void> {
  if (video.readyState >= HTMLMediaElement.HAVE_METADATA && video.videoWidth > 0) {
    return Promise.resolve();
  }
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error('Timed out waiting for camera metadata.'));
    }, 10_000);
    const onLoaded = () => {
      cleanup();
      resolve();
    };
    const onError = () => {
      cleanup();
      reject(video.error ?? new Error('Video metadata loading failed.'));
    };
    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener('loadedmetadata', onLoaded);
      video.removeEventListener('error', onError);
    };
    video.addEventListener('loadedmetadata', onLoaded, { once: true });
    video.addEventListener('error', onError, { once: true });
  });
}

function appendDiagnosticSection(current: string, key: string, value: unknown): string {
  try {
    const parsed = JSON.parse(current) as Record<string, unknown>;
    parsed[key] = value;
    return JSON.stringify(parsed, null, 2);
  } catch {
    return `${current}\n\n${key}:\n${JSON.stringify(value, null, 2)}`;
  }
}

function appendCell(row: HTMLTableRowElement, text: string): void {
  const cell = document.createElement('td');
  cell.textContent = text;
  row.append(cell);
}

function providerOrder(provider: string): number {
  switch (provider) {
    case 'webgl':
      return 0;
    case 'wasm-simd':
      return 1;
    case 'wasm-threaded':
      return 2;
    default:
      return 99;
  }
}

function formatCameraFrame(video: HTMLVideoElement): string {
  if (video.videoWidth <= 0 || video.videoHeight <= 0) {
    return '—';
  }
  const crop = centeredSquareCrop(video.videoWidth, video.videoHeight);
  return `${video.videoWidth} × ${video.videoHeight} → ${crop.size}² center`;
}

function formatMilliseconds(value: number | undefined): string {
  return value === undefined ? '—' : `${value.toFixed(1)} ms`;
}

function formatSignedPercent(value: number): string {
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(1)}%`;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function requiredElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) {
    throw new Error(`Missing #${id} element.`);
  }
  return element as T;
}

function requireCanvas2dContext(
  context: CanvasRenderingContext2D | null,
  label: string,
): CanvasRenderingContext2D {
  if (context === null) {
    throw new Error(`${label} 2D context is unavailable.`);
  }
  return context;
}
