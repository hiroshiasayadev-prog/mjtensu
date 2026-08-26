import { registerSW } from 'virtual:pwa-register';
import './styles.css';
import {
  CaptureUploadError,
  fetchNextTask,
  fetchOverview,
  undoLastCapture,
  uploadCapture,
} from './api';
import { buildComposite, canvasToBlob, mapDetections } from './composite';
import {
  calculateVideoContainGeometry,
  fullSourceRect,
  loadCaptureLayout,
  normalizeRect,
  sourceRectToDisplay,
} from './layout';
import {
  INPUT_SIZE,
  NMS_IOU_THRESHOLD,
  createInputBuffer,
  decodeNanoDetOutput,
  preprocessComposite,
} from './nanodet';
import {
  countPendingCaptures,
  deletePendingCapture,
  listPendingCaptures,
  putPendingCapture,
} from './pending';
import {
  initializeRuntime,
  providerChoiceFromLocation,
  type RuntimeHandle,
} from './runtime';
import { TelemetrySeries } from './telemetry';
import type {
  CampaignOverview,
  CatalogCaptureDraft,
  CatalogCaptureManifest,
  CatalogCaptureTask,
  CatalogRow,
  CaptureLayoutDocument,
  DetectionRecord,
  ModelMetadata,
  PendingCatalogCapture,
  Rect,
  RegionKey,
} from './types';

const CAMPAIGN_ID = 'tile-catalog-warm-4-v2';
const MODEL_URL = new URL('models/nanodet-plus-m-320-tile-catalog.onnx', document.baseURI).href;
const MODEL_METADATA_URL = new URL(
  'models/nanodet-plus-m-320-tile-catalog.metadata.json',
  document.baseURI,
).href;
const DETECTOR_INTERVAL_MS = 120;

const app = document.querySelector<HTMLDivElement>('#app');
if (app === null) throw new Error('Missing #app root.');

let layout!: CaptureLayoutDocument;
let modelMetadata!: ModelMetadata;
let runtimePromise!: Promise<RuntimeHandle>;
let runtime: RuntimeHandle | null = null;
let overview: CampaignOverview | null = null;
let task: CatalogCaptureTask | null = null;
let currentDraft: CatalogCaptureDraft | null = null;
let video: HTMLVideoElement | null = null;
let overlay: HTMLCanvasElement | null = null;
let stream: MediaStream | null = null;
let track: MediaStreamTrack | null = null;
let detectorTimer: number | null = null;
let inferenceBusy = false;
let activeInferencePromise: Promise<void> | null = null;
let latestFrameRequested = false;
let captureSessionGeneration = 0;
let confidenceThreshold = 0.25;
let latestDetections: DetectionRecord[] = [];
const inputBuffer = createInputBuffer();
const telemetry = new TelemetrySeries();

void boot();

async function boot(): Promise<void> {
  configureServiceWorker();
  renderLoading('牌カタログ撮影を準備中');
  try {
    [layout, modelMetadata] = await Promise.all([
      loadCaptureLayout(),
      fetchJson<ModelMetadata>(MODEL_METADATA_URL),
    ]);
    runtimePromise = initializeRuntime(providerChoiceFromLocation(), MODEL_URL).then(async (handle) => {
      runtime = handle;
      await warmUp(handle);
      return handle;
    });
    void runtimePromise.catch((error) => console.error('[runtime-init-failed]', error));
    await refreshCampaign();
  } catch (error) {
    renderFatal(error);
  }
}

function configureServiceWorker(): void {
  if (import.meta.env.PROD) {
    registerSW({ immediate: true });
    return;
  }
  void navigator.serviceWorker?.getRegistrations().then((registrations) =>
    Promise.all(registrations.map((registration) => registration.unregister())),
  );
}

async function refreshCampaign(): Promise<void> {
  const [nextOverview, nextTask, pendingCount] = await Promise.all([
    fetchOverview(CAMPAIGN_ID),
    fetchNextTask(CAMPAIGN_ID),
    countPendingCaptures(),
  ]);
  overview = nextOverview;
  task = nextTask;
  renderInstruction(pendingCount);
}

function renderLoading(message: string): void {
  app.innerHTML = `
    <main class="center-screen">
      <div class="loading-card"><div class="spinner"></div><p>${escapeHtml(message)}</p></div>
    </main>`;
}

function renderFatal(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  app.innerHTML = `
    <main class="center-screen">
      <section class="fatal-card">
        <h1>起動失敗</h1>
        <pre>${escapeHtml(message)}</pre>
        <button id="reload-button">Reload</button>
      </section>
    </main>`;
  requireElement<HTMLButtonElement>('reload-button').onclick = () => window.location.reload();
}

function renderInstruction(pendingCount: number): void {
  stopCamera();
  releaseDraftUrl();
  currentDraft = null;

  const currentOverview = overview;
  const currentTask = task;
  if (currentOverview === null) throw new Error('Campaign overview is unavailable.');
  if (currentTask === null) {
    app.innerHTML = `
      <main class="center-screen">
        <section class="campaign-card complete-card">
          <p class="eyebrow">TILE CATALOG / WARM LIGHT</p>
          <h1>4条件の撮影完了 🎉</h1>
          <p>PCのannotation toolでNanoDet候補を修正し、牌ラベルを確定する。</p>
          <div class="button-row">
            <button id="undo-last-button" class="danger-secondary" ${currentOverview.completedTasks === 0 ? 'disabled' : ''}>直前を撮り直す</button>
            <button id="retry-pending-button" class="secondary" ${pendingCount === 0 ? 'disabled' : ''}>未送信 ${pendingCount}件を再送</button>
          </div>
        </section>
      </main>`;
    requireElement<HTMLButtonElement>('undo-last-button').onclick = () => void undoLatestSavedCapture();
    requireElement<HTMLButtonElement>('retry-pending-button').onclick = () => void retryPending();
    return;
  }

  app.innerHTML = `
    <main class="instruction-screen">
      <header class="campaign-header">
        <div>
          <p class="eyebrow">TILE CATALOG / WARM LIGHT</p>
          <h1>撮影 ${currentTask.taskOrder + 1} / ${currentOverview.totalTasks}</h1>
        </div>
        <div class="progress-block">
          <strong>${currentOverview.completedTasks} / ${currentOverview.totalTasks}</strong>
          <span>37牌を並べたまま · 未送信 ${pendingCount}</span>
        </div>
      </header>

      <section class="catalog-layout-card">
        ${currentTask.catalogRows.map(renderCatalogRow).join('')}
      </section>

      <section class="condition-card">
        <div>
          <p class="eyebrow">今回の条件</p>
          <h2>${escapeHtml(currentTask.environment.label)}</h2>
        </div>
        <p>${escapeHtml(currentTask.environment.instruction)}</p>
      </section>

      <aside class="placement-note">
        37牌が全部写っていればOK。スマホのbboxは目視確認用で、漏れや余分はPC側で直す。
      </aside>

      <footer class="instruction-actions">
        <button id="undo-last-button" class="danger-secondary" ${currentOverview.completedTasks === 0 ? 'disabled' : ''}>直前を撮り直す</button>
        <button id="retry-pending-button" class="secondary" ${pendingCount === 0 ? 'disabled' : ''}>未送信を再送</button>
        <button id="start-capture-button" class="primary">この条件にした → 撮影</button>
      </footer>
    </main>`;

  requireElement<HTMLButtonElement>('start-capture-button').onclick = () => void enterCapture();
  requireElement<HTMLButtonElement>('undo-last-button').onclick = () => void undoLatestSavedCapture();
  requireElement<HTMLButtonElement>('retry-pending-button').onclick = () => void retryPending();
}

function renderCatalogRow(row: CatalogRow): string {
  return `
    <div class="catalog-row">
      <strong>${escapeHtml(row.label)}</strong>
      <div class="tile-sequence">
        ${row.tiles.map((tile, index) => `
          <span class="tile-chip ${tile.startsWith('red5') ? 'red-tile' : ''}">
            <small>${index + 1}</small>${escapeHtml(tileLabel(tile))}
          </span>`).join('')}
      </div>
    </div>`;
}

async function undoLatestSavedCapture(): Promise<void> {
  if (!window.confirm('このカタログcampaignで最後に保存した1枚を削除して撮り直します。')) return;
  renderLoading('直前の保存を取り消し中');
  try {
    await undoLastCapture(CAMPAIGN_ID);
    await refreshCampaign();
  } catch (error) {
    renderFatal(error);
  }
}

async function enterCapture(): Promise<void> {
  if (task === null) return;
  if (!isLandscapeViewport()) {
    renderRotatePrompt();
    return;
  }
  renderCaptureScreen();
  try {
    runtime = await runtimePromise;
    await startCamera();
    startDetector();
  } catch (error) {
    showCaptureError(error);
  }
}

function renderRotatePrompt(): void {
  app.innerHTML = `
    <main class="center-screen">
      <section class="campaign-card">
        <p class="eyebrow">撮影準備</p>
        <h1>iPhoneを横持ちにしてください</h1>
        <p>横向きになったら自動的にカメラを起動する。</p>
        <button id="rotate-back-button" class="secondary">配置指示へ戻る</button>
      </section>
    </main>`;
  const onResize = (): void => {
    if (!isLandscapeViewport()) return;
    window.removeEventListener('resize', onResize);
    void enterCapture();
  };
  window.addEventListener('resize', onResize);
  requireElement<HTMLButtonElement>('rotate-back-button').onclick = () => {
    window.removeEventListener('resize', onResize);
    void refreshCampaign().catch(renderFatal);
  };
}

function renderCaptureScreen(): void {
  captureSessionGeneration += 1;
  telemetry.clear();
  app.innerHTML = `
    <main class="capture-screen">
      <video id="camera-video" autoplay muted playsinline></video>
      <canvas id="detection-overlay"></canvas>
      <header class="capture-topbar">
        <button id="back-button" class="icon-button">←</button>
        <div class="capture-condition">
          <strong>${escapeHtml(task?.environment.label ?? '')}</strong>
          <span id="runtime-label">runtime loading…</span>
        </div>
        <div class="capture-counts" id="capture-counts">候補 —</div>
      </header>
      <aside class="telemetry-strip" id="telemetry-strip">—</aside>
      <div class="orientation-warning hidden" id="orientation-warning">iPhoneを横持ちにしてください</div>
      <div class="capture-error hidden" id="capture-error"></div>
      <footer class="capture-controls">
        <label class="threshold-control">conf <span id="threshold-value">${confidenceThreshold.toFixed(2)}</span>
          <input id="threshold-input" type="range" min="0.05" max="0.80" step="0.05" value="${confidenceThreshold}">
        </label>
        <button id="shutter-button" class="shutter-button" disabled aria-label="撮影"><span></span></button>
        <span class="capture-hint">全37牌が写っているかだけ確認 · bboxは参考</span>
      </footer>
    </main>`;

  video = requireElement<HTMLVideoElement>('camera-video');
  overlay = requireElement<HTMLCanvasElement>('detection-overlay');
  requireElement<HTMLButtonElement>('back-button').onclick = () => {
    void refreshCampaign().catch(renderFatal);
  };
  requireElement<HTMLInputElement>('threshold-input').oninput = (event) => {
    confidenceThreshold = Number((event.currentTarget as HTMLInputElement).value);
    requireElement('threshold-value').textContent = confidenceThreshold.toFixed(2);
  };
  requireElement<HTMLButtonElement>('shutter-button').onclick = () => void takeCapture();
  window.addEventListener('resize', refreshGeometry);
}

async function startCamera(): Promise<void> {
  const currentVideo = video;
  if (currentVideo === null) throw new Error('Camera video element is missing.');
  const mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: { ideal: 'environment' },
      width: { ideal: 1920 },
      height: { ideal: 1080 },
      frameRate: { ideal: 30, max: 30 },
    },
  });
  stream = mediaStream;
  track = mediaStream.getVideoTracks()[0] ?? null;
  if (track === null) throw new Error('No video track was returned.');
  currentVideo.srcObject = mediaStream;
  await currentVideo.play();
  await waitForVideo(currentVideo);
  refreshGeometry();
  requireElement<HTMLButtonElement>('shutter-button').disabled = !isLandscapeViewport();
  requireElement('runtime-label').textContent = runtime === null
    ? 'runtime unavailable'
    : `${runtime.provider}${runtime.wasmThreads > 0 ? ` / ${runtime.wasmThreads} threads` : ''}`;
}

function refreshGeometry(): void {
  const warning = document.getElementById('orientation-warning');
  const shutter = document.getElementById('shutter-button') as HTMLButtonElement | null;
  if (!isLandscapeViewport()) {
    warning?.classList.remove('hidden');
    if (shutter !== null) shutter.disabled = true;
    clearDetectionOverlay();
    return;
  }
  warning?.classList.add('hidden');
  if (shutter !== null && track !== null) shutter.disabled = false;
  drawOverlay();
}

function startDetector(): void {
  stopDetector();
  detectorTimer = window.setInterval(() => requestLiveDetection(), DETECTOR_INTERVAL_MS);
  requestLiveDetection();
}

function stopDetector(): void {
  if (detectorTimer !== null) window.clearInterval(detectorTimer);
  detectorTimer = null;
  latestFrameRequested = false;
}

function requestLiveDetection(): void {
  if (inferenceBusy) {
    telemetry.drop();
    latestFrameRequested = true;
    return;
  }
  const promise = runLiveDetection();
  activeInferencePromise = promise;
  void promise.finally(() => {
    if (activeInferencePromise === promise) activeInferencePromise = null;
  });
}

async function runLiveDetection(): Promise<void> {
  const generation = captureSessionGeneration;
  const currentVideo = video;
  const activeRuntime = runtime;
  if (currentVideo === null || activeRuntime === null) return;
  if (currentVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  inferenceBusy = true;
  latestFrameRequested = false;
  try {
    const geometry = calculateVideoContainGeometry(currentVideo);
    const sourceRect = fullSourceRect(geometry);
    const sourceRects = fullFrameSourceRects(sourceRect);
    const compositionStarted = performance.now();
    const { compositeCanvas } = buildComposite(
      currentVideo,
      sourceRects,
      catalogEnabledRegions(),
      layout,
      false,
    );
    const detections = await inferComposite(
      compositeCanvas,
      sourceRects,
      activeRuntime,
      performance.now() - compositionStarted,
    );
    if (generation !== captureSessionGeneration) return;
    latestDetections = detections;
    clearCaptureError();
    drawOverlay();
  } catch (error) {
    if (generation === captureSessionGeneration) showCaptureError(error);
  } finally {
    inferenceBusy = false;
    if (generation === captureSessionGeneration && latestFrameRequested) {
      queueMicrotask(requestLiveDetection);
    }
  }
}

async function inferComposite(
  compositeCanvas: HTMLCanvasElement,
  sourceRects: Record<RegionKey, Rect>,
  activeRuntime: RuntimeHandle,
  compositionMs = 0,
): Promise<DetectionRecord[]> {
  const preprocessStarted = performance.now();
  const inputData = preprocessComposite(compositeCanvas, inputBuffer);
  const tensor = activeRuntime.createFloat32Tensor(inputData, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const preprocessMs = compositionMs + (performance.now() - preprocessStarted);
  const inputName = activeRuntime.session.inputNames[0];
  const outputName = activeRuntime.session.outputNames[0];
  if (inputName === undefined || outputName === undefined) throw new Error('Unexpected ONNX IO contract.');

  const inferenceStarted = performance.now();
  try {
    const outputs = await activeRuntime.session.run({ [inputName]: tensor });
    try {
      const inferenceMs = performance.now() - inferenceStarted;
      const output = outputs[outputName];
      if (output === undefined) throw new Error(`Missing output ${outputName}.`);
      const decodeStarted = performance.now();
      const decoded = decodeNanoDetOutput(output, confidenceThreshold);
      const records = mapDetections(decoded, sourceRects, catalogEnabledRegions(), layout);
      const decodeMs = performance.now() - decodeStarted;
      telemetry.add({
        preprocessMs,
        inferenceMs,
        decodeMs,
        endToEndMs: preprocessMs + inferenceMs + decodeMs,
        completedAt: performance.now(),
        detectionCount: records.length,
      });
      return records;
    } finally {
      for (const output of Object.values(outputs)) output.dispose();
    }
  } finally {
    tensor.dispose();
  }
}

function drawOverlay(): void {
  const currentOverlay = overlay;
  const currentVideo = video;
  if (currentOverlay === null || currentVideo === null) return;
  if (currentVideo.videoWidth <= 0 || currentVideo.videoHeight <= 0) return;

  const dpr = window.devicePixelRatio || 1;
  currentOverlay.width = Math.round(window.innerWidth * dpr);
  currentOverlay.height = Math.round(window.innerHeight * dpr);
  currentOverlay.style.width = `${window.innerWidth}px`;
  currentOverlay.style.height = `${window.innerHeight}px`;
  const context = currentOverlay.getContext('2d');
  if (context === null) return;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, window.innerWidth, window.innerHeight);

  const geometry = calculateVideoContainGeometry(currentVideo);
  const frame = sourceRectToDisplay(fullSourceRect(geometry), geometry);
  context.strokeStyle = 'rgba(87, 227, 137, 0.9)';
  context.lineWidth = 2;
  context.strokeRect(frame.x, frame.y, frame.width, frame.height);

  for (const detection of latestDetections) {
    if (detection.original === null) continue;
    const rect = sourceRectToDisplay(detection.original, geometry);
    context.strokeStyle = '#ffcf5c';
    context.lineWidth = Math.max(2, window.innerWidth / 550);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
  }
  updateCaptureMetrics();
}

function updateCaptureMetrics(): void {
  const count = latestDetections.filter((detection) => detection.region === 'melds').length;
  const countElement = document.getElementById('capture-counts');
  if (countElement !== null) countElement.textContent = `候補 ${count}`;
  const telemetryElement = document.getElementById('telemetry-strip');
  if (telemetryElement === null) return;
  const snapshot = telemetry.snapshot();
  telemetryElement.textContent = [
    `${runtime?.provider ?? '—'}`,
    `infer ${snapshot.inferenceMs.toFixed(1)}ms`,
    `e2e ${snapshot.endToEndMs.toFixed(1)}ms`,
    `${snapshot.effectiveHz.toFixed(1)}Hz`,
  ].join(' · ');
}

async function takeCapture(): Promise<void> {
  if (!isLandscapeViewport()) return;
  const generation = captureSessionGeneration;
  const capturedAt = new Date().toISOString();
  const currentVideo = video;
  const currentTask = task;
  const currentTrack = track;
  const activeRuntime = runtime;
  if (currentVideo === null || currentTask === null || currentTrack === null || activeRuntime === null) return;

  const shutter = requireElement<HTMLButtonElement>('shutter-button');
  const backButton = requireElement<HTMLButtonElement>('back-button');
  shutter.disabled = true;
  backButton.disabled = true;
  stopDetector();
  const liveInference = activeInferencePromise;
  try {
    const geometry = calculateVideoContainGeometry(currentVideo);
    const sourceRect = fullSourceRect(geometry);
    const sourceRects = fullFrameSourceRects(sourceRect);
    const displayRect = sourceRectToDisplay(sourceRect, geometry);
    const originalCanvas = document.createElement('canvas');
    originalCanvas.width = currentVideo.videoWidth;
    originalCanvas.height = currentVideo.videoHeight;
    const originalContext = originalCanvas.getContext('2d', { alpha: false });
    if (originalContext === null) throw new Error('Original canvas is unavailable.');
    originalContext.drawImage(currentVideo, 0, 0, originalCanvas.width, originalCanvas.height);

    const compositionStarted = performance.now();
    const built = buildComposite(
      originalCanvas,
      sourceRects,
      catalogEnabledRegions(),
      layout,
      false,
    );
    const compositionMs = performance.now() - compositionStarted;

    if (liveInference !== null) await liveInference;
    if (generation !== captureSessionGeneration) return;
    const detections = await inferComposite(
      built.compositeCanvas,
      sourceRects,
      activeRuntime,
      compositionMs,
    );
    if (generation !== captureSessionGeneration) return;
    const captureDetections = detections.map((detection) => ({
      ...detection,
      preview: detection.original === null
        ? null
        : sourceRectToDisplay(detection.original, geometry),
    }));
    const previewCanvas = annotateOriginal(originalCanvas, captureDetections);

    const [originalBlob, compositeBlob, previewBlob] = await Promise.all([
      canvasToBlob(originalCanvas, 'image/jpeg', 0.98),
      canvasToBlob(built.compositeCanvas, 'image/png'),
      canvasToBlob(previewCanvas, 'image/jpeg', 0.92),
    ]);

    const zeroRect = { x: 0, y: 0, width: 0, height: 0 };
    const manifest: CatalogCaptureManifest = {
      uploadClientId: crypto.randomUUID(),
      taskId: currentTask.id,
      campaignId: currentTask.campaignId,
      capturedAt,
      original: { width: originalCanvas.width, height: originalCanvas.height },
      preview: {
        width: window.innerWidth,
        height: window.innerHeight,
        devicePixelRatio: window.devicePixelRatio || 1,
        videoElement: {
          x: geometry.element.x,
          y: geometry.element.y,
          width: geometry.element.width,
          height: geometry.element.height,
        },
        sourceToDisplayScale: geometry.scale,
        sourceDisplayOffsetX: geometry.offsetX,
        sourceDisplayOffsetY: geometry.offsetY,
      },
      model: modelMetadata,
      layoutVersion: layout.id,
      confidenceThreshold,
      nmsIouThreshold: NMS_IOU_THRESHOLD,
      provider: `${activeRuntime.provider}-visual-only`,
      camera: currentTrack.getSettings(),
      telemetry: {
        ...telemetry.snapshot(),
        visualCandidateCount: captureDetections.length,
      },
      regionRects: {
        completed_hand: {
          enabled: false,
          pixel: zeroRect,
          normalized: zeroRect,
          display: zeroRect,
        },
        dora_indicators: {
          enabled: false,
          pixel: zeroRect,
          normalized: zeroRect,
          display: zeroRect,
        },
        melds: {
          enabled: true,
          pixel: sourceRect,
          normalized: normalizeRect(sourceRect, originalCanvas.width, originalCanvas.height),
          display: displayRect,
        },
      },
      detections: [],
      catalog: {
        schemaVersion: 2,
        variantId: currentTask.environment.variantId,
        rows: currentTask.catalogRows,
        smartphoneDetector: 'visual-only',
        annotationDetector: 'pc-after-upload',
      },
    };

    currentDraft = {
      manifest,
      original: originalBlob,
      composite: compositeBlob,
      previewUrl: URL.createObjectURL(previewBlob),
    };
    stopCamera();
    renderReview();
  } catch (error) {
    if (generation !== captureSessionGeneration) return;
    showCaptureError(error);
    shutter.disabled = false;
    backButton.disabled = false;
    startDetector();
  }
}

function renderReview(): void {
  const draft = currentDraft;
  const currentTask = task;
  if (draft === null || currentTask === null) return;
  const candidateCount = Number(draft.manifest.telemetry.visualCandidateCount ?? 0);
  app.innerHTML = `
    <main class="review-screen simple-review">
      <header class="review-header">
        <div>
          <p class="eyebrow">撮影確認</p>
          <h1>${escapeHtml(currentTask.environment.label)}</h1>
        </div>
        <div class="review-counts">検出候補 ${candidateCount}</div>
      </header>
      <section class="review-grid single-review-grid">
        <figure class="review-main"><img src="${draft.previewUrl}" alt="original with detector candidates"><figcaption>原画＋NanoDet候補bbox</figcaption></figure>
      </section>
      <p class="review-note">37牌すべてが写っていて文字が読めれば保存。bboxの漏れ・余分・位置はPC側で修正する。</p>
      <footer class="review-actions">
        <button id="retake-button" class="secondary">撮り直す</button>
        <button id="save-button" class="primary">保存して次へ</button>
      </footer>
      <div id="save-status" class="save-status"></div>
    </main>`;
  requireElement<HTMLButtonElement>('retake-button').onclick = () => void discardDraftAndRetake();
  requireElement<HTMLButtonElement>('save-button').onclick = () => void saveDraft();
}

async function discardDraftAndRetake(): Promise<void> {
  const draft = currentDraft;
  if (draft === null) return;
  try {
    await deletePendingCapture(draft.manifest.uploadClientId);
    releaseDraftUrl();
    currentDraft = null;
    await enterCapture();
  } catch (error) {
    requireElement('save-status').textContent = `撮り直し準備に失敗: ${errorMessage(error)}`;
  }
}

async function saveDraft(): Promise<void> {
  const draft = currentDraft;
  if (draft === null) return;
  const saveButton = requireElement<HTMLButtonElement>('save-button');
  saveButton.disabled = true;
  const status = requireElement('save-status');
  const pending: PendingCatalogCapture = {
    id: draft.manifest.uploadClientId,
    manifest: draft.manifest,
    original: draft.original,
    composite: draft.composite,
  };
  let persistedLocally = false;
  try {
    status.textContent = '端末へ一時保存中…';
    await putPendingCapture(pending);
    persistedLocally = true;
    status.textContent = 'Windowsへ送信中…';
    await uploadCapture(pending);
    await deletePendingCapture(pending.id);
    status.textContent = '保存完了';
    await refreshCampaign();
  } catch (error) {
    if (error instanceof CaptureUploadError && error.status === 409) {
      await deletePendingCapture(pending.id);
      status.textContent = 'この条件は既に保存済み。重複draftを破棄した。';
      await refreshCampaign();
      return;
    }
    status.textContent = persistedLocally
      ? `未送信として端末に保持: ${errorMessage(error)}`
      : `端末への一時保存に失敗: ${errorMessage(error)}`;
    saveButton.disabled = false;
  }
}

async function retryPending(): Promise<void> {
  renderLoading('未送信captureを再送中');
  const captures = await listPendingCaptures();
  const failures: string[] = [];
  for (const capture of captures) {
    try {
      await uploadCapture(capture);
      await deletePendingCapture(capture.id);
    } catch (error) {
      if (error instanceof CaptureUploadError && error.status === 409) {
        await deletePendingCapture(capture.id);
        continue;
      }
      failures.push(`${capture.id}: ${errorMessage(error)}`);
    }
  }
  if (failures.length > 0) {
    app.innerHTML = `
      <main class="center-screen">
        <section class="fatal-card">
          <h1>一部再送失敗</h1>
          <pre>${escapeHtml(failures.join('\n'))}</pre>
          <button id="continue-button">戻る</button>
        </section>
      </main>`;
    requireElement<HTMLButtonElement>('continue-button').onclick = () => void refreshCampaign();
    return;
  }
  await refreshCampaign();
}

function stopCamera(): void {
  captureSessionGeneration += 1;
  stopDetector();
  window.removeEventListener('resize', refreshGeometry);
  if (stream !== null) for (const streamTrack of stream.getTracks()) streamTrack.stop();
  stream = null;
  track = null;
  if (video !== null) video.srcObject = null;
  video = null;
  overlay = null;
  latestDetections = [];
}

function annotateOriginal(source: HTMLCanvasElement, detections: DetectionRecord[]): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext('2d');
  if (context === null) throw new Error('Review canvas is unavailable.');
  context.drawImage(source, 0, 0);
  context.lineWidth = Math.max(3, source.width / 500);
  for (const detection of detections) {
    const rect = detection.original;
    if (rect === null) continue;
    context.strokeStyle = '#ffcf5c';
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
  }
  return canvas;
}

function catalogEnabledRegions(): Record<RegionKey, boolean> {
  return { completed_hand: false, dora_indicators: false, melds: true };
}

function fullFrameSourceRects(sourceRect: Rect): Record<RegionKey, Rect> {
  const zero = { x: 0, y: 0, width: 0, height: 0 };
  return {
    completed_hand: zero,
    dora_indicators: zero,
    melds: sourceRect,
  };
}

function clearDetectionOverlay(): void {
  const currentOverlay = overlay;
  if (currentOverlay === null) return;
  currentOverlay.getContext('2d')?.clearRect(0, 0, currentOverlay.width, currentOverlay.height);
}

function isLandscapeViewport(): boolean {
  return window.innerWidth > window.innerHeight;
}

function clearCaptureError(): void {
  const element = document.getElementById('capture-error');
  if (element !== null) {
    element.textContent = '';
    element.classList.add('hidden');
  }
}

function showCaptureError(error: unknown): void {
  const element = document.getElementById('capture-error');
  console.error('[capture-error]', error);
  if (element !== null) {
    element.textContent = errorMessage(error);
    element.classList.remove('hidden');
  }
}

function releaseDraftUrl(): void {
  if (currentDraft === null) return;
  URL.revokeObjectURL(currentDraft.previewUrl);
}

async function warmUp(handle: RuntimeHandle): Promise<void> {
  const inputName = handle.session.inputNames[0];
  if (inputName === undefined) throw new Error('Model has no input.');
  const zeros = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  for (let pass = 0; pass < 2; pass += 1) {
    const input = handle.createFloat32Tensor(zeros, [1, 3, INPUT_SIZE, INPUT_SIZE]);
    try {
      const outputs = await handle.session.run({ [inputName]: input });
      for (const output of Object.values(outputs)) output.dispose();
    } finally {
      input.dispose();
    }
  }
}

function waitForVideo(target: HTMLVideoElement): Promise<void> {
  if (target.videoWidth > 0 && target.videoHeight > 0) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => reject(new Error('Camera metadata timed out.')), 10_000);
    target.addEventListener('loadedmetadata', () => {
      window.clearTimeout(timeout);
      resolve();
    }, { once: true });
  });
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

function tileLabel(tile: string): string {
  const names: Record<string, string> = {
    east: '東', south: '南', west: '西', north: '北',
    white: '白', green: '發', red: '中',
    red5m: '赤5m', red5p: '赤5p', red5s: '赤5s',
  };
  return names[tile] ?? tile;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function requireElement<T extends HTMLElement = HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (element === null) throw new Error(`Missing #${id}.`);
  return element as T;
}

function escapeHtml(value: string): string {
  return value.replace(/[&<>'"]/g, (character) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
  })[character] ?? character);
}
