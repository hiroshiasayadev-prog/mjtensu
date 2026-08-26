import { registerSW } from 'virtual:pwa-register';
import './styles.css';
import {
  CaptureUploadError,
  fetchNextTask,
  fetchOverview,
  undoLastCapture,
  uploadCapture,
} from './api';
import { buildComposite, canvasToBlob, mapDetections, regionKeys } from './composite';
import {
  calculateVideoCoverGeometry,
  computeDisplayRegionRects,
  displayRectToSource,
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
import { createMeldStack, createTileStrip } from './tiles';
import type {
  CampaignOverview,
  CaptureDraft,
  CaptureLayoutDocument,
  CaptureManifest,
  CaptureTask,
  DetectionRecord,
  ModelMetadata,
  PendingCapture,
  Rect,
  RegionKey,
} from './types';

const CAMPAIGN_ID = 'initial-120';
const MODEL_URL = new URL('models/nanodet-plus-m-320-composite-augmented.onnx', document.baseURI).href;
const MODEL_METADATA_URL = new URL(
  'models/nanodet-plus-m-320-composite-augmented.metadata.json',
  document.baseURI,
).href;
const DETECTOR_INTERVAL_MS = 100;

const app = document.querySelector<HTMLDivElement>('#app');
if (app === null) throw new Error('Missing #app root.');

let layout!: CaptureLayoutDocument;
let modelMetadata!: ModelMetadata;
let runtimePromise!: Promise<RuntimeHandle>;
let runtime: RuntimeHandle | null = null;
let overview: CampaignOverview | null = null;
let task: CaptureTask | null = null;
let currentDraft: CaptureDraft | null = null;
let video: HTMLVideoElement | null = null;
let overlay: HTMLCanvasElement | null = null;
let stream: MediaStream | null = null;
let track: MediaStreamTrack | null = null;
let detectorTimer: number | null = null;
let inferenceBusy = false;
let activeInferencePromise: Promise<void> | null = null;
let latestFrameRequested = false;
let captureSessionGeneration = 0;
let confidenceThreshold = 0.3;
let displayRects: Record<RegionKey, Rect> | null = null;
let sourceRects: Record<RegionKey, Rect> | null = null;
let latestDetections: DetectionRecord[] = [];
const inputBuffer = createInputBuffer();
const telemetry = new TelemetrySeries();

void boot();

async function boot(): Promise<void> {
  configureServiceWorker();
  renderLoading('capture layoutとmodelを読み込み中');
  try {
    [layout, modelMetadata] = await Promise.all([
      loadCaptureLayout(),
      fetchJson<ModelMetadata>(MODEL_METADATA_URL),
    ]);
    runtimePromise = initializeRuntime(providerChoiceFromLocation(), MODEL_URL).then((handle) => {
      runtime = handle;
      return warmUp(handle).then(() => handle);
    });
    void runtimePromise.catch((error) => {
      console.error('[runtime-init-failed]', error);
    });
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
  app.innerHTML = `<main class="center-screen"><div class="loading-card"><div class="spinner"></div><p>${escapeHtml(message)}</p></div></main>`;
}

function renderFatal(error: unknown): void {
  const message = error instanceof Error ? error.message : String(error);
  app.innerHTML = `<main class="center-screen"><section class="fatal-card"><h1>起動失敗</h1><pre>${escapeHtml(message)}</pre><button id="reload-button">Reload</button></section></main>`;
  requireElement<HTMLButtonElement>('reload-button').onclick = () => window.location.reload();
}

function renderInstruction(pendingCount: number): void {
  stopCamera();
  releaseDraftUrls();
  currentDraft = null;

  const currentOverview = overview;
  const currentTask = task;
  if (currentOverview === null) throw new Error('Campaign overview is unavailable.');
  if (currentTask === null) {
    app.innerHTML = `
      <main class="instruction-screen complete-screen">
        <section class="campaign-card">
          <p class="eyebrow">${escapeHtml(currentOverview.name)}</p>
          <h1>全task完了 🎉</h1>
          <p>${currentOverview.completedTasks} / ${currentOverview.totalTasks} captures</p>
          <div class="complete-actions">
            <button id="undo-last-button" class="danger-secondary" ${currentOverview.completedTasks === 0 ? 'disabled' : ''}>直前の保存を取り消す</button>
            <button id="retry-pending-button" ${pendingCount === 0 ? 'disabled' : ''}>未送信 ${pendingCount}件を再送</button>
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
          <p class="eyebrow">${escapeHtml(currentOverview.name)}</p>
          <h1>配置 ${currentTask.layoutOrdinal + 1} / ${currentOverview.totalLayouts}</h1>
        </div>
        <div class="progress-block">
          <strong>${currentOverview.completedTasks} / ${currentOverview.totalTasks}</strong>
          <span>牌種 ${coveredClassCount(currentOverview)} / ${Object.keys(currentOverview.coverage).length} · 未送信 ${pendingCount}</span>
        </div>
      </header>

      <section class="instruction-list">
        <article class="layout-card instruction-row">
          <div class="section-heading"><h2>手牌</h2><span>${currentTask.hand.length}枚・左→右</span></div>
          <div id="hand-instruction"></div>
        </article>

        <article class="layout-card instruction-row">
          <div class="section-heading"><h2>ドラちゃん</h2><span>表示牌 / 裏ドラ</span></div>
          <div class="dora-instruction">
            <div><label>表示牌</label><div id="dora-visible-instruction"></div></div>
            <div><label>裏ドラ</label><div id="dora-ura-instruction"></div></div>
          </div>
        </article>

        <article class="layout-card instruction-row meld-card">
          <div class="section-heading"><h2>副露</h2><span>${currentTask.melds.length} group・上→下</span></div>
          <div id="meld-instruction"></div>
        </article>
      </section>

      <aside class="environment-strip">
        <div><span>今回の環境</span><strong>${environmentLabel(currentTask)}</strong></div>
        <span>手牌 ${currentTask.expected.hand}・ドラ ${currentTask.expected.dora}・副露 ${currentTask.expected.meld}</span>
      </aside>

      <footer class="instruction-actions">
        <button id="undo-last-button" class="danger-secondary" ${currentOverview.completedTasks === 0 ? 'disabled' : ''}>直前の保存を取り消す</button>
        <button id="retry-pending-button" class="secondary" ${pendingCount === 0 ? 'disabled' : ''}>未送信を再送</button>
        <button id="start-capture-button" class="primary">配置できた → 撮影</button>
      </footer>
    </main>`;

  requireElement('hand-instruction').append(createTileStrip(currentTask.hand, 'instruction-tiles'));
  requireElement('dora-visible-instruction').append(createTileStrip(currentTask.dora.visible, 'instruction-tiles'));
  requireElement('dora-ura-instruction').append(createTileStrip(currentTask.dora.ura, 'instruction-tiles'));
  requireElement('meld-instruction').append(createMeldStack(currentTask.melds));
  requireElement<HTMLButtonElement>('start-capture-button').onclick = () => void enterCapture();
  requireElement<HTMLButtonElement>('undo-last-button').onclick = () => void undoLatestSavedCapture();
  requireElement<HTMLButtonElement>('retry-pending-button').onclick = () => void retryPending();
}

async function undoLatestSavedCapture(): Promise<void> {
  const confirmed = window.confirm(
    'このcampaignで最後に保存した写真を削除し、そのtaskを撮り直します。よろしいですか？',
  );
  if (!confirmed) return;

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
    <main class="center-screen rotate-screen">
      <section class="campaign-card rotate-card">
        <p class="eyebrow">撮影準備</p>
        <h1>iPhoneを横持ちにしてください</h1>
        <p>横向きになったら自動的にcameraを起動する。</p>
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
        <div>
          <strong>${task === null ? '' : environmentLabel(task)}</strong>
          <span id="runtime-label">runtime loading…</span>
        </div>
        <div class="capture-counts" id="capture-counts">—</div>
      </header>
      <aside class="telemetry-strip" id="telemetry-strip">—</aside>
      <div class="orientation-warning hidden" id="orientation-warning">iPhoneを横持ちにしてください</div>
      <div class="capture-error hidden" id="capture-error"></div>
      <footer class="capture-controls">
        <label class="threshold-control">conf <span id="threshold-value">${confidenceThreshold.toFixed(2)}</span>
          <input id="threshold-input" type="range" min="0.05" max="0.80" step="0.05" value="${confidenceThreshold}">
        </label>
        <button id="shutter-button" class="shutter-button" disabled aria-label="撮影"><span></span></button>
        <span class="capture-hint">bboxは参考表示のみ</span>
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
  const currentTrack = mediaStream.getVideoTracks()[0] ?? null;
  track = currentTrack;
  if (currentTrack === null) throw new Error('No video track was returned.');
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
  const currentVideo = video;
  if (currentVideo === null || currentVideo.videoWidth <= 0 || currentVideo.videoHeight <= 0) return;
  displayRects = computeDisplayRegionRects(window.innerWidth, window.innerHeight);
  const geometry = calculateVideoCoverGeometry(currentVideo);
  sourceRects = mapRecord(displayRects, (rect) => displayRectToSource(rect, geometry));
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
  const currentRuntime = runtime;
  const currentSourceRects = sourceRects;
  const currentTask = task;
  if (currentVideo === null || currentRuntime === null || currentSourceRects === null || currentTask === null) return;
  if (currentVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  inferenceBusy = true;
  latestFrameRequested = false;
  try {
    const enabled = enabledRegions(currentTask);
    const compositionStarted = performance.now();
    const { compositeCanvas } = buildComposite(currentVideo, currentSourceRects, enabled, layout, false);
    const compositionMs = performance.now() - compositionStarted;
    const detections = await inferComposite(
      compositeCanvas,
      currentSourceRects,
      enabled,
      currentRuntime,
      compositionMs,
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
  currentSourceRects: Record<RegionKey, Rect>,
  enabled: Record<RegionKey, boolean>,
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
      const records = mapDetections(decoded, currentSourceRects, enabled, layout);
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
  const currentDisplayRects = displayRects;
  const currentTask = task;
  if (currentOverlay === null || currentVideo === null || currentDisplayRects === null || currentTask === null) return;
  const dpr = window.devicePixelRatio || 1;
  currentOverlay.width = Math.round(window.innerWidth * dpr);
  currentOverlay.height = Math.round(window.innerHeight * dpr);
  currentOverlay.style.width = `${window.innerWidth}px`;
  currentOverlay.style.height = `${window.innerHeight}px`;
  const context = currentOverlay.getContext('2d');
  if (context === null) return;
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, window.innerWidth, window.innerHeight);

  const enabled = enabledRegions(currentTask);
  const names: Record<RegionKey, string> = {
    completed_hand: 'HAND',
    dora_indicators: 'DORA',
    melds: 'MELD',
  };
  for (const key of regionKeys()) {
    const rect = currentDisplayRects[key];
    context.lineWidth = 2;
    context.strokeStyle = enabled[key] ? '#57e389' : 'rgba(210, 216, 225, 0.35)';
    context.setLineDash(enabled[key] ? [] : [7, 5]);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.setLineDash([]);
    context.font = '700 13px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.fillStyle = enabled[key] ? '#b9fbcf' : '#aab1bd';
    context.fillText(names[key], rect.x + 5, rect.y + 5);
  }

  const geometry = calculateVideoCoverGeometry(currentVideo);
  for (const detection of latestDetections) {
    if (detection.original === null) continue;
    const rect = sourceRectToDisplay(detection.original, geometry);
    context.strokeStyle = regionColor(detection.region);
    context.lineWidth = Math.max(2, window.innerWidth / 500);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    const label = `${regionShort(detection.region)} ${detection.confidence.toFixed(2)}`;
    context.font = '600 11px ui-monospace, SFMono-Regular, Menlo, monospace';
    const width = context.measureText(label).width + 8;
    context.fillStyle = 'rgba(9, 11, 16, 0.82)';
    context.fillRect(rect.x, Math.max(0, rect.y - 16), width, 16);
    context.fillStyle = '#f5f7fa';
    context.fillText(label, rect.x + 4, Math.max(1, rect.y - 14));
  }
  updateCaptureMetrics();
}

function updateCaptureMetrics(): void {
  const currentTask = task;
  if (currentTask === null || document.getElementById('capture-counts') === null) return;
  const counts = countByRegion(latestDetections);
  requireElement('capture-counts').textContent = [
    `H ${counts.completed_hand}/${currentTask.expected.hand}`,
    `D ${counts.dora_indicators}/${currentTask.expected.dora}`,
    `M ${counts.melds}/${currentTask.expected.meld}`,
  ].join(' · ');
  const snapshot = telemetry.snapshot();
  requireElement('telemetry-strip').textContent = [
    `${runtime?.provider ?? '—'}`,
    `pre ${snapshot.preprocessMs.toFixed(1)}ms`,
    `infer ${snapshot.inferenceMs.toFixed(1)}ms`,
    `decode ${snapshot.decodeMs.toFixed(1)}ms`,
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
    refreshGeometry();
    const currentSourceRects = sourceRects;
    const currentDisplayRects = displayRects;
    if (currentSourceRects === null || currentDisplayRects === null) {
      throw new Error('Capture regions are unavailable.');
    }
    const previewGeometry = calculateVideoCoverGeometry(currentVideo);
    const previewWidth = window.innerWidth;
    const previewHeight = window.innerHeight;
    const previewDevicePixelRatio = window.devicePixelRatio || 1;
    const originalCanvas = document.createElement('canvas');
    originalCanvas.width = currentVideo.videoWidth;
    originalCanvas.height = currentVideo.videoHeight;
    const originalContext = originalCanvas.getContext('2d', { alpha: false });
    if (originalContext === null) throw new Error('Original canvas is unavailable.');
    originalContext.drawImage(currentVideo, 0, 0, originalCanvas.width, originalCanvas.height);

    const frozenRects = cloneRects(currentSourceRects);
    const frozenDisplayRects = cloneRects(currentDisplayRects);
    const enabled = enabledRegions(currentTask);
    const compositionStarted = performance.now();
    const built = buildComposite(originalCanvas, frozenRects, enabled, layout, true);
    const compositionMs = performance.now() - compositionStarted;

    if (liveInference !== null) await liveInference;
    if (generation !== captureSessionGeneration) return;
    const detections = await inferComposite(
      built.compositeCanvas,
      frozenRects,
      enabled,
      activeRuntime,
      compositionMs,
    );
    if (generation !== captureSessionGeneration) return;
    const captureDetections = detections.map((detection) => ({
      ...detection,
      preview: detection.original === null
        ? null
        : sourceRectToDisplay(detection.original, previewGeometry),
    }));
    const previewCanvas = annotateOriginal(originalCanvas, captureDetections);
    const compositePreviewCanvas = annotateComposite(built.compositeCanvas, captureDetections);

    const [originalBlob, compositeBlob, previewBlob, compositePreviewBlob] = await Promise.all([
      canvasToBlob(originalCanvas, 'image/jpeg', 0.98),
      canvasToBlob(built.compositeCanvas, 'image/png'),
      canvasToBlob(previewCanvas, 'image/jpeg', 0.92),
      canvasToBlob(compositePreviewCanvas, 'image/png'),
    ]);
    const regionBlobs: Partial<Record<RegionKey, Blob>> = {};
    for (const key of regionKeys()) {
      const canvas = built.regionCanvases[key];
      if (canvas !== undefined) regionBlobs[key] = await canvasToBlob(canvas, 'image/png');
    }

    const manifest: CaptureManifest = {
      uploadClientId: crypto.randomUUID(),
      taskId: currentTask.id,
      campaignId: currentTask.campaignId,
      capturedAt,
      original: { width: originalCanvas.width, height: originalCanvas.height },
      preview: {
        width: previewWidth,
        height: previewHeight,
        devicePixelRatio: previewDevicePixelRatio,
        videoElement: {
          x: previewGeometry.element.x,
          y: previewGeometry.element.y,
          width: previewGeometry.element.width,
          height: previewGeometry.element.height,
        },
        sourceToDisplayScale: previewGeometry.scale,
        sourceDisplayOffsetX: previewGeometry.offsetX,
        sourceDisplayOffsetY: previewGeometry.offsetY,
      },
      model: modelMetadata,
      layoutVersion: layout.id,
      confidenceThreshold,
      nmsIouThreshold: NMS_IOU_THRESHOLD,
      provider: activeRuntime.provider,
      camera: currentTrack.getSettings(),
      telemetry: telemetry.snapshot(),
      regionRects: mapRecord(frozenRects, (rect, key) => ({
        enabled: enabled[key],
        pixel: rect,
        normalized: normalizeRect(rect, originalCanvas.width, originalCanvas.height),
        display: frozenDisplayRects[key],
      })),
      detections: captureDetections,
    };

    if (generation !== captureSessionGeneration) return;
    currentDraft = {
      manifest,
      original: originalBlob,
      composite: compositeBlob,
      handCrop: regionBlobs.completed_hand ?? null,
      doraCrop: regionBlobs.dora_indicators ?? null,
      meldCrop: regionBlobs.melds ?? null,
      previewUrl: URL.createObjectURL(previewBlob),
      compositePreviewUrl: URL.createObjectURL(compositePreviewBlob),
      regionUrls: mapOptionalRecord(regionBlobs, (blob) => URL.createObjectURL(blob)),
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
  const counts = countByRegion(draft.manifest.detections);
  app.innerHTML = `
    <main class="review-screen">
      <header class="review-header">
        <div><p class="eyebrow">撮影確認</p><h1>${escapeHtml(environmentLabel(currentTask))}</h1></div>
        <div class="review-counts">H ${counts.completed_hand}/${currentTask.expected.hand} · D ${counts.dora_indicators}/${currentTask.expected.dora} · M ${counts.melds}/${currentTask.expected.meld}</div>
      </header>
      <section class="review-grid">
        <figure class="review-main"><img src="${draft.previewUrl}" alt="original with detections"><figcaption>原画＋現行bbox</figcaption></figure>
        <figure><img src="${draft.compositePreviewUrl}" alt="320 composite with detections"><figcaption>320 × 320 composite＋bbox</figcaption></figure>
        ${reviewRegionFigure(draft.regionUrls.completed_hand, '手牌crop')}
        ${reviewRegionFigure(draft.regionUrls.dora_indicators, 'ドラcrop')}
        ${reviewRegionFigure(draft.regionUrls.melds, '副露crop')}
      </section>
      <p class="review-note">見逃し・重複があってもそのまま保存してよい。後でPC側から再検出する。</p>
      <footer class="review-actions">
        <button id="retake-button" class="secondary">撮り直す</button>
        <button id="save-button" class="primary">保存</button>
      </footer>
      <div id="save-status" class="save-status"></div>
    </main>`;
  requireElement<HTMLButtonElement>('retake-button').onclick = () => {
    void discardDraftAndRetake(draft);
  };
  requireElement<HTMLButtonElement>('save-button').onclick = () => void saveDraft();
}

async function discardDraftAndRetake(draft: CaptureDraft): Promise<void> {
  const status = requireElement('save-status');
  try {
    await deletePendingCapture(draft.manifest.uploadClientId);
    releaseDraftUrls();
    currentDraft = null;
    await enterCapture();
  } catch (error) {
    status.textContent = `撮り直し前の一時保存削除に失敗: ${error instanceof Error ? error.message : String(error)}`;
  }
}

async function saveDraft(): Promise<void> {
  const draft = currentDraft;
  if (draft === null) return;
  const saveButton = requireElement<HTMLButtonElement>('save-button');
  saveButton.disabled = true;
  const status = requireElement('save-status');
  const pending: PendingCapture = {
    id: draft.manifest.uploadClientId,
    manifest: draft.manifest,
    original: draft.original,
    composite: draft.composite,
    handCrop: draft.handCrop,
    doraCrop: draft.doraCrop,
    meldCrop: draft.meldCrop,
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
      status.textContent = 'このtaskは既に保存済み。重複draftを破棄した。';
      await refreshCampaign();
      return;
    }
    status.textContent = persistedLocally
      ? `未送信として端末に保持: ${error instanceof Error ? error.message : String(error)}`
      : `端末への一時保存に失敗。まだ保存されていない: ${error instanceof Error ? error.message : String(error)}`;
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
      failures.push(`${capture.id}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
  if (failures.length > 0) {
    app.innerHTML = `<main class="center-screen"><section class="fatal-card"><h1>一部再送失敗</h1><pre>${escapeHtml(failures.join('\n'))}</pre><button id="continue-button">戻る</button></section></main>`;
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
  displayRects = null;
  sourceRects = null;
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
  context.font = `${Math.max(18, source.width / 55)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
  for (const detection of detections) {
    const rect = detection.original;
    if (rect === null) continue;
    context.strokeStyle = regionColor(detection.region);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    context.fillStyle = 'rgba(9, 11, 16, 0.82)';
    context.fillRect(rect.x, Math.max(0, rect.y - 28), 130, 28);
    context.fillStyle = '#ffffff';
    context.fillText(`${regionShort(detection.region)} ${detection.confidence.toFixed(2)}`, rect.x + 4, Math.max(0, rect.y - 5));
  }
  return canvas;
}

function annotateComposite(source: HTMLCanvasElement, detections: DetectionRecord[]): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext('2d');
  if (context === null) throw new Error('Composite review canvas is unavailable.');
  context.drawImage(source, 0, 0);
  context.lineWidth = 2;
  context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  for (const detection of detections) {
    const rect = detection.composite;
    context.strokeStyle = regionColor(detection.region);
    context.strokeRect(rect.x, rect.y, rect.width, rect.height);
    const label = `${regionShort(detection.region)} ${detection.confidence.toFixed(2)}`;
    const labelWidth = context.measureText(label).width + 5;
    const labelY = Math.max(0, rect.y - 12);
    context.fillStyle = 'rgba(9, 11, 16, 0.84)';
    context.fillRect(rect.x, labelY, labelWidth, 12);
    context.fillStyle = '#ffffff';
    context.fillText(label, rect.x + 2, labelY + 10);
  }
  return canvas;
}

function enabledRegions(currentTask: CaptureTask): Record<RegionKey, boolean> {
  return {
    completed_hand: currentTask.hand.length > 0,
    dora_indicators: currentTask.dora.visible.length + currentTask.dora.ura.length > 0,
    melds: currentTask.melds.length > 0,
  };
}

function countByRegion(detections: DetectionRecord[]): Record<RegionKey, number> {
  const counts: Record<RegionKey, number> = { completed_hand: 0, dora_indicators: 0, melds: 0 };
  for (const detection of detections) if (detection.region !== 'invalid') counts[detection.region] += 1;
  return counts;
}

function clearDetectionOverlay(): void {
  const currentOverlay = overlay;
  if (currentOverlay === null) return;
  const context = currentOverlay.getContext('2d');
  context?.clearRect(0, 0, currentOverlay.width, currentOverlay.height);
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
  const message = error instanceof Error ? error.message : String(error);
  console.error('[capture-error]', error);
  if (element !== null) {
    element.textContent = message;
    element.classList.remove('hidden');
  }
}

function coveredClassCount(currentOverview: CampaignOverview): number {
  return Object.values(currentOverview.coverage).filter((count) => count > 0).length;
}

function environmentLabel(currentTask: CaptureTask): string {
  const brightness = currentTask.environment.brightness === 'bright' ? '明るい' : '暗い';
  const shadow = currentTask.environment.shadow === 'none' ? '影なし' : '部分的な影あり';
  return `${brightness}・${shadow}`;
}

function regionColor(region: DetectionRecord['region']): string {
  switch (region) {
    case 'completed_hand': return '#57e389';
    case 'dora_indicators': return '#70b8ff';
    case 'melds': return '#f7ce46';
    case 'invalid': return '#e5484d';
  }
}

function regionShort(region: DetectionRecord['region']): string {
  switch (region) {
    case 'completed_hand': return 'H';
    case 'dora_indicators': return 'D';
    case 'melds': return 'M';
    case 'invalid': return 'X';
  }
}

function reviewRegionFigure(url: string | undefined, label: string): string {
  return url === undefined ? '' : `<figure><img src="${url}" alt="${escapeHtml(label)}"><figcaption>${escapeHtml(label)}</figcaption></figure>`;
}

function releaseDraftUrls(): void {
  if (currentDraft === null) return;
  URL.revokeObjectURL(currentDraft.previewUrl);
  URL.revokeObjectURL(currentDraft.compositePreviewUrl);
  for (const url of Object.values(currentDraft.regionUrls)) if (url !== undefined) URL.revokeObjectURL(url);
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

function mapRecord<T>(
  source: Record<RegionKey, Rect>,
  transform: (rect: Rect, key: RegionKey) => T,
): Record<RegionKey, T> {
  return {
    completed_hand: transform(source.completed_hand, 'completed_hand'),
    dora_indicators: transform(source.dora_indicators, 'dora_indicators'),
    melds: transform(source.melds, 'melds'),
  };
}

function mapOptionalRecord<T, U>(
  source: Partial<Record<RegionKey, T>>,
  transform: (value: T, key: RegionKey) => U,
): Partial<Record<RegionKey, U>> {
  const target: Partial<Record<RegionKey, U>> = {};
  for (const key of regionKeys()) {
    const value = source[key];
    if (value !== undefined) target[key] = transform(value, key);
  }
  return target;
}

function cloneRects(source: Record<RegionKey, Rect>): Record<RegionKey, Rect> {
  return mapRecord(source, (rect) => ({ ...rect }));
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
