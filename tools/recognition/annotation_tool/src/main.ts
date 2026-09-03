import './styles.css';
import {
  assetUrl,
  fetchCampaigns,
  fetchCapture,
  fetchCaptureList,
  saveAnnotation,
} from './api';
import { expectedGroups, regionKeys, validateAnnotations } from './assignment';
import { CanvasEditor } from './editor';
import { normalizeAngle } from './geometry';
import type {
  AnnotationBox,
  AnnotationDocument,
  AnnotationReview,
  CampaignSummary,
  CaptureDetail,
  CaptureSummary,
  RegionKey,
  ValidationResult,
} from './types';

const app = document.querySelector<HTMLDivElement>('#app');
if (app === null) throw new Error('Missing #app root.');
const appRoot = app;

const initialQuery = new URLSearchParams(window.location.search);
const obbReviewMode = initialQuery.get('mode') === 'obb-review';

let campaigns: CampaignSummary[] = [];
let captures: CaptureSummary[] = [];
let currentCampaignId = '';
let currentDetail: CaptureDetail | null = null;
let currentRegion: RegionKey = 'completed_hand';
let boxesByRegion: Record<RegionKey, AnnotationBox[]> = emptyBoxes();
let currentReview: AnnotationReview | undefined;
let validation: ValidationResult | null = null;
let editor: CanvasEditor;
let captureLoadGeneration = 0;
let autosaveTimer: number | null = null;
let saveSerial = Promise.resolve();
let suppressAutosave = false;
let filterStatus = obbReviewMode ? 'review-pending' : 'all';

renderShell();
editor = new CanvasEditor(
  requireElement<HTMLCanvasElement>('annotation-canvas'),
  onEditorBoxesChanged,
  onEditorSelectionChanged,
);
bindShellEvents();
void boot();

async function boot(): Promise<void> {
  setGlobalStatus('campaignを読み込み中…');
  try {
    campaigns = await fetchCampaigns();
    renderCampaignOptions();
    if (campaigns.length === 0) {
      setError('保存済みcaptureが見つからない。capture APIのstorage rootを確認してください。');
      return;
    }
    const requested = new URLSearchParams(window.location.search).get('campaign');
    const selected = campaigns.find((campaign) => campaign.campaignId === requested) ?? campaigns[0];
    if (selected === undefined) return;
    await loadCampaign(selected.campaignId);
  } catch (error) {
    setError(errorMessage(error));
  }
}

function renderShell(): void {
  appRoot.innerHTML = `
    <div class="app-shell">
      <header class="topbar">
        <div class="brand-block">
          <span class="eyebrow">MJTENSU / RECOGNITION</span>
          <h1>${obbReviewMode ? 'Tile OBB review' : 'Tile annotation'}</h1>
        </div>
        <label class="campaign-picker">campaign
          <select id="campaign-select"></select>
        </label>
        <div class="top-progress" id="top-progress">—</div>
        <div class="global-status" id="global-status">起動中</div>
      </header>

      <aside class="capture-sidebar">
        <div class="sidebar-controls">
          <label>表示
            <select id="status-filter">
              <option value="all">すべて</option>
              <option value="annotated">annotationあり</option>
              <option value="review-pending">AI未レビュー</option>
              <option value="reviewed">レビュー済</option>
              <option value="unannotated">未着手</option>
              <option value="draft">途中</option>
              <option value="complete">完了</option>
            </select>
          </label>
        </div>
        <div id="capture-list" class="capture-list"></div>
      </aside>

      <main class="workspace">
        <section class="editor-panel">
          <div class="capture-heading">
            <div>
              <span class="eyebrow" id="capture-eyebrow">capture未選択</span>
              <h2 id="capture-title">—</h2>
            </div>
            <div class="capture-badges">
              <div id="capture-review-badge" class="review-badge">—</div>
              <div id="capture-environment" class="environment-badge">—</div>
            </div>
          </div>

          <div class="region-tabs" id="region-tabs">
            <button data-region="completed_hand">手牌</button>
            <button data-region="dora_indicators">ドラ</button>
            <button data-region="melds">副露</button>
          </div>

          <div class="editor-toolbar">
            <button id="reset-detections-button">検出結果で再設定</button>
            <button id="equal-layout-button">等間隔で作る</button>
            <button id="add-box-button">矩形を追加</button>
            <button id="delete-box-button" disabled>削除</button>
            <button id="split-x-button" disabled>左右に分割</button>
            <button id="split-y-button" disabled>上下に分割</button>
          </div>

          <div class="canvas-stage" id="canvas-stage">
            <canvas id="annotation-canvas"></canvas>
            <div id="canvas-empty" class="canvas-empty hidden">このregionは撮影されていません</div>
          </div>

          <div class="editor-help">
            矩形内をdrag: 移動　四隅をdrag: resize　上の丸をdrag: 自由回転　Shift+回転: 5° snap　Delete: 削除
          </div>
        </section>

        <aside class="inspector">
          <section class="inspector-card selection-card">
            <span class="eyebrow">選択中</span>
            <div id="selection-label">矩形なし</div>
            <div class="size-grid">
              <label>幅 px
                <input id="width-input" type="number" step="0.5" min="2" disabled>
              </label>
              <label>高さ px
                <input id="height-input" type="number" step="0.5" min="2" disabled>
              </label>
            </div>
            <label>角度
              <div class="angle-row">
                <input id="angle-input" type="number" step="0.1" min="-180" max="180" disabled>
                <span>°</span>
              </div>
            </label>
          </section>

          <section class="inspector-card validation-card">
            <div class="card-heading">
              <span class="eyebrow">完了条件</span>
              <strong id="validation-overall">—</strong>
            </div>
            <div id="validation-list" class="validation-list"></div>
          </section>

          <section class="inspector-card save-card">
            <div id="save-status">—</div>
            <div class="save-actions">
              <button id="previous-button">← 前</button>
              <button id="complete-next-button" class="primary" disabled>${obbReviewMode ? 'OBB保存して次へ →' : '保存して次へ →'}</button>
            </div>
          </section>
        </aside>
      </main>

      <div id="error-banner" class="error-banner hidden"></div>
    </div>`;
}

function bindShellEvents(): void {
  requireElement<HTMLSelectElement>('status-filter').value = filterStatus;
  requireElement<HTMLSelectElement>('campaign-select').onchange = (event) => {
    const campaignId = (event.currentTarget as HTMLSelectElement).value;
    void loadCampaign(campaignId).catch((error) => setError(errorMessage(error)));
  };
  requireElement<HTMLSelectElement>('status-filter').onchange = (event) => {
    filterStatus = (event.currentTarget as HTMLSelectElement).value;
    renderCaptureList();
  };
  requireElement('region-tabs').addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-region]');
    const region = button?.dataset.region as RegionKey | undefined;
    if (region !== undefined) {
      void switchRegion(region).catch((error) => setError(errorMessage(error)));
    }
  });
  requireElement<HTMLButtonElement>('reset-detections-button').onclick = () => resetCurrentRegionFromDetections();
  requireElement<HTMLButtonElement>('equal-layout-button').onclick = () => resetCurrentRegionEqualSpacing();
  requireElement<HTMLButtonElement>('add-box-button').onclick = () => {
    editor.setAddMode(!editor.isAddMode());
    renderAddMode();
  };
  requireElement<HTMLButtonElement>('delete-box-button').onclick = () => editor.deleteSelected();
  requireElement<HTMLButtonElement>('split-x-button').onclick = () => editor.splitSelected('screen-x');
  requireElement<HTMLButtonElement>('split-y-button').onclick = () => editor.splitSelected('screen-y');
  requireElement<HTMLInputElement>('width-input').oninput = () => updateSelectedSizeFromInputs();
  requireElement<HTMLInputElement>('height-input').oninput = () => updateSelectedSizeFromInputs();
  requireElement<HTMLInputElement>('angle-input').oninput = (event) => {
    const value = (event.currentTarget as HTMLInputElement).valueAsNumber;
    if (Number.isFinite(value)) editor.setSelectedAngle(value);
  };
  requireElement<HTMLButtonElement>('previous-button').onclick = () => {
    void movePrevious().catch((error) => setError(errorMessage(error)));
  };
  requireElement<HTMLButtonElement>('complete-next-button').onclick = () => void completeAndMoveNext();
}

async function loadCampaign(campaignId: string): Promise<void> {
  await flushAutosave();
  clearError();
  setGlobalStatus('capture一覧を読み込み中…');
  currentCampaignId = campaignId;
  requireElement<HTMLSelectElement>('campaign-select').value = campaignId;
  const url = new URL(window.location.href);
  url.searchParams.set('campaign', campaignId);
  window.history.replaceState(null, '', url);
  captures = await fetchCaptureList(campaignId);
  renderCaptureList();
  renderProgress();
  const first = obbReviewMode
    ? captures.find((capture) => capture.reviewState === 'model_suggested')
      ?? captures.find((capture) => capture.annotationStatus !== 'unannotated')
    : captures.find((capture) => capture.annotationStatus !== 'complete') ?? captures[0];
  if (first === undefined) {
    setGlobalStatus('captureなし');
    clearCurrentCapture();
    return;
  }
  await loadCapture(first.captureId);
}

async function loadCapture(captureId: string): Promise<void> {
  await flushAutosave();
  const generation = ++captureLoadGeneration;
  let initializedFromDetector = false;
  suppressAutosave = true;
  currentDetail = null;
  clearError();
  setGlobalStatus('captureを読み込み中…');
  editor.clearImage();
  requireElement('canvas-empty').classList.add('hidden');
  try {
    const detail = await fetchCapture(captureId);
    if (generation !== captureLoadGeneration) return;
    currentDetail = detail;
    currentReview = detail.annotation?.document.review;
    const emptyCatalogDraftCanUseDetector = (
      detail.campaignId.startsWith('tile-catalog')
      && detail.annotation?.status === 'draft'
      && annotationBoxCount(detail.annotation.document.boxes) === 0
      && detail.detections.some((detection) => detection.region !== 'invalid' && detection.original !== null)
    );
    initializedFromDetector = detail.annotation === null || emptyCatalogDraftCanUseDetector;
    boxesByRegion = initializedFromDetector
      ? autoBoxesFromDetections(detail)
      : cloneBoxes(detail.annotation?.document.boxes ?? emptyBoxes());
    validation = validateAnnotations(detail, boxesByRegion);
    currentRegion = firstAvailableRegion(detail);
    renderCaptureHeading();
    renderCaptureList();
    await switchRegion(currentRegion);
    renderValidation();
    renderNavigation();
    setGlobalStatus(
      initializedFromDetector
        ? '検出結果から初期化'
        : currentReview?.state === 'model_suggested'
          ? 'AI OBB候補を読込・未レビュー'
          : currentReview?.state === 'human_reviewed'
            ? '人間レビュー済みOBBを読込'
            : `${detail.annotation?.status ?? 'draft'}を読込`,
    );
  } finally {
    suppressAutosave = false;
  }
}

async function switchRegion(region: RegionKey): Promise<void> {
  const detail = currentDetail;
  if (detail === null) return;
  currentRegion = region;
  editor.setAddMode(false);
  renderAddMode();
  renderRegionTabs();
  editor.setBoxes(boxesByRegion[region]);
  updateValidationAndLabels();
  const path = detail.regionPaths[region];
  const empty = requireElement('canvas-empty');
  if (path === null) {
    editor.clearImage();
    empty.classList.remove('hidden');
    setToolbarDisabled(true);
    return;
  }
  setToolbarDisabled(false);
  empty.classList.add('hidden');
  const rect = detail.manifest.regionRects[region].pixel;
  const width = Math.max(1, Math.floor(rect.width + 0.5));
  const height = Math.max(1, Math.floor(rect.height + 0.5));
  const applied = await editor.setImage(assetUrl(path), width, height);
  if (!applied || currentDetail?.captureId !== detail.captureId || currentRegion !== region) return;
  editor.setBoxes(boxesByRegion[region]);
  updateValidationAndLabels();
}

function onEditorBoxesChanged(boxes: AnnotationBox[]): void {
  if (currentDetail === null) return;
  boxesByRegion = { ...boxesByRegion, [currentRegion]: boxes };
  updateValidationAndLabels();
  scheduleAutosave();
}

function onEditorSelectionChanged(box: AnnotationBox | null): void {
  const widthInput = requireElement<HTMLInputElement>('width-input');
  const heightInput = requireElement<HTMLInputElement>('height-input');
  const angleInput = requireElement<HTMLInputElement>('angle-input');
  const selectedLabel = requireElement('selection-label');
  const hasSelection = box !== null;
  widthInput.disabled = !hasSelection;
  heightInput.disabled = !hasSelection;
  angleInput.disabled = !hasSelection;
  requireElement<HTMLButtonElement>('delete-box-button').disabled = !hasSelection;
  requireElement<HTMLButtonElement>('split-x-button').disabled = !hasSelection;
  requireElement<HTMLButtonElement>('split-y-button').disabled = !hasSelection;
  if (box === null) {
    widthInput.value = '';
    heightInput.value = '';
    angleInput.value = '';
    selectedLabel.textContent = '矩形なし';
    return;
  }
  widthInput.value = box.width.toFixed(1);
  heightInput.value = box.height.toFixed(1);
  angleInput.value = normalizeAngle(box.angleDeg).toFixed(1);
  const label = validation?.regions[currentRegion].labels.get(box.id);
  selectedLabel.textContent = `${label?.text ?? '未割当'} / ${box.width.toFixed(1)} × ${box.height.toFixed(1)} px`;
}

function updateSelectedSizeFromInputs(): void {
  const width = requireElement<HTMLInputElement>('width-input').valueAsNumber;
  const height = requireElement<HTMLInputElement>('height-input').valueAsNumber;
  if (Number.isFinite(width) && Number.isFinite(height)) editor.setSelectedSize(width, height);
}

function updateValidationAndLabels(): void {
  const detail = currentDetail;
  if (detail === null) return;
  validation = validateAnnotations(detail, boxesByRegion);
  editor.setLabels(validation.regions[currentRegion].labels);
  renderValidation();
  renderRegionTabs();
  renderNavigation();
  onEditorSelectionChanged(editor.selectedBox());
}

function resetCurrentRegionFromDetections(): void {
  const detail = currentDetail;
  if (detail === null) return;
  if (!window.confirm('現在regionの手動修正を破棄し、DBのdetector候補へ戻します。')) return;
  boxesByRegion = {
    ...boxesByRegion,
    [currentRegion]: autoBoxesForRegion(detail, currentRegion),
  };
  editor.setBoxes(boxesByRegion[currentRegion]);
  updateValidationAndLabels();
  scheduleAutosave();
}

function resetCurrentRegionEqualSpacing(): void {
  const detail = currentDetail;
  if (detail === null) return;
  if (!window.confirm('現在regionの矩形を破棄し、期待牌数どおりの仮矩形を等間隔で作ります。')) return;
  boxesByRegion = {
    ...boxesByRegion,
    [currentRegion]: equalSpacingBoxes(detail, currentRegion),
  };
  editor.setBoxes(boxesByRegion[currentRegion]);
  updateValidationAndLabels();
  scheduleAutosave();
}

function scheduleAutosave(): void {
  if (suppressAutosave || currentDetail === null) return;
  if (autosaveTimer !== null) window.clearTimeout(autosaveTimer);
  requireElement('save-status').textContent = '変更あり・draft未保存';
  autosaveTimer = window.setTimeout(() => {
    autosaveTimer = null;
    void enqueueSave('draft');
  }, 700);
}

async function flushAutosave(): Promise<void> {
  if (autosaveTimer !== null) {
    window.clearTimeout(autosaveTimer);
    autosaveTimer = null;
    await enqueueSave('draft');
  }
  await saveSerial;
}

function enqueueSave(status: 'draft' | 'complete'): Promise<void> {
  const detail = currentDetail;
  if (detail === null) return Promise.resolve();
  const captureId = detail.captureId;
  const document = makeDocument(captureId, boxesByRegion, status);
  const operation = saveSerial.then(async () => {
    if (status === 'draft') requireElement('save-status').textContent = 'draft保存中…';
    await saveAnnotation(captureId, status, document);
    updateCaptureStatus(captureId, status, document.review);
    if (currentDetail?.captureId === captureId) {
      currentReview = document.review;
      requireElement('save-status').textContent = status === 'complete'
        ? (obbReviewMode ? '人間レビュー済みとして保存' : '完了として保存済み')
        : currentReview?.state === 'model_suggested'
          ? 'AI候補をdraft保存・未レビュー'
          : 'draft自動保存済み';
      renderCaptureHeading();
    }
    renderCaptureList();
    renderProgress();
  }).catch((error) => {
    if (currentDetail?.captureId === captureId) {
      requireElement('save-status').textContent = `保存失敗: ${errorMessage(error)}`;
    }
    setError(errorMessage(error));
    throw error;
  });
  saveSerial = operation.catch(() => undefined);
  return operation;
}

async function completeAndMoveNext(): Promise<void> {
  if (currentDetail === null || validation?.complete !== true) return;
  if (autosaveTimer !== null) {
    window.clearTimeout(autosaveTimer);
    autosaveTimer = null;
  }
  const completedId = currentDetail.captureId;
  setGlobalStatus(obbReviewMode ? 'OBB annotationを保存中…' : '完了annotationを保存中…');
  try {
    await enqueueSave('complete');
    const next = obbReviewMode
      ? nextReviewPendingAfter(completedId)
      : nextIncompleteAfter(completedId);
    if (next === null) {
      setGlobalStatus(obbReviewMode ? 'AI OBB候補のレビュー完了 🎉' : '全captureのannotation完了 🎉');
      renderProgress();
      return;
    }
    await loadCapture(next.captureId);
  } catch (error) {
    setError(errorMessage(error));
  }
}

async function movePrevious(): Promise<void> {
  const detail = currentDetail;
  if (detail === null) return;
  const index = captures.findIndex((capture) => capture.captureId === detail.captureId);
  if (!obbReviewMode) {
    const previous = captures[index - 1];
    if (previous !== undefined) await loadCapture(previous.captureId);
    return;
  }
  for (let candidateIndex = index - 1; candidateIndex >= 0; candidateIndex -= 1) {
    const candidate = captures[candidateIndex];
    if (candidate !== undefined && candidate.annotationStatus !== 'unannotated') {
      await loadCapture(candidate.captureId);
      return;
    }
  }
}

async function navigateToCapture(captureId: string): Promise<void> {
  const detail = currentDetail;
  if (detail === null || detail.captureId === captureId) {
    if (detail === null && captureId) await loadCapture(captureId);
    return;
  }
  if (obbReviewMode) {
    await flushAutosave();
    await loadCapture(captureId);
    return;
  }
  const currentIndex = captures.findIndex((capture) => capture.captureId === detail.captureId);
  const targetIndex = captures.findIndex((capture) => capture.captureId === captureId);
  if (targetIndex < 0) return;
  if (targetIndex > currentIndex) {
    if (validation?.complete !== true) {
      setError('現在のcaptureは期待数またはregion内包条件を満たしていないため、後のcaptureへ進めません。');
      return;
    }
    if (autosaveTimer !== null) {
      window.clearTimeout(autosaveTimer);
      autosaveTimer = null;
    }
    await enqueueSave('complete');
  }
  await loadCapture(captureId);
}

function autoBoxesFromDetections(detail: CaptureDetail): Record<RegionKey, AnnotationBox[]> {
  return {
    completed_hand: autoBoxesForRegion(detail, 'completed_hand'),
    dora_indicators: autoBoxesForRegion(detail, 'dora_indicators'),
    melds: autoBoxesForRegion(detail, 'melds'),
  };
}

function autoBoxesForRegion(detail: CaptureDetail, region: RegionKey): AnnotationBox[] {
  const origin = detail.manifest.regionRects[region].pixel;
  return detail.detections
    .filter((detection) => detection.region === region && detection.original !== null)
    .map((detection) => {
      const rect = detection.original;
      if (rect === null) throw new Error('Unexpected null detection rectangle.');
      const sideways = rect.width > rect.height;
      return {
        id: crypto.randomUUID(),
        centerX: rect.x - origin.x + rect.width / 2,
        centerY: rect.y - origin.y + rect.height / 2,
        width: sideways ? rect.height : rect.width,
        height: sideways ? rect.width : rect.height,
        angleDeg: sideways ? 90 : 0,
      };
    });
}

function equalSpacingBoxes(detail: CaptureDetail, region: RegionKey): AnnotationBox[] {
  const groups = expectedGroups(detail.task, region);
  const rect = detail.manifest.regionRects[region].pixel;
  const regionWidth = Math.max(1, Math.floor(rect.width + 0.5));
  const regionHeight = Math.max(1, Math.floor(rect.height + 0.5));
  if (groups.length === 0) return [];
  const rowHeight = regionHeight / groups.length;
  const result: AnnotationBox[] = [];

  groups.forEach((group, groupIndex) => {
    const slots = group.slots;
    if (slots.length === 0) return;
    let tileHeight = rowHeight * 0.72;
    let tileWidth = tileHeight * 0.68;
    const footprintWidths = slots.map((slot) => isSideways(slot.rotation) ? tileHeight : tileWidth);
    const baseGap = Math.max(2, regionWidth * 0.012);
    const required = footprintWidths.reduce((sum, width) => sum + width, 0) + baseGap * Math.max(0, slots.length - 1);
    if (required > regionWidth * 0.92) {
      const factor = regionWidth * 0.92 / required;
      tileHeight *= factor;
      tileWidth *= factor;
    }
    const scaledFootprints = slots.map((slot) => isSideways(slot.rotation) ? tileHeight : tileWidth);
    const totalWidth = scaledFootprints.reduce((sum, width) => sum + width, 0)
      + baseGap * Math.max(0, slots.length - 1);
    let cursor = (regionWidth - totalWidth) / 2;
    const centerY = rowHeight * groupIndex + rowHeight / 2;
    slots.forEach((slot, slotIndex) => {
      const footprint = scaledFootprints[slotIndex] ?? tileWidth;
      result.push({
        id: crypto.randomUUID(),
        centerX: cursor + footprint / 2,
        centerY,
        width: tileWidth,
        height: tileHeight,
        angleDeg: normalizeAngle(slot.rotation),
      });
      cursor += footprint + baseGap;
    });
  });
  return result;
}

function makeDocument(
  captureId: string,
  boxes: Record<RegionKey, AnnotationBox[]>,
  status: 'draft' | 'complete',
): AnnotationDocument {
  const review = obbReviewMode
    ? status === 'complete'
      ? {
          ...(currentReview ?? {}),
          state: 'human_reviewed' as const,
          source: 'annotation_tool',
          reviewedAt: new Date().toISOString(),
        }
      : currentReview
    : currentReview;
  return {
    schemaVersion: 1,
    captureId,
    boxes: cloneBoxes(boxes),
    ...(review === undefined ? {} : { review }),
  };
}

function renderCampaignOptions(): void {
  const select = requireElement<HTMLSelectElement>('campaign-select');
  select.innerHTML = campaigns.map((campaign) => (
    `<option value="${escapeHtml(campaign.campaignId)}">${escapeHtml(campaign.campaignId)} · ${campaign.completeCount}/${campaign.captureCount}</option>`
  )).join('');
}

function renderCaptureList(): void {
  const list = requireElement('capture-list');
  const visible = captures.filter((capture) => (
    filterStatus === 'all'
    || (filterStatus === 'annotated' && capture.annotationStatus !== 'unannotated')
    || (filterStatus === 'review-pending' && capture.reviewState === 'model_suggested')
    || (filterStatus === 'reviewed' && capture.reviewState === 'human_reviewed')
    || capture.annotationStatus === filterStatus
  ));
  list.innerHTML = visible.map((capture) => {
    const active = capture.captureId === currentDetail?.captureId;
    return `
      <button class="capture-item ${active ? 'active' : ''}" data-capture-id="${escapeHtml(capture.captureId)}">
        <span class="status-dot ${reviewDotClass(capture)}"></span>
        <span class="capture-item-main">
          <strong>${currentCampaignId.startsWith('tile-catalog') ? `撮影 ${capture.taskOrder + 1}` : `配置 ${capture.layoutOrdinal + 1}`}</strong>
          <small>${escapeHtml(capture.environment.label ?? environmentText(capture.environment.brightness, capture.environment.shadow))}</small>
        </span>
        <span class="status-text">${captureStatusText(capture)}</span>
      </button>`;
  }).join('');
  for (const button of list.querySelectorAll<HTMLButtonElement>('button[data-capture-id]')) {
    button.onclick = () => {
      void navigateToCapture(button.dataset.captureId ?? '')
        .catch((error) => setError(errorMessage(error)));
    };
  }
}

function renderCaptureHeading(): void {
  const detail = currentDetail;
  if (detail === null) return;
  requireElement('capture-eyebrow').textContent = `${detail.campaignId} / ${detail.captureId}`;
  requireElement('capture-title').textContent = detail.campaignId.startsWith('tile-catalog')
    ? `カタログ撮影 ${detail.task.taskOrder + 1}`
    : `配置 ${detail.task.layoutOrdinal + 1}`;
  requireElement('capture-environment').textContent = detail.task.environment.label ?? environmentText(
    detail.task.environment.brightness,
    detail.task.environment.shadow,
  );
  const reviewBadge = requireElement('capture-review-badge');
  reviewBadge.textContent = reviewStateText(currentReview?.state);
  reviewBadge.className = `review-badge ${currentReview?.state ?? 'untracked'}`;
}

function renderRegionTabs(): void {
  const detail = currentDetail;
  for (const button of requireElement('region-tabs').querySelectorAll<HTMLButtonElement>('button[data-region]')) {
    const region = button.dataset.region as RegionKey;
    const result = validation?.regions[region];
    const path = detail?.regionPaths[region] ?? null;
    button.classList.toggle('active', region === currentRegion);
    button.classList.toggle('valid', result?.valid === true);
    button.classList.toggle('invalid', result?.valid === false && path !== null);
    button.disabled = path === null;
    const catalog = detail?.campaignId.startsWith('tile-catalog') === true;
    const base = catalog
      ? (region === 'melds' ? '全牌' : '未使用')
      : region === 'completed_hand' ? '手牌' : region === 'dora_indicators' ? 'ドラ' : '副露';
    button.textContent = result === undefined ? base : `${base} ${result.actualCount}/${result.expectedCount}`;
  }
}

function renderValidation(): void {
  const result = validation;
  if (result === null) {
    requireElement('validation-overall').textContent = '—';
    requireElement('validation-list').innerHTML = '';
    return;
  }
  requireElement('validation-overall').textContent = result.complete ? 'OK' : '要修正';
  requireElement('validation-overall').className = result.complete ? 'ok-text' : 'error-text';
  const rows: string[] = [];
  for (const region of regionKeys()) {
    const assignment = result.regions[region];
    if (assignment.groups.length === 0) {
      rows.push(validationRow(regionLabel(region), assignment.actualCount, 0, assignment.valid));
      continue;
    }
    for (const group of assignment.groups) {
      rows.push(validationRow(group.label, group.boxes.length, group.expected.length, group.valid));
    }
  }
  rows.push(`
    <div class="validation-row ${result.allInside ? 'valid' : 'invalid'}">
      <span>region内</span><strong>${result.allInside ? '✓' : 'はみ出しあり'}</strong>
    </div>`);
  requireElement('validation-list').innerHTML = rows.join('');
}

function renderProgress(): void {
  const complete = captures.filter((capture) => capture.annotationStatus === 'complete').length;
  const draft = captures.filter((capture) => capture.annotationStatus === 'draft').length;
  if (obbReviewMode) {
    const reviewed = captures.filter((capture) => capture.reviewState === 'human_reviewed').length;
    const pending = captures.filter((capture) => capture.reviewState === 'model_suggested').length;
    requireElement('top-progress').textContent = `${reviewed} レビュー済 · ${pending} AI未レビュー`;
  } else {
    requireElement('top-progress').textContent = `${complete} / ${captures.length} 完了 · ${draft} 途中`;
  }
  const campaign = campaigns.find((candidate) => candidate.campaignId === currentCampaignId);
  if (campaign !== undefined) {
    campaign.completeCount = complete;
    campaign.draftCount = draft;
    renderCampaignOptions();
    requireElement<HTMLSelectElement>('campaign-select').value = currentCampaignId;
  }
}

function renderNavigation(): void {
  const detail = currentDetail;
  const completeButton = requireElement<HTMLButtonElement>('complete-next-button');
  completeButton.disabled = validation?.complete !== true;
  if (detail === null) {
    requireElement<HTMLButtonElement>('previous-button').disabled = true;
    return;
  }
  const index = captures.findIndex((capture) => capture.captureId === detail.captureId);
  requireElement<HTMLButtonElement>('previous-button').disabled = obbReviewMode
    ? !captures.slice(0, index).some((capture) => capture.annotationStatus !== 'unannotated')
    : index <= 0;
}

function renderAddMode(): void {
  const button = requireElement<HTMLButtonElement>('add-box-button');
  button.classList.toggle('active-tool', editor.isAddMode());
  button.textContent = editor.isAddMode() ? '追加モード終了' : '矩形を追加';
}

function setToolbarDisabled(disabled: boolean): void {
  for (const id of ['reset-detections-button', 'equal-layout-button', 'add-box-button']) {
    requireElement<HTMLButtonElement>(id).disabled = disabled;
  }
}

function updateCaptureStatus(
  captureId: string,
  status: 'draft' | 'complete',
  review: AnnotationReview | undefined,
): void {
  const capture = captures.find((candidate) => candidate.captureId === captureId);
  if (capture !== undefined) {
    capture.annotationStatus = status;
    capture.reviewState = review?.state ?? null;
    capture.reviewSource = review?.source ?? null;
  }
}

function nextIncompleteAfter(captureId: string): CaptureSummary | null {
  const index = captures.findIndex((capture) => capture.captureId === captureId);
  for (let offset = 1; offset <= captures.length; offset += 1) {
    const candidate = captures[(index + offset) % captures.length];
    if (candidate !== undefined && candidate.annotationStatus !== 'complete') return candidate;
  }
  return null;
}

function nextReviewPendingAfter(captureId: string): CaptureSummary | null {
  const index = captures.findIndex((capture) => capture.captureId === captureId);
  for (let offset = 1; offset <= captures.length; offset += 1) {
    const candidate = captures[(index + offset) % captures.length];
    if (candidate !== undefined && candidate.reviewState === 'model_suggested') return candidate;
  }
  return null;
}

function firstAvailableRegion(detail: CaptureDetail): RegionKey {
  return regionKeys().find((region) => detail.regionPaths[region] !== null) ?? 'completed_hand';
}

function clearCurrentCapture(): void {
  currentDetail = null;
  currentReview = undefined;
  boxesByRegion = emptyBoxes();
  validation = null;
  editor.clearImage();
  requireElement('capture-title').textContent = '—';
  requireElement('validation-list').innerHTML = '';
  renderNavigation();
}

function emptyBoxes(): Record<RegionKey, AnnotationBox[]> {
  return { completed_hand: [], dora_indicators: [], melds: [] };
}

function annotationBoxCount(boxes: Record<RegionKey, AnnotationBox[]>): number {
  return regionKeys().reduce((sum, region) => sum + boxes[region].length, 0);
}

function cloneBoxes(
  boxes: Record<RegionKey, AnnotationBox[]>,
): Record<RegionKey, AnnotationBox[]> {
  return {
    completed_hand: boxes.completed_hand.map((box) => ({ ...box })),
    dora_indicators: boxes.dora_indicators.map((box) => ({ ...box })),
    melds: boxes.melds.map((box) => ({ ...box })),
  };
}

function validationRow(label: string, actual: number, expected: number, valid: boolean): string {
  return `
    <div class="validation-row ${valid ? 'valid' : 'invalid'}">
      <span>${escapeHtml(label)}</span>
      <strong>${actual} / ${expected} ${valid ? '✓' : '✕'}</strong>
    </div>`;
}

function regionLabel(region: RegionKey): string {
  return region === 'completed_hand' ? '手牌' : region === 'dora_indicators' ? 'ドラ' : '副露';
}

function environmentText(brightness: string, shadow: string): string {
  const brightnessText = brightness === 'bright' ? '明るい' : '暗い';
  const shadowText = shadow === 'none' ? '影なし' : '部分影';
  return `${brightnessText}・${shadowText}`;
}

function captureStatusText(capture: CaptureSummary): string {
  if (capture.reviewState === 'model_suggested') return 'AI未レビュー';
  if (capture.reviewState === 'human_reviewed') return 'レビュー済';
  return capture.annotationStatus === 'complete'
    ? '完了'
    : capture.annotationStatus === 'draft'
      ? '途中'
      : '未着手';
}

function reviewDotClass(capture: CaptureSummary): string {
  if (capture.reviewState === 'model_suggested') return 'model-suggested';
  if (capture.reviewState === 'human_reviewed') return 'human-reviewed';
  return capture.annotationStatus;
}

function reviewStateText(state: AnnotationReview['state'] | undefined): string {
  return state === 'model_suggested'
    ? 'AI候補・未レビュー'
    : state === 'human_reviewed'
      ? '✓ 人間レビュー済み'
      : 'レビュー状態なし';
}

function isSideways(rotation: number): boolean {
  const normalized = ((rotation % 180) + 180) % 180;
  return Math.abs(normalized - 90) < 45;
}

function setGlobalStatus(message: string): void {
  requireElement('global-status').textContent = message;
}

function setError(message: string): void {
  const banner = requireElement('error-banner');
  banner.textContent = message;
  banner.classList.remove('hidden');
}

function clearError(): void {
  requireElement('error-banner').classList.add('hidden');
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
