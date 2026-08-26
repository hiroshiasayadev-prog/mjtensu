import type {
  CampaignOverview,
  CatalogCaptureTask,
  PendingCatalogCapture,
} from './types';

export class CaptureUploadError extends Error {
  constructor(
    readonly status: number,
    readonly responseBody: string,
  ) {
    super(`Capture upload failed: HTTP ${status}: ${responseBody}`);
    this.name = 'CaptureUploadError';
  }
}

export async function fetchOverview(campaignId: string): Promise<CampaignOverview> {
  return getJson(`/api/campaigns/${encodeURIComponent(campaignId)}/overview`);
}

export async function fetchNextTask(campaignId: string): Promise<CatalogCaptureTask | null> {
  const response = await fetch(`/api/campaigns/${encodeURIComponent(campaignId)}/next-task`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Failed to fetch next task: HTTP ${response.status}`);
  return response.json() as Promise<CatalogCaptureTask>;
}

export async function undoLastCapture(campaignId: string): Promise<{
  captureId: string;
  taskId: string;
  removedPaths: string[];
}> {
  const response = await fetch(
    `/api/campaigns/${encodeURIComponent(campaignId)}/last-capture`,
    { method: 'DELETE' },
  );
  if (!response.ok) {
    throw new Error(`Undo failed: HTTP ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<{
    captureId: string;
    taskId: string;
    removedPaths: string[];
  }>;
}

export async function uploadCapture(capture: PendingCatalogCapture): Promise<{
  captureId: string;
  taskCompleted: boolean;
  nextTaskId: string | null;
}> {
  const form = new FormData();
  form.append(
    'manifest',
    new Blob([JSON.stringify(capture.manifest)], { type: 'application/json' }),
    'manifest.json',
  );
  form.append('original', capture.original, 'original.jpg');
  form.append('composite', capture.composite, 'composite.png');

  const response = await fetch('/api/captures', { method: 'POST', body: form });
  if (!response.ok) {
    throw new CaptureUploadError(response.status, await response.text());
  }
  return response.json() as Promise<{
    captureId: string;
    taskCompleted: boolean;
    nextTaskId: string | null;
  }>;
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed: HTTP ${response.status}: ${url}`);
  return response.json() as Promise<T>;
}
