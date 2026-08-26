import type {
  AnnotationDocument,
  CampaignSummary,
  CaptureDetail,
  CaptureSummary,
} from './types';

export async function fetchCampaigns(): Promise<CampaignSummary[]> {
  const payload = await getJson<{ campaigns: CampaignSummary[] }>('/api/annotation-campaigns');
  return payload.campaigns;
}

export async function fetchCaptureList(campaignId: string): Promise<CaptureSummary[]> {
  const payload = await getJson<{ captures: CaptureSummary[] }>(
    `/api/annotations/captures?campaignId=${encodeURIComponent(campaignId)}`,
  );
  return payload.captures;
}

export async function fetchCapture(captureId: string): Promise<CaptureDetail> {
  return getJson(`/api/annotations/captures/${encodeURIComponent(captureId)}`);
}

export function assetUrl(relativePath: string): string {
  return `/api/annotation-asset?path=${encodeURIComponent(relativePath)}`;
}

export async function saveAnnotation(
  captureId: string,
  status: 'draft' | 'complete',
  document: AnnotationDocument,
): Promise<void> {
  const response = await fetch(`/api/annotations/captures/${encodeURIComponent(captureId)}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status, document }),
  });
  if (!response.ok) {
    throw new Error(`Annotation save failed: HTTP ${response.status}: ${await response.text()}`);
  }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed: HTTP ${response.status}: ${await response.text()}`);
  return response.json() as Promise<T>;
}
