import '@mantine/core/styles.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

import { App, createProductionServiceGraph } from '@/app';
import { registerProductionPwaLifecycle } from '@/pwa';

async function bootstrapProductionApp(): Promise<void> {
  const rootElement = document.getElementById('root');
  if (rootElement === null) {
    throw new Error('Application root element #root was not found.');
  }

  const services = await createProductionServiceGraph();

  createRoot(rootElement).render(
    <StrictMode>
      <App
        applicationStore={services.applicationStore}
        recognitionPageServices={services.recognitionPageServices}
        scoringFlowServices={services.scoringFlowServices}
      />
    </StrictMode>,
  );

  // Model acquisition is app-lifetime background work. Do not await it before
  // the initial route can render; RecognitionRuntime.initialize() shares the
  // same asset resolver if Recognition is entered while this is still in flight.
  globalThis.setTimeout(() => {
    void services.prefetchRecognitionModels().catch(() => {
      // Top remains usable after a background prefetch failure. Recognition
      // initialization owns the visible retry/error boundary on first use.
    });
  }, 0);

  if (import.meta.env.PROD && import.meta.env.MODE !== 'e2e') {
    void registerProductionPwaLifecycle().catch(() => {
      // PWA registration failure must not make the already-running app unusable.
    });
  }
}

void bootstrapProductionApp();
