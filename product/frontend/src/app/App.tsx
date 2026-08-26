import { MantineProvider } from '@mantine/core';
import type { ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';

import type { ApplicationStore } from '@/application';
import {
  ApplicationStateProvider,
  RecognitionPageServicesProvider,
  ScoringFlowServicesProvider,
  type RecognitionPageServices,
  type ScoringFlowServices,
} from '@/ui';

import { AppRoutes } from './routes';

export interface AppProps {
  readonly applicationStore?: ApplicationStore;
  readonly router?: ReactNode;
  readonly recognitionPageServices?: RecognitionPageServices;
  readonly scoringFlowServices?: ScoringFlowServices;
}

function OptionalScoringFlowServicesProvider({
  children,
  services,
}: {
  readonly children: ReactNode;
  readonly services: ScoringFlowServices | undefined;
}) {
  return services === undefined ? children : (
    <ScoringFlowServicesProvider services={services}>
      {children}
    </ScoringFlowServicesProvider>
  );
}

function OptionalRecognitionPageServicesProvider({
  children,
  services,
}: {
  readonly children: ReactNode;
  readonly services: RecognitionPageServices | undefined;
}) {
  return services === undefined ? children : (
    <RecognitionPageServicesProvider services={services}>
      {children}
    </RecognitionPageServicesProvider>
  );
}

export function App({
  applicationStore,
  router,
  recognitionPageServices,
  scoringFlowServices,
}: AppProps = {}) {
  const routes = router ?? (
    <BrowserRouter>
      <AppRoutes />
    </BrowserRouter>
  );

  return (
    <MantineProvider>
      <ApplicationStateProvider store={applicationStore}>
        <OptionalScoringFlowServicesProvider services={scoringFlowServices}>
          <OptionalRecognitionPageServicesProvider services={recognitionPageServices}>
            {routes}
          </OptionalRecognitionPageServicesProvider>
        </OptionalScoringFlowServicesProvider>
      </ApplicationStateProvider>
    </MantineProvider>
  );
}
