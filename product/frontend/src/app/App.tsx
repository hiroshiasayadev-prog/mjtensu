import { MantineProvider } from '@mantine/core';
import type { ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';

import type { ApplicationStore } from '@/application';
import {
  ApplicationStateProvider,
  ScoringFlowServicesProvider,
  type ScoringFlowServices,
} from '@/ui';

import { AppRoutes } from './routes';

export interface AppProps {
  readonly applicationStore?: ApplicationStore;
  readonly router?: ReactNode;
  readonly scoringFlowServices?: ScoringFlowServices;
}

export function App({
  applicationStore,
  router,
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
        {scoringFlowServices === undefined ? (
          routes
        ) : (
          <ScoringFlowServicesProvider services={scoringFlowServices}>
            {routes}
          </ScoringFlowServicesProvider>
        )}
      </ApplicationStateProvider>
    </MantineProvider>
  );
}
