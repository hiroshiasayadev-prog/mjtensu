import { MantineProvider } from '@mantine/core';
import type { ReactNode } from 'react';
import { BrowserRouter } from 'react-router-dom';

import type { ApplicationStore } from '@/application';
import { ApplicationStateProvider } from '@/ui';

import { AppRoutes } from './routes';

export interface AppProps {
  readonly applicationStore?: ApplicationStore;
  readonly router?: ReactNode;
}

export function App({ applicationStore, router }: AppProps = {}) {
  return (
    <MantineProvider>
      <ApplicationStateProvider store={applicationStore}>
        {router ?? (
          <BrowserRouter>
            <AppRoutes />
          </BrowserRouter>
        )}
      </ApplicationStateProvider>
    </MantineProvider>
  );
}
