import {
  createContext,
  type ReactNode,
  useContext,
  useRef,
} from 'react';
import { useStore } from 'zustand';

import {
  createApplicationStore,
  type ApplicationStore,
  type ApplicationStoreState,
} from '@/application';

const ApplicationStoreContext = createContext<ApplicationStore | null>(null);

export interface ApplicationStateProviderProps {
  readonly children: ReactNode;
  readonly store?: ApplicationStore;
}

export function ApplicationStateProvider({
  children,
  store,
}: ApplicationStateProviderProps) {
  const defaultStore = useRef<ApplicationStore | null>(null);

  defaultStore.current ??= createApplicationStore();

  return (
    <ApplicationStoreContext.Provider value={store ?? defaultStore.current}>
      {children}
    </ApplicationStoreContext.Provider>
  );
}

export function useApplicationStore<T>(
  selector: (state: ApplicationStoreState) => T,
): T {
  const store = useContext(ApplicationStoreContext);

  if (store === null) {
    throw new Error('ApplicationStateProvider is required.');
  }

  return useStore(store, selector);
}
