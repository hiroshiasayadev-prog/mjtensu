import { createContext, type ReactNode, useContext } from 'react';

import type {
  CorrectionEditorService,
  ScoringSessionService,
} from '@/application';

export interface ScoringFlowServices {
  readonly correctionEditor: CorrectionEditorService;
  readonly scoringSession: Pick<
    ScoringSessionService,
    'update' | 'preview' | 'calculate'
  >;
}

export interface ScoringFlowServicesProviderProps {
  readonly children: ReactNode;
  readonly services: ScoringFlowServices;
}

const ScoringFlowServicesContext = createContext<ScoringFlowServices | null>(null);

export function ScoringFlowServicesProvider({
  children,
  services,
}: ScoringFlowServicesProviderProps) {
  return (
    <ScoringFlowServicesContext.Provider value={services}>
      {children}
    </ScoringFlowServicesContext.Provider>
  );
}

export function useScoringFlowServices(): ScoringFlowServices | null {
  return useContext(ScoringFlowServicesContext);
}
