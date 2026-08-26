import { createContext, type ReactNode, useContext } from 'react';

import {
  createCorrectionEditorService,
  createScoringSessionService,
  type CorrectionEditorService,
  type ScoringSessionService,
} from '@/application';
import type { ScoringService } from '@/scoring';

export interface ScoringFlowServices {
  readonly scoringSession: ScoringSessionService;
  readonly correctionEditor: CorrectionEditorService;
}

export interface ScoringFlowServicesProviderProps {
  readonly children: ReactNode;
  readonly services: ScoringFlowServices;
}

const deferredScoringService: ScoringService = {
  validateWinningStructure: () => ({ kind: 'valid' }),
  preview: () => ({ kind: 'no-yaku' }),
  calculate: () => {
    throw new Error('Scoring service is not connected.');
  },
};

const defaultScoringFlowServices: ScoringFlowServices = {
  scoringSession: createScoringSessionService(deferredScoringService),
  correctionEditor: createCorrectionEditorService(deferredScoringService),
};

const ScoringFlowServicesContext = createContext<ScoringFlowServices>(
  defaultScoringFlowServices,
);

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

export function useScoringFlowServices(): ScoringFlowServices {
  return useContext(ScoringFlowServicesContext);
}
