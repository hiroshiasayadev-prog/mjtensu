import {
  createApplicationStore,
  createCorrectionEditorService,
  createScoringSessionService,
  type ApplicationStore,
  type ScoringSessionService,
} from '@/application';
import {
  createBrowserCameraService,
  type CameraService,
} from '@/camera';
import {
  createProductionRecognitionServices,
  type ProductionRecognitionServices,
} from '@/recognition';
import {
  loadProductionScoringService,
  type ScoringService,
} from '@/scoring';
import type {
  RecognitionPageServices,
  ScoringFlowServices,
} from '@/ui';

export interface ProductionServiceFactories {
  readonly createCameraService?: () => CameraService;
  readonly createRecognitionServices?: () => ProductionRecognitionServices;
  readonly loadScoringService?: () => Promise<ScoringService>;
}

export interface ProductionServiceGraph {
  readonly camera: CameraService;
  readonly recognition: ProductionRecognitionServices;
  readonly scoring: ScoringService;
  readonly scoringSession: ScoringSessionService;
  readonly applicationStore: ApplicationStore;
  readonly recognitionPageServices: RecognitionPageServices;
  readonly scoringFlowServices: ScoringFlowServices;
  prefetchRecognitionModels(): Promise<void>;
  dispose(): Promise<void>;
}

export async function createProductionServiceGraph(
  factories: ProductionServiceFactories = {},
): Promise<ProductionServiceGraph> {
  const camera = (factories.createCameraService ?? createBrowserCameraService)();
  const recognition = (
    factories.createRecognitionServices ?? createProductionRecognitionServices
  )();
  const scoring = await (
    factories.loadScoringService ?? loadProductionScoringService
  )();
  const scoringSession = createScoringSessionService(scoring);
  const correctionEditor = createCorrectionEditorService(scoring);
  const applicationStore = createApplicationStore(
    {},
    { scoringSessionService: scoringSession },
  );
  const recognitionPageServices: RecognitionPageServices = {
    camera,
    runtime: recognition.runtime,
    recognizer: recognition.recognizer,
  };
  const scoringFlowServices: ScoringFlowServices = {
    correctionEditor,
    scoringSession,
  };
  let disposalInFlight: Promise<void> | null = null;

  return {
    camera,
    recognition,
    scoring,
    scoringSession,
    applicationStore,
    recognitionPageServices,
    scoringFlowServices,
    prefetchRecognitionModels() {
      return recognition.prefetch();
    },
    dispose() {
      if (disposalInFlight !== null) {
        return disposalInFlight;
      }
      disposalInFlight = recognition.dispose();
      return disposalInFlight;
    },
  };
}
