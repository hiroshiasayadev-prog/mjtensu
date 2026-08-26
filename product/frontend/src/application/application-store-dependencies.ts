import type { ScoringConditionPolicy } from './scoring-condition-policy';
import type { ScoringSessionService } from './scoring-session-service';

export type ApplicationSessionPort = ScoringSessionService;

export interface ApplicationStoreDependencies {
  readonly scoringSessionService?: ApplicationSessionPort;
  readonly conditionPolicy?: ScoringConditionPolicy;
}
