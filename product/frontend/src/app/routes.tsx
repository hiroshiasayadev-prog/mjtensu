import { Route, Routes } from 'react-router-dom';

import {
  appRoutePaths,
  ConditionsPage,
  HelpPage,
  ProductionShell,
  RecognitionPage,
  RequireActiveScoringSession,
  ResultPage,
  TopPage,
} from '@/ui';

export const productionRouteTable = [
  {
    name: 'top',
    path: appRoutePaths.top,
    requiresActiveScoringSession: false,
  },
  {
    name: 'recognition',
    path: appRoutePaths.recognition,
    requiresActiveScoringSession: false,
  },
  {
    name: 'conditions',
    path: appRoutePaths.conditions,
    requiresActiveScoringSession: true,
  },
  {
    name: 'result',
    path: appRoutePaths.result,
    requiresActiveScoringSession: true,
  },
  {
    name: 'help',
    path: appRoutePaths.help,
    requiresActiveScoringSession: false,
  },
] as const;

export function AppRoutes() {
  return (
    <Routes>
      <Route element={<ProductionShell />}>
        <Route path={appRoutePaths.top} element={<TopPage />} />
        <Route path={appRoutePaths.recognition} element={<RecognitionPage />} />
        <Route
          path={appRoutePaths.conditions}
          element={
            <RequireActiveScoringSession>
              <ConditionsPage />
            </RequireActiveScoringSession>
          }
        />
        <Route
          path={appRoutePaths.result}
          element={
            <RequireActiveScoringSession>
              <ResultPage />
            </RequireActiveScoringSession>
          }
        />
        <Route path={appRoutePaths.help} element={<HelpPage />} />
      </Route>
    </Routes>
  );
}
