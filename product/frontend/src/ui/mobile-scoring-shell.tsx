import type { CSSProperties, ReactNode } from 'react';

export interface MobileScoringPageShellProps {
  readonly title: string;
  readonly onBack?: () => void;
  readonly children: ReactNode;
  readonly bottomBar?: ReactNode;
  readonly bottomClearancePx?: number;
}

export interface PersistentBottomBarProps {
  readonly children: ReactNode;
  readonly ariaLabel: string;
}

const shellStyle: CSSProperties = {
  minHeight: '100dvh',
  background: '#f4f5f7',
  color: '#1b1d22',
};

const appBarStyle: CSSProperties = {
  position: 'sticky',
  top: 0,
  zIndex: 40,
  paddingTop: 'env(safe-area-inset-top)',
  borderBottom: '1px solid #e2e5e9',
  background: 'rgba(255, 255, 255, 0.97)',
  backdropFilter: 'blur(12px)',
};

const appBarInnerStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: '44px 1fr 44px',
  alignItems: 'center',
  minHeight: 52,
  maxWidth: 720,
  margin: '0 auto',
  padding: '0 max(12px, env(safe-area-inset-right)) 0 max(12px, env(safe-area-inset-left))',
};

const backButtonStyle: CSSProperties = {
  width: 40,
  height: 40,
  padding: 0,
  border: 0,
  borderRadius: 20,
  background: 'transparent',
  color: '#1b1d22',
  fontSize: 30,
  lineHeight: 1,
  cursor: 'pointer',
  touchAction: 'manipulation',
};

const titleStyle: CSSProperties = {
  margin: 0,
  fontSize: 18,
  fontWeight: 800,
  lineHeight: 1.2,
  textAlign: 'center',
};

const bottomBarOuterStyle: CSSProperties = {
  position: 'fixed',
  left: 0,
  right: 0,
  bottom: 0,
  zIndex: 50,
  borderTop: '1px solid #d9dde3',
  background: 'rgba(255, 255, 255, 0.97)',
  boxShadow: '0 -8px 24px rgba(20, 24, 32, 0.08)',
  backdropFilter: 'blur(12px)',
};

const bottomBarInnerStyle: CSSProperties = {
  maxWidth: 720,
  margin: '0 auto',
  padding:
    '10px max(16px, env(safe-area-inset-right)) calc(10px + env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left))',
};

export function MobileScoringPageShell({
  title,
  onBack,
  children,
  bottomBar,
  bottomClearancePx = 112,
}: MobileScoringPageShellProps) {
  return (
    <div data-testid="mobile-scoring-page-shell" style={shellStyle}>
      <header
        data-safe-area-top="true"
        data-testid="mobile-scoring-app-bar"
        style={appBarStyle}
      >
        <div style={appBarInnerStyle}>
          {onBack === undefined ? (
            <span aria-hidden="true" />
          ) : (
            <button
              aria-label="戻る"
              onClick={onBack}
              style={backButtonStyle}
              type="button"
            >
              ‹
            </button>
          )}
          <h1 style={titleStyle}>{title}</h1>
          <span aria-hidden="true" />
        </div>
      </header>

      <div
        data-bottom-clearance-px={bottomClearancePx}
        data-testid="mobile-scoring-scroll-content"
        style={{
          maxWidth: 720,
          margin: '0 auto',
          padding: `12px max(16px, env(safe-area-inset-right)) calc(${bottomClearancePx}px + env(safe-area-inset-bottom)) max(16px, env(safe-area-inset-left))`,
        }}
      >
        {children}
      </div>

      {bottomBar}
    </div>
  );
}

export function PersistentBottomBar({
  children,
  ariaLabel,
}: PersistentBottomBarProps) {
  return (
    <aside
      aria-label={ariaLabel}
      data-safe-area-bottom="true"
      data-testid="persistent-bottom-bar"
      style={bottomBarOuterStyle}
    >
      <div style={bottomBarInnerStyle}>{children}</div>
    </aside>
  );
}
