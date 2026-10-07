import { MantineProvider } from '@mantine/core';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { IntegrationCapturePage } from '@/ui';

const originalMediaDevices = Object.getOwnPropertyDescriptor(navigator, 'mediaDevices');

function setViewportSize(width: number, height: number): void {
  Object.defineProperties(window, {
    innerWidth: { configurable: true, value: width },
    innerHeight: { configurable: true, value: height },
  });
}

function createCameraStream() {
  const stop = vi.fn();
  const track = {
    stop,
    getSettings: () => ({
      width: 720,
      height: 1280,
      frameRate: 30,
      facingMode: 'environment',
    }),
  };
  return {
    stream: {
      getTracks: () => [track],
      getVideoTracks: () => [track],
    } as unknown as MediaStream,
    stop,
  };
}
function renderPage() {
  render(
    <MantineProvider>
      <MemoryRouter>
        <IntegrationCapturePage />
      </MemoryRouter>
    </MantineProvider>,
  );
}

beforeEach(() => {
  setViewportSize(844, 390);
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
  if (originalMediaDevices === undefined) {
    Reflect.deleteProperty(navigator, 'mediaDevices');
  } else {
    Object.defineProperty(navigator, 'mediaDevices', originalMediaDevices);
  }
});

describe('IntegrationCapturePage', () => {
  it('uses the production recognition guides and prepares a 30 second take', async () => {
    const camera = createCameraStream();
    const getUserMedia = vi.fn(async () => camera.stream);
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia },
    });

    const { unmount } = render(
      <MantineProvider>
        <MemoryRouter>
          <IntegrationCapturePage />
        </MemoryRouter>
      </MantineProvider>,
    );
    expect(screen.getByLabelText('ドラ認識領域')).toBeVisible();
    expect(screen.getByLabelText('手牌認識領域')).toBeVisible();
    expect(screen.getByLabelText('副露認識領域')).toBeVisible();

    await waitFor(() => expect(screen.getByText(/撮影準備OK/)).toBeVisible());
    expect(screen.getByRole('button', { name: 'REC 30s' })).toBeEnabled();
    expect(screen.getByText(/Take 1 \/ 5/)).toBeVisible();
    expect(getUserMedia).toHaveBeenCalledWith({
      audio: false,
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
    });

    unmount();
    expect(camera.stop).toHaveBeenCalledTimes(1);
  });

  it('uses the same rotated logical landscape surface in a portrait viewport', async () => {
    setViewportSize(390, 844);
    const camera = createCameraStream();
    Object.defineProperty(navigator, 'mediaDevices', {
      configurable: true,
      value: { getUserMedia: vi.fn(async () => camera.stream) },
    });

    renderPage();

    await waitFor(() => expect(screen.getByText(/撮影準備OK/)).toBeVisible());
    const captureSurface = screen.getByTestId('integration-capture-surface');
    const landscapeUi = screen.getByTestId('integration-capture-landscape-ui');

    expect(captureSurface.style.width).toBe('min(100vw, 56.25dvh)');
    expect(captureSurface.style.height).toBe('min(100dvh, 177.7778vw)');
    expect(landscapeUi.style.transform).toContain('rotate(90deg)');
  });
});
