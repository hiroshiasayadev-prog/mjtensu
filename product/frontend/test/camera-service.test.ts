import { createBrowserCameraService } from '@/camera';
import { describe, expect, it, vi } from 'vitest';

describe('browser CameraService', () => {
  it('requests the environment camera with the accepted ideal capture preference', async () => {
    const track = { stop: vi.fn() };
    const stream = {
      getTracks: () => [track],
    } as unknown as MediaStream;
    const getUserMedia = vi.fn(async () => stream);
    const service = createBrowserCameraService({
      mediaDevices: { getUserMedia } as Pick<MediaDevices, 'getUserMedia'>,
    });

    const session = await service.open({ facingMode: 'environment' });

    expect(getUserMedia).toHaveBeenCalledWith({
      video: {
        facingMode: { ideal: 'environment' },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    await session.stop();
    await session.stop();
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it('copies the attached current video frame and releases preview ownership on stop', async () => {
    const track = { stop: vi.fn() };
    const stream = {
      getTracks: () => [track],
    } as unknown as MediaStream;
    const drawImage = vi.fn();
    const canvases: HTMLCanvasElement[] = [];
    const service = createBrowserCameraService({
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
      } as unknown as Pick<MediaDevices, 'getUserMedia'>,
      createCanvas: () => {
        const canvas = {
          width: 0,
          height: 0,
          getContext: () => ({ drawImage }),
        } as unknown as HTMLCanvasElement;
        canvases.push(canvas);
        return canvas;
      },
      now: () => 1234,
    });
    const session = await service.open({ facingMode: 'environment' });
    const video = document.createElement('video');
    Object.defineProperties(video, {
      readyState: { configurable: true, value: 2 },
      videoWidth: { configurable: true, value: 640 },
      videoHeight: { configurable: true, value: 360 },
    });
    video.play = vi.fn(async () => undefined);
    video.pause = vi.fn();

    session.preview.attach(video);
    const frame = session.captureLatest();

    expect(frame).toEqual({
      image: canvases[0],
      size: { width: 640, height: 360 },
      capturedAtMs: 1234,
    });
    expect(drawImage).toHaveBeenCalledWith(video, 0, 0, 640, 360);

    await session.stop();
    expect(video.srcObject).toBeNull();
    expect(session.captureLatest()).toBeNull();
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it('normalizes browser camera failures at the Camera boundary', async () => {
    const service = createBrowserCameraService({
      mediaDevices: {
        getUserMedia: vi.fn(async () => {
          throw { name: 'NotAllowedError' };
        }),
      } as unknown as Pick<MediaDevices, 'getUserMedia'>,
    });

    await expect(service.open({ facingMode: 'environment' })).rejects.toEqual({
      kind: 'permission-denied',
    });
  });
});
