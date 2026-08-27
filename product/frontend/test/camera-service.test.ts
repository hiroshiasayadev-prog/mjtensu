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
    expect(drawImage).toHaveBeenCalledWith(
      video,
      0,
      0,
      640,
      360,
      0,
      0,
      640,
      360,
    );

    await session.stop();
    expect(video.srcObject).toBeNull();
    expect(session.captureLatest()).toBeNull();
    expect(track.stop).toHaveBeenCalledTimes(1);
  });

  it('quarter-turns a native 9:16 portrait frame into canonical 16:9 Recognition geometry', async () => {
    const stream = {
      getTracks: () => [{ stop: vi.fn() }],
    } as unknown as MediaStream;
    const drawImage = vi.fn();
    const save = vi.fn();
    const translate = vi.fn();
    const rotate = vi.fn();
    const restore = vi.fn();
    const service = createBrowserCameraService({
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
      } as unknown as Pick<MediaDevices, 'getUserMedia'>,
      createCanvas: () => ({
        width: 0,
        height: 0,
        getContext: () => ({ drawImage, save, translate, rotate, restore }),
      }) as unknown as HTMLCanvasElement,
      now: () => 4321,
    });
    const session = await service.open({ facingMode: 'environment' });
    const video = document.createElement('video');
    Object.defineProperties(video, {
      readyState: { configurable: true, value: 2 },
      videoWidth: { configurable: true, value: 720 },
      videoHeight: { configurable: true, value: 1280 },
    });
    video.play = vi.fn(async () => undefined);
    video.pause = vi.fn();

    session.preview.attach(video);
    const frame = session.captureLatest({
      aspectRatio: '9:16',
      rotation: -90,
    });

    expect(frame?.size).toEqual({ width: 1280, height: 720 });
    expect(save).toHaveBeenCalledTimes(1);
    expect(translate).toHaveBeenCalledWith(0, 720);
    expect(rotate).toHaveBeenCalledWith(-Math.PI / 2);
    expect(drawImage).toHaveBeenCalledWith(
      video,
      0,
      0,
      720,
      1280,
      0,
      0,
      720,
      1280,
    );
    expect(restore).toHaveBeenCalledTimes(1);
  });

  it('treats a transient first-frame draw failure as camera warmup instead of fatal Recognition failure', async () => {
    const stream = {
      getTracks: () => [{ stop: vi.fn() }],
    } as unknown as MediaStream;
    const drawImage = vi
      .fn()
      .mockImplementationOnce(() => {
        throw { name: 'InvalidStateError' };
      })
      .mockImplementation(() => undefined);
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
      now: () => 5678,
    });
    const session = await service.open({ facingMode: 'environment' });
    const video = document.createElement('video');
    Object.defineProperties(video, {
      readyState: { configurable: true, value: 2 },
      videoWidth: { configurable: true, value: 1280 },
      videoHeight: { configurable: true, value: 720 },
    });
    video.play = vi.fn(async () => undefined);
    video.pause = vi.fn();
    session.preview.attach(video);

    expect(session.captureLatest()).toBeNull();
    expect(session.captureLatest()).toEqual({
      image: canvases[1],
      size: { width: 1280, height: 720 },
      capturedAtMs: 5678,
    });
  });

  it('waits for attached video data before explicitly starting preview playback', async () => {
    const stream = {
      getTracks: () => [{ stop: vi.fn() }],
    } as unknown as MediaStream;
    const service = createBrowserCameraService({
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
      } as unknown as Pick<MediaDevices, 'getUserMedia'>,
    });
    const session = await service.open({ facingMode: 'environment' });
    const video = document.createElement('video');
    const play = vi.fn(async () => undefined);
    video.play = play;
    video.pause = vi.fn();

    session.preview.attach(video);
    expect(play).not.toHaveBeenCalled();

    video.dispatchEvent(new Event('loadeddata'));
    await Promise.resolve();

    expect(play).toHaveBeenCalledTimes(1);
  });

  it('retries a transient first preview play rejection without reopening the camera', async () => {
    const stream = {
      getTracks: () => [{ stop: vi.fn() }],
    } as unknown as MediaStream;
    const service = createBrowserCameraService({
      mediaDevices: {
        getUserMedia: vi.fn(async () => stream),
      } as unknown as Pick<MediaDevices, 'getUserMedia'>,
    });
    const session = await service.open({ facingMode: 'environment' });
    const video = document.createElement('video');
    Object.defineProperty(video, 'readyState', { configurable: true, value: 2 });
    const play = vi
      .fn<HTMLMediaElement['play']>()
      .mockRejectedValueOnce({ name: 'AbortError' })
      .mockResolvedValueOnce(undefined);
    video.play = play;
    video.pause = vi.fn();

    session.preview.attach(video);
    await Promise.resolve();
    await Promise.resolve();
    await Promise.resolve();

    expect(play).toHaveBeenCalledTimes(2);
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
