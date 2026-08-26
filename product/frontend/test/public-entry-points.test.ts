import * as app from '@/app';
import * as application from '@/application';
import * as camera from '@/camera';
import * as domain from '@/domain';
import * as recognition from '@/recognition';
import * as scoring from '@/scoring';
import * as ui from '@/ui';
import { describe, expect, it } from 'vitest';

describe('top-level public entry points', () => {
  it('resolves every architecture module through its public entry point', () => {
    expect(app).toHaveProperty('App');
    expect([domain, camera, recognition, scoring, application, ui]).toHaveLength(6);
  });
});
