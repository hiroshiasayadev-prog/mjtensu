import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { App } from '@/app';

describe('application bootstrap', () => {
  it('renders the application shell without production feature services', () => {
    render(<App />);

    expect(screen.getByRole('heading', { name: 'mjtensu' })).toBeVisible();
    expect(screen.getByRole('button', { name: '判定する' })).toBeVisible();
  });
});
