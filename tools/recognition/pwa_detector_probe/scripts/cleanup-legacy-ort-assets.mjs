import { rm } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const projectDirectory = dirname(scriptDirectory);
const legacyDirectory = join(projectDirectory, 'public', 'ort');

await rm(legacyDirectory, { recursive: true, force: true });
console.log('Removed legacy public/ort artifacts; Vite now bundles provider-specific ORT assets.');
