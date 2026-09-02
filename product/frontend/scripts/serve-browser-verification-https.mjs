import { createReadStream, existsSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:https';
import { extname, join, normalize, resolve, sep } from 'node:path';

const options = parseArgs(process.argv.slice(2));
const root = resolve(options.root);
const cert = readFileSync(resolve(options.cert));
const key = readFileSync(resolve(options.key));

const mimeTypes = {
  '.css': 'text/css; charset=utf-8',
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.wasm': 'application/wasm',
  '.webmanifest': 'application/manifest+json',
};

const server = createServer({ cert, key }, (request, response) => {
  try {
    const requestUrl = new URL(request.url ?? '/', `https://${request.headers.host ?? 'localhost'}`);
    const decoded = decodeURIComponent(requestUrl.pathname);
    const relative = decoded.replace(/^\/+/, '');
    const normalized = normalize(relative || 'index.html');
    const target = resolve(join(root, normalized));
    if (target !== root && !target.startsWith(`${root}${sep}`)) {
      response.writeHead(403).end('Forbidden');
      return;
    }

    const file = resolveFile(target);
    if (file === null) {
      response.writeHead(404).end('Not Found');
      return;
    }

    const contentType = mimeTypes[extname(file).toLowerCase()] ?? 'application/octet-stream';
    response.writeHead(200, {
      'Cache-Control': 'no-store',
      'Content-Type': contentType,
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    });
    createReadStream(file).pipe(response);
  } catch (error) {
    response.writeHead(500).end(error instanceof Error ? error.message : String(error));
  }
});

server.listen(options.port, options.host, () => {
  console.log(`Serving ${root}`);
  console.log(`HTTPS: https://${options.host}:${options.port}/`);
  console.log(
    `Benchmark: https://${options.host}:${options.port}/test/e2e/mobile-classifier-benchmark.html`,
  );
});

function resolveFile(target) {
  if (!existsSync(target)) {
    return null;
  }
  const stat = statSync(target);
  if (stat.isFile()) {
    return target;
  }
  if (stat.isDirectory()) {
    const index = join(target, 'index.html');
    return existsSync(index) && statSync(index).isFile() ? index : null;
  }
  return null;
}

function parseArgs(args) {
  const result = {
    root: 'dist',
    cert: '',
    key: '',
    host: '0.0.0.0',
    port: 8443,
  };
  for (let index = 0; index < args.length; index += 1) {
    const name = args[index];
    const value = args[index + 1];
    if (name === '--root' && value) {
      result.root = value;
      index += 1;
    } else if (name === '--cert' && value) {
      result.cert = value;
      index += 1;
    } else if (name === '--key' && value) {
      result.key = value;
      index += 1;
    } else if (name === '--host' && value) {
      result.host = value;
      index += 1;
    } else if (name === '--port' && value) {
      result.port = Number.parseInt(value, 10);
      index += 1;
    } else {
      throw new Error(`Unknown or incomplete argument: ${name ?? '<missing>'}`);
    }
  }
  if (!result.cert || !result.key) {
    throw new Error('Usage: --cert <cert.pem> --key <key.pem> [--root dist] [--host 0.0.0.0] [--port 8443]');
  }
  if (!Number.isInteger(result.port) || result.port < 1 || result.port > 65535) {
    throw new Error(`Invalid --port: ${result.port}`);
  }
  return result;
}
