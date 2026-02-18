import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { createHash } from 'crypto';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CACHE_DIR = join(__dirname, '..', 'cache');

function hashKey(key) {
  return createHash('md5').update(key).digest('hex');
}

export async function getCached(key) {
  const file = join(CACHE_DIR, `${hashKey(key)}.json`);
  try {
    const raw = await readFile(file, 'utf-8');
    const { data, expiry } = JSON.parse(raw);
    if (Date.now() < expiry) return data;
  } catch {}
  return null;
}

export async function setCache(key, data, ttlMs) {
  if (!existsSync(CACHE_DIR)) await mkdir(CACHE_DIR, { recursive: true });
  const file = join(CACHE_DIR, `${hashKey(key)}.json`);
  await writeFile(file, JSON.stringify({ data, expiry: Date.now() + ttlMs }));
}

export async function cachedFetch(url, options, ttlMs) {
  const cached = await getCached(url);
  if (cached) return cached;
  const res = await fetch(url, options);
  if (!res.ok) return null;
  const data = await res.json();
  await setCache(url, data, ttlMs);
  return data;
}
