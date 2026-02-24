import process from 'node:process';

import { chromium } from 'playwright';

import { runPlaywrightPreflight } from './playwright-preflight.mjs';

const VALID_MODES = new Set(['required', 'skip']);

function normalizeMode(rawMode) {
  const cleaned = (rawMode || 'required').trim().toLowerCase();
  if (!VALID_MODES.has(cleaned)) {
    console.error(
      `[config-timestamp-check] Invalid BROWSER_CHECK_MODE='${rawMode}'. Use one of: required, skip.`,
    );
    process.exit(64);
  }
  return cleaned;
}

async function assertRenderedLocalTimestamps(configUrl) {
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      args:
        typeof process.getuid === 'function' && process.getuid() === 0
          ? ['--no-sandbox']
          : [],
    });
    const page = await browser.newPage();
    const response = await page.goto(configUrl, { waitUntil: 'domcontentloaded' });
    if (!response || !response.ok()) {
      throw new Error(
        `Expected 2xx from ${configUrl}, got ${response ? response.status() : 'no response'}.`,
      );
    }

    await page.waitForFunction(() => {
      const nodes = Array.from(document.querySelectorAll('time[data-local-datetime][datetime]'));
      if (!nodes.length) {
        return false;
      }
      return nodes.every((node) => {
        const raw = (node.getAttribute('datetime') || '').trim();
        const rendered = (node.textContent || '').trim();
        return rendered.length > 0 && rendered !== raw;
      });
    }, { timeout: 5000 });

    const summary = await page.evaluate(() => {
      const nodes = Array.from(document.querySelectorAll('time[data-local-datetime][datetime]'));
      const invalid = [];
      for (const node of nodes) {
        const raw = (node.getAttribute('datetime') || '').trim();
        const rendered = (node.textContent || '').trim();
        if (!raw || !rendered || raw === rendered) {
          invalid.push({ raw, rendered });
        }
      }
      return { count: nodes.length, invalid };
    });

    if (summary.count < 1) {
      throw new Error('Expected at least one time[data-local-datetime][datetime] node.');
    }
    if (summary.invalid.length > 0) {
      throw new Error(
        `Found ${summary.invalid.length} node(s) that did not render localized text: ${JSON.stringify(summary.invalid)}`,
      );
    }

    console.log(
      `[config-timestamp-check] Rendered timestamp check passed (${summary.count} localized time node(s)).`,
    );
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

function assertRawHtmlFallback(html, configUrl) {
  const timeTags = html.match(/<time\b[^>]*>/gi) || [];
  const markers = timeTags.filter(
    (tag) => /\bdata-local-datetime\b/i.test(tag) && /\bdatetime\s*=\s*"[^"]+"/i.test(tag),
  );
  if (markers.length < 1) {
    throw new Error(
      `Expected at least one <time> marker with data-local-datetime + datetime in ${configUrl}.`,
    );
  }
  console.log(
    `[config-timestamp-check] HTML marker fallback passed (${markers.length} marker node(s)).`,
  );
}

async function assertHtmlMarkerFallback(configUrl) {
  const response = await fetch(configUrl);
  if (!response.ok) {
    throw new Error(`Expected 2xx from ${configUrl}, got ${response.status}.`);
  }
  const html = await response.text();
  assertRawHtmlFallback(html, configUrl);
}

async function main() {
  const configUrl = process.env.CONFIG_URL || 'http://127.0.0.1:8000/config';
  const mode = normalizeMode(process.env.BROWSER_CHECK_MODE);

  if (mode === 'required') {
    await runPlaywrightPreflight();
    await assertRenderedLocalTimestamps(configUrl);
    return;
  }

  console.log('[config-timestamp-check] Browser check intentionally skipped; running HTML marker fallback.');
  await assertHtmlMarkerFallback(configUrl);
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.stack || error.message : String(error);
  console.error('[config-timestamp-check] Check failed.');
  console.error(message);
  process.exit(1);
}
