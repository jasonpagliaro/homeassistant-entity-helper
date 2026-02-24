import process from 'node:process';
import { pathToFileURL } from 'node:url';

import { chromium } from 'playwright';

function formatError(error) {
  if (error instanceof Error && error.stack) {
    return error.stack;
  }
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
}

function resolveLaunchArgs() {
  if (typeof process.getuid === 'function' && process.getuid() === 0) {
    // Chromium requires this when executed as root in many CI containers.
    return ['--no-sandbox'];
  }
  return [];
}

export async function runPlaywrightPreflight() {
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      args: resolveLaunchArgs(),
    });
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto('data:text/html,<title>preflight</title><p>ok</p>');
    await page.waitForLoadState('load');
    await context.close();
    console.log('[preflight] Chromium launch succeeded. Browser runtime is ready.');
  } catch (error) {
    console.error('[preflight] Chromium launch failed.');
    console.error('[preflight] This harness intentionally does not install browsers at runtime.');
    console.error('[preflight] Rebuild or reprovision the auto-test runner image with baked Playwright + Chromium.');
    console.error(formatError(error));
    process.exit(1);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

const isDirectExecution =
  process.argv[1] !== undefined && pathToFileURL(process.argv[1]).href === import.meta.url;

if (isDirectExecution) {
  await runPlaywrightPreflight();
}
