// Real UI/API ownership with synthetic workers and injected browser transport faults.
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base = process.env.PIPELINE_QA_URL || 'http://127.0.0.1:5099';
assert.equal(new URL(base).hostname, '127.0.0.1');
assert.equal((await fetch(base + '/api/health')).headers.get('X-Verbaloom-QA'), 'isolated');
const output = process.env.PIPELINE_QA_OUTPUT || '/tmp/verbaloom-submission-browser';
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true,
  ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : {}) });
const page = await browser.newPage({ viewport: { width: 1365, height: 900 } });
const report = { browser: 'Browser plugin not available; Playwright with installed Chrome', checks: [], errors: [], console: [] };
page.on('pageerror', error => report.errors.push(error.message));
page.on('console', message => {
  if (['error', 'warning'].includes(message.type())) report.console.push(message.text());
});
page.on('dialog', dialog => dialog.accept());
const getJob = async id => (await page.request.get(`${base}/api/translation/${id}`)).json();
const receipt = () => page.evaluate(() => JSON.parse(sessionStorage.getItem('verbaloom.pending-submission.v1') || 'null'));
const current = () => page.evaluate(async () => {
  const job = (await import('/static/js/core/state-manager.js')).StateManager.getState('translation.currentJob');
  return job ? { translationId: job.translationId } : null;
});
let posts = 0;
page.on('request', request => { if (request.url().endsWith('/api/translate') && request.method() === 'POST') posts++; });
async function until(read, predicate, message) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (predicate(await read())) return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  assert.fail(message);
}
async function pauseFixtures() {
  const data = await (await page.request.get(base + '/api/translations')).json();
  for (const job of data.translations || []) {
    if (['running', 'queued', 'pricing_wait', 'provider_wait'].includes(job.status)) {
      await page.request.post(`${base}/api/translation/${job.translation_id}/interrupt`);
    }
  }
}
async function upload(name) {
  await page.locator('[data-tab="translate"]').click();
  await page.locator('#sourceLang').selectOption('English');
  await page.locator('#fileInput').setInputFiles({ name, mimeType: 'text/plain',
    buffer: Buffer.from('A private synthetic paragraph.\n\nAnother paragraph for the fixture.') });
  await page.waitForFunction(() => !document.getElementById('translateBtn').disabled);
}
try {
  await pauseFixtures();
  await page.goto(base, { waitUntil: 'networkidle' });
  assert.match(await page.title(), /VerbaLoom/);
  assert.equal(new URL(page.url()).origin, base);
  assert.ok((await page.locator('body').innerText()).includes('VerbaLoom'));
  assert.equal(await page.locator('vite-error-overlay, nextjs-portal').count(), 0);
  await upload('delayed-response.txt');
  let reached, release;
  const arrived = new Promise(resolve => { reached = resolve; });
  const gate = new Promise(resolve => { release = resolve; });
  await page.route('**/api/translate', async route => {
    const response = await route.fetch();
    reached(await response.json());
    await gate;
    await route.fulfill({ response });
  }, { times: 1 });
  await page.locator('#translateBtn').click();
  const accepted = await arrived;
  const pending = await receipt();
  assert.equal(pending.translationId, accepted.translation_id);
  assert.ok(!JSON.stringify(pending).includes('private synthetic paragraph'));
  assert.ok(!JSON.stringify(pending).includes('api_key'));
  const other = await (await page.request.post(base + '/api/translate', { data: {
    text: 'An independent synthetic book.', source_language: 'English', target_language: 'Spanish',
    model: 'fixture-model', llm_provider: 'ollama', output_filename: 'independent.txt', file_type: 'txt',
  } })).json();
  await page.locator('#interruptBtn').click();
  await until(() => getJob(accepted.translation_id), job => job.status === 'interrupted', 'Own job must stop');
  assert.equal((await getJob(other.translation_id)).status, 'running');
  release();
  await page.waitForResponse(response => response.url().endsWith('/api/translate') && response.request().method() === 'POST');
  assert.equal(await receipt(), null);
  assert.equal(posts, 1);
  report.checks.push('Cancel during delayed start response stops only its own job; late response cannot revive it');
  await pauseFixtures();
  await page.reload({ waitUntil: 'networkidle' });

  await upload('lost-response.txt');
  let lostId;
  let reachedLost;
  const lostReady = new Promise(resolve => { reachedLost = resolve; });
  await page.route('**/api/translate', async route => {
    const response = await route.fetch();
    lostId = (await response.json()).translation_id;
    reachedLost(lostId);
    await route.abort('failed');
  }, { times: 1 });
  await page.locator('#translateBtn').click();
  await lostReady;
  assert.ok(lostId);
  await until(current, job => job?.translationId === lostId, 'Lost response must attach the accepted job');
  assert.equal((await current()).translationId, lostId);
  assert.equal(await receipt(), null);
  assert.equal(posts, 2);
  await page.reload({ waitUntil: 'networkidle' });
  assert.equal((await current()).translationId, lostId);
  await page.locator('#interruptBtn').click();
  await until(() => getJob(lostId), job => job.status === 'interrupted', 'Recovered job must stop');
  report.checks.push('Lost accepted response reconciles by exact ID; reload attaches the same job without another POST');

  await page.reload({ waitUntil: 'networkidle' });
  await upload('uncertain-response.txt');
  let delayedPayload;
  await page.route('**/api/translate', async route => {
    delayedPayload = route.request().postDataJSON();
    await route.abort('failed');
  }, { times: 1 });
  await page.locator('#translateBtn').click();
  await page.waitForFunction(() => document.getElementById('translateBtn').dataset.i18n === 'settings:pipe_submission_check');
  assert.ok(await receipt());
  await page.locator('#translateBtn').click();
  assert.equal(posts, 3, 'Check submission must not send a new start request');
  for (const locale of ['en', 'fr', 'es', 'de', 'zh-CN', 'ja', 'ko']) {
    await page.evaluate(async value => (await import('/static/js/i18n/i18n.js')).setLocale(value), locale);
    assert.equal(await page.locator('#translateBtn').innerText(), await page.evaluate(async () =>
      (await import('/static/js/i18n/i18n.js')).t('settings:pipe_submission_check')));
    assert.equal(await page.locator('#messages .message-text').innerText(), await page.evaluate(async () =>
      (await import('/static/js/i18n/i18n.js')).t('settings:pipe_submission_unknown')));
  }
  await page.evaluate(async () => (await import('/static/js/i18n/i18n.js')).setLocale('es'));
  await page.locator('#messages').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/uncertain-desktop.png` });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#messages').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/uncertain-mobile.png` });
  assert.ok(await page.locator('#messages').evaluate(node => node.scrollWidth <= node.clientWidth + 1));
  report.checks.push('Unconfirmed submission shows bounded, reactive retry status in seven locales and fits mobile');
  await page.locator('#interruptBtn').click();
  await page.waitForFunction(() => JSON.parse(sessionStorage.getItem('verbaloom.pending-submission.v1')).cancelRequested);
  const unresolved = await receipt();
  await page.reload({ waitUntil: 'networkidle' });
  assert.deepEqual(await receipt(), unresolved);
  assert.equal(posts, 3);
  // Simulate the original POST reaching the server after the browser lost it.
  const delayed = await page.request.post(base + '/api/translate', { data: delayedPayload });
  assert.equal(delayed.status(), 200);
  assert.equal((await delayed.json()).translation_id, unresolved.translationId);
  await page.reload({ waitUntil: 'networkidle' });
  await until(() => getJob(unresolved.translationId), job => job.status === 'interrupted', 'Late request must honor pending pause');
  assert.equal(await receipt(), null);
  assert.equal(posts, 3);
  report.checks.push('Pause intent survives reload and stops a late-arriving request without resubmission');
  assert.deepEqual(report.errors, []);
  const unexpected = report.console.filter(text =>
    text !== '[CustomInstructions] No files found in response'
    && text !== '[SearchableSelect] No options found for #model, retrying...'
    && !text.includes('net::ERR_FAILED')
    && !text.includes('the server responded with a status of 404 (NOT FOUND)')
    && !text.includes('the server responded with a status of 404 (Not Found)'));
  assert.deepEqual(unexpected, []);
  report.consoleNotes = 'Intentional transport aborts and 404 lookups of unreceived submissions; empty instruction fixture and transient populated model selector.';
} catch (error) {
  report.failure = error.stack;
  await page.screenshot({ path: `${output}/failure.png` });
  process.exitCode = 1;
} finally {
  await pauseFixtures();
  await fs.writeFile(`${output}/browser-results.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
}
