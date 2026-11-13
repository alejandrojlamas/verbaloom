const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

// Only run against the isolated fixture server, never a production job service.
const base = process.env.PIPELINE_QA_URL || 'http://127.0.0.1:5099';
assert.equal(new URL(base).hostname, '127.0.0.1');
const probe = await fetch(base + '/api/health');
assert.equal(probe.headers.get('X-Verbaloom-QA'), 'isolated');
const output = process.env.PIPELINE_QA_OUTPUT || '/tmp/verbaloom-pipeline-browser';
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : {}) });
const report = { browser: 'Browser plugin not available; Playwright with installed Chrome', checks: [], errors: [], console: [] };
const page = await browser.newPage({ viewport: { width: 1365, height: 900 } });
page.on('pageerror', error => report.errors.push(error.message));
page.on('console', message => {
  if (['error', 'warning'].includes(message.type())) report.console.push({ level: message.type(), text: message.text() });
});
page.on('response', response => { if (response.status() >= 500) report.errors.push(`${response.status()} ${response.url()}`); });
const storage = () => page.evaluate(() => JSON.parse(localStorage.getItem('verbaloom.pipeline.v1') || 'null'));
const snapshot = () => page.evaluate(async () => {
  try { return { value: (await import('/static/js/ui/pipeline-settings.js')).PipelineSettings.snapshot() }; }
  catch (error) { return { error: error.message }; }
});
async function settings() {
  await page.locator('[data-tab="settings"]').click();
  if (!await page.locator('#pipelinePreset').isVisible()) await page.locator('[onclick="toggleSettingsOptions()"]').click();
}
async function preset(name) {
  await page.locator('#pipelinePreset').selectOption(name);
  await page.waitForFunction(name => {
    const data = JSON.parse(localStorage.getItem('verbaloom.pipeline.v1') || 'null');
    return name === 'legacy' ? !data : data?.preset === name;
  }, name);
}
async function waitForJobStatus(id, status) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    const job = await (await page.request.get(`${base}/api/translation/${id}`)).json();
    if (job.status === status) return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  assert.fail(`Job ${id} did not reach ${status}`);
}
try {
  await page.goto(base, { waitUntil: 'networkidle' });
  assert.match(await page.title(), /VerbaLoom/);
  assert.equal(new URL(page.url()).origin, base);
  assert.ok((await page.locator('body').innerText()).includes('VerbaLoom'));
  assert.equal(await page.locator('vite-error-overlay, nextjs-portal').count(), 0);
  report.checks.push('Correct page, meaningful content, no framework overlay');
  await page.screenshot({ path: `${output}/home-desktop.png` });
  await settings();
  for (const name of ['compatible', 'openai_equivalent', 'deepseek_source_audited', 'openai_source_audited']) {
    await preset(name);
    const value = await storage();
    assert.equal(Object.keys(value.roles).length, 7);
    assert.equal(value.review_policy, name.endsWith('source_audited') ? 'source_audited_v1' : 'compatible');
    assert.ok(Object.values(value.roles).every(role => role.reasoning_effort === 'none'));
  }
  report.checks.push('All four presets preserve seven explicit roles, policy and none effort');
  assert.ok(await page.locator('#openaiApiKeyGroup').isVisible(), 'OpenAI credentials must be reachable with an OpenAI role preset');
  report.checks.push('Role-specific credentials visible without changing baseline');
  await page.locator('.pipeline-roles summary').click();
  await page.locator('#pipeline-translate-effort').selectOption('low');
  assert.equal((await storage()).roles.translate.reasoning_effort, 'low');
  await page.locator('#pipelineBudgetEnabled').check();
  await page.locator('#pipelineBudgetUsd').fill('');
  await page.locator('#pipelineMaxCalls').focus();
  assert.ok((await snapshot()).error, 'Empty budget must block job creation locally');
  assert.equal((await storage()).budget.limit_usd, '5', 'Invalid edit must not replace valid persisted budget');
  await page.locator('#pipelineBudgetUsd').fill('1.25');
  await page.locator('#pipelineMaxCalls').fill('12');
  await page.locator('#pipelinePreflight').focus();
  assert.deepEqual((await storage()).budget, { limit_usd: '1.25', max_calls: 12 });
  await page.locator('#pipeline-audit-provider').selectOption('deepseek');
  await page.locator('#pipeline-audit-model').selectOption('deepseek-v4-pro');
  assert.ok(await page.locator('#deepseekSettings').isVisible());
  assert.ok(await page.locator('#openaiApiKeyGroup').isVisible());
  report.checks.push('Hybrid role editing, explicit low effort and finite budget validation');
  const frozen = await storage();
  await page.reload({ waitUntil: 'networkidle' });
  await settings();
  await page.waitForFunction(() => document.querySelectorAll('[data-role]').length === 7);
  assert.deepEqual(await storage(), frozen);
  assert.deepEqual((await snapshot()).value, frozen);
  report.checks.push('Settings survive reload with no provider credentials in snapshot');
  assert.ok(!JSON.stringify(frozen).includes('api_key'));
  await page.locator('#uiLocaleSelect').selectOption('en');
  await page.waitForFunction(() => document.getElementById('pipelineHeading').textContent === 'Models and review');
  await page.locator('#uiLocaleSelect').selectOption('es');
  await page.waitForFunction(() => document.getElementById('pipelineHeading').textContent === 'Modelos y revisión');
  await page.locator('#pipelinePreflight').click();
  await page.waitForFunction(() => document.getElementById('pipelineStatus').dataset.i18n === 'settings:pipe_visible');
  assert.match(await page.locator('#pipelineStatus').innerText(), /saldo/);
  report.checks.push('Reactive language switch and nonbillable access result with unknown balance');
  await page.locator('#pipelineSettings').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/settings-desktop.png` });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#pipelineSettings').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/settings-mobile.png` });
  assert.ok(await page.locator('#pipelineSettings').evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Pipeline surface must not overflow');
  await page.locator('.pipeline-roles summary').click();
  await page.locator('[data-role="profile_prepare"]').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/settings-roles-mobile.png` });
  assert.ok(await page.locator('.pipeline-role').evaluateAll(nodes => nodes.every(node => node.scrollWidth <= node.clientWidth + 1)));
  await page.locator('.pipeline-roles summary').click();
  report.checks.push('Mobile settings fit width');
  await page.setViewportSize({ width: 1365, height: 900 });
  await page.evaluate(async () => (await import('/static/js/utils/status-manager.js')).StatusManager.setDisconnected('Synthetic baseline outage'));
  await page.locator('[data-tab="translate"]').click();
  await page.locator('#sourceLang').selectOption('English');
  await page.locator('#fileInput').setInputFiles({ name: 'pipeline-browser-fixture.txt', mimeType: 'text/plain', buffer: Buffer.from('A small original paragraph.\n\nA second original paragraph.') });
  await page.waitForFunction(() => !document.getElementById('translateBtn').disabled, null, { timeout: 12000 });
  report.checks.push('Unavailable baseline provider does not block configured role provider');
  page.on('dialog', dialog => dialog.accept());
  let releasePreflight, reachedPreflight;
  const reached = new Promise(resolve => { reachedPreflight = resolve; });
  const gate = new Promise(resolve => { releasePreflight = resolve; });
  let submitted = 0;
  page.on('request', request => {
    if (request.url().endsWith('/api/translate') && request.method() === 'POST') submitted++;
  });
  await page.route('**/api/pipeline/preflight', async route => {
    reachedPreflight(route.request());
    await gate;
    await route.continue();
  }, { times: 1 });
  await page.locator('#translateBtn').click();
  const heldRequest = await reached;
  assert.deepEqual(Object.keys(heldRequest.postDataJSON()).sort(), ['deepseek_api_key', 'openai_api_key', 'pipeline', 'roles']);
  assert.ok(!heldRequest.postData().includes('original paragraph'), 'Preflight must not transmit book text');
  await page.locator('#interruptBtn').click();
  await page.waitForFunction(() => !document.getElementById('translateBtn').disabled);
  assert.equal(submitted, 0, 'Cancellation before preflight must not create a job');
  const startedPromise = page.waitForResponse(response => response.url().endsWith('/api/translate') && response.request().method() === 'POST');
  await page.locator('#translateBtn').click();
  const started = await startedPromise;
  const startPayload = await started.json();
  assert.equal(started.status(), 200, JSON.stringify(startPayload));
  const runId = startPayload.translation_id;
  const staleResponse = page.waitForResponse(response => response.request() === heldRequest);
  releasePreflight();
  await staleResponse;
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  assert.equal(submitted, 1, 'A stale preflight must not start a second job after cancellation/restart');
  report.checks.push('Cancellation and restart fence stale preflight; access check contains no book text');
  assert.deepEqual(startPayload.config_received.pipeline, frozen);
  assert.ok(!JSON.stringify(startPayload).includes('YOUR_API_KEY_HERE'));
  await page.locator('#interruptBtn').click();
  await waitForJobStatus(runId, 'interrupted');
  const resume = page.locator(`button[onclick="resumeJob('${runId}')"]`);
  await resume.waitFor({ state: 'visible', timeout: 15000 });
  await settings();
  await preset('compatible');
  assert.notDeepEqual(await storage(), frozen);
  await page.locator('[data-tab="translate"]').click();
  const resumePromise = page.waitForResponse(response => response.url().endsWith(`/api/resume/${runId}`));
  await resume.click();
  const resumed = await resumePromise;
  assert.equal(resumed.status(), 200, await resumed.text());
  const saved = await (await page.request.get(`${base}/api/translation/${runId}`)).json();
  assert.deepEqual(saved.config.pipeline, frozen);
  assert.equal(saved.stats.completed_chunks, 1);
  assert.equal(saved.status, 'running');
  report.checks.push('UI upload/start/pause/resume preserves role snapshot and accepted chunk despite global edits');
  await page.screenshot({ path: `${output}/resumed-desktop.png` });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator('#progressSection').scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/resumed-mobile.png` });
  await page.locator('#interruptBtn').click();
  await waitForJobStatus(runId, 'interrupted');
  await page.setViewportSize({ width: 1365, height: 900 });
  await settings();
  for (const [failure, key] of [
    ['authentication_error', 'access_auth'], ['permission_denied', 'access_denied'],
    ['model_unavailable', 'access_model'], ['rate_limited', 'access_rate'], ['connection_error', 'access_connection'],
  ]) {
    await page.route('**/api/pipeline/preflight', route => route.fulfill({ json: {
      status: 'not_ready', providers: [{ status: 'not_ready', models: [{ status: failure }] }],
      generation_tested: false, balance: null,
    } }), { times: 1 });
    await page.locator('#pipelinePreflight').click();
    await page.waitForFunction(key => document.getElementById('pipelineStatus').dataset.i18n === `settings:pipe_${key}`, key);
    if (failure === 'authentication_error') {
      await page.locator('#pipelineStatus').scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${output}/access-error-desktop.png` });
    }
  }
  const localized = {};
  for (const locale of ['en', 'fr', 'es', 'de', 'zh-CN', 'ja', 'ko']) {
    localized[locale] = await (await page.request.get(`${base}/static/locales/${locale}/settings.json`)).json();
    await page.locator('#uiLocaleSelect').selectOption(locale);
    await page.waitForFunction(expected => document.getElementById('pipelineStatus').textContent === expected,
      localized[locale].pipe_access_connection);
    assert.equal(await page.locator('#pipelineHeading').innerText(), localized[locale].pipe_title);
    assert.equal(await page.locator('#providerStatusText').innerText(), localized[locale].pipe_roles_status);
  }
  await page.locator('#uiLocaleSelect').selectOption('es');
  report.checks.push('Access failure categories and role status update reactively in all seven locales');

  await preset('openai_equivalent');
  await page.locator('.pipeline-roles summary').click();
  await page.locator('#pipeline-profile_prepare-provider').selectOption('deepseek');
  await page.locator('#pipeline-profile_review-provider').selectOption('deepseek');
  const profileSnapshot = await storage();
  await page.locator('[data-tab="profile-prep"]').click();
  await page.locator('#profilePrepFileInput').setInputFiles({ name: 'profile-fixture.txt', mimeType: 'text/plain', buffer: Buffer.from('A private profile fixture.') });
  const profileAccess = page.waitForRequest(request => request.url().endsWith('/api/pipeline/preflight'));
  let profileForm;
  await page.route('**/api/book-profiles/prepare-jobs', async route => {
    if (route.request().method() !== 'POST') return route.continue();
    profileForm = await new Response(route.request().postDataBuffer(), {
      headers: { 'Content-Type': route.request().headers()['content-type'] },
    }).formData();
    await route.fulfill({ status: 400, json: { error: 'Synthetic preparation stop; no provider generation' } });
  });
  await page.locator('#profilePrepStartBtn').click();
  const profileRequest = (await profileAccess).postDataJSON();
  assert.deepEqual(profileRequest.roles, ['profile_prepare', 'profile_review']);
  assert.ok(!('openai_api_key' in profileRequest));
  await page.waitForFunction(() => !document.getElementById('profilePrepStartBtn').disabled);
  assert.deepEqual(JSON.parse(profileForm.get('pipeline')), profileSnapshot);
  assert.equal(profileForm.has('openai_api_key'), false);
  assert.ok(profileForm.has('deepseek_api_key'));
  const label = await page.locator('#profilePrepModelLabel').innerText();
  assert.match(label, /Preparación: deepseek/);
  assert.match(label, /Revisión: deepseek/);
  assert.ok(!label.includes('gpt-6'));
  await page.locator('#uiLocaleSelect').selectOption('en');
  await page.waitForFunction(() => document.getElementById('profilePrepModelLabel').textContent.startsWith('Preparation:'));
  report.checks.push('Profile form/preflight use only profile credentials and show reactive frozen model labels');
  await page.unroute('**/api/book-profiles/prepare-jobs');
  await page.evaluate(() => localStorage.setItem('verbaloom.pipeline.v1', '{invalid'));
  await page.reload({ waitUntil: 'networkidle' });
  await settings();
  await page.waitForFunction(() => document.getElementById('pipelineStatus').dataset.i18n === 'settings:pipe_invalid');
  assert.equal(await page.locator('#pipelinePreset').inputValue(), 'custom');
  assert.ok((await snapshot()).error, 'Corrupted saved configuration must not silently use legacy models');
  await preset('legacy');
  assert.equal((await snapshot()).value, null);
  await page.waitForFunction(() => [...document.getElementById('model').options].some(option => option.value === 'deepseek-v4-pro'));
  report.checks.push('Corrupt saved settings fail closed; explicit legacy rollback remains available');
  const unexpectedConsole = report.console.filter(item =>
    item.text !== '[CustomInstructions] No files found in response'
    && item.text !== '[SearchableSelect] No options found for #model, retrying...'
    && !item.text.includes('the server responded with a status of 400 (Bad Request)'));
  assert.deepEqual(unexpectedConsole, [], 'Unexpected browser console errors or warnings');
  report.consoleNotes = 'Empty instruction list warnings are expected in the fixture. A transient model-selector retry is allowed only with verified loaded model options. One HTTP 400 is the deliberately intercepted profile start.';
  assert.deepEqual(report.errors, []);
} catch (error) {
  report.failure = error.stack;
  await page.screenshot({ path: `${output}/failure.png` });
  process.exitCode = 1;
} finally {
  await fs.writeFile(`${output}/browser-results.json`, JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  await browser.close();
}
