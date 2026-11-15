// Use only with pipeline_settings_server.py: real UI, simulated provider IO.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.PIPELINE_QA_URL || 'http://127.0.0.1:5099';
assert.equal(new URL(base).hostname, '127.0.0.1');
assert.equal((await fetch(`${base}/api/health`)).headers.get('X-Verbaloom-QA'), 'isolated');
const output = process.env.PIPELINE_QA_OUTPUT || '/tmp/verbaloom-profile-models-browser';
await fs.mkdir(output, { recursive: true });
const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : {}),
});
const report = { checks: [], errors: [] };
const page = await browser.newPage();
page.on('pageerror', error => report.errors.push(error.message));
try {
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForFunction(() => document.getElementById('pipelineStatus').dataset.i18n !== 'settings:pipe_loading');
    const cases = await page.evaluate(async () => {
        const { ProfilePrepManager: manager } = await import('/static/js/translation/profile-prep-manager.js?v=20261114-openai-primary');
        const provider = document.getElementById('llmProvider');
        const endpoint = document.getElementById('openaiEndpoint');
        const original = [provider.value, endpoint.value];
        const selections = [];
        for (const [name, url] of [
            ['official', 'https://api.openai.com/v1/chat/completions'],
            ['official-port', 'https://api.openai.com:443/v1/chat/completions'],
            ['custom', 'http://127.0.0.1:1234/v1/chat/completions'],
            ['spoofed', 'https://api.openai.com.example.invalid/v1/chat/completions'],
            ['userinfo', 'https://user@api.openai.com/v1/chat/completions'],
        ]) {
            provider.value = 'openai'; endpoint.value = url;
            selections.push([name, manager.analysisModels(null)]);
        }
        selections.push(['hybrid', manager.analysisModels({ roles: {
            profile_prepare: { provider: 'deepseek', model: 'deepseek-flash' },
            profile_review: { provider: 'openai', model: 'gpt-6.1-terra' },
        } })]);
        [provider.value, endpoint.value] = original;
        manager.refreshModelLabels();
        return selections;
    });
    for (const [name, selected] of cases) {
        if (name.startsWith('official')) {
            assert.equal(selected.provider, 'openai');
            assert.equal(selected.model, 'gpt-6-luna');
            assert.equal(selected.keyField, 'openaiApiKey');
        } else if (name === 'hybrid') {
            assert.equal(selected.provider, 'deepseek');
            assert.equal(selected.reviewProvider, 'openai');
            assert.equal(selected.reviewModel, 'gpt-6.1-terra');
        } else {
            assert.equal(selected.provider, 'deepseek');
            assert.equal(selected.keyField, 'deepseekApiKey');
        }
    }
    report.checks.push('Official endpoint, HTTPS port, custom and spoofed hosts, and hybrid role identities');
    const labels = [];
    for (const locale of ['en', 'es', 'fr', 'de', 'ja', 'ko', 'zh-CN']) {
        await page.locator('#uiLocaleSelect').selectOption(locale);
        await page.waitForFunction(locale => document.documentElement.lang === locale, locale);
        const label = await page.locator('[data-i18n="transform:profile_prep_button"]').textContent();
        assert.ok(label.includes('GPT-6 Luna'), `${locale}: ${label}`);
        assert.ok(!label.includes('{{'), `${locale}: unresolved model placeholder`);
        const strict = await page.locator('[data-i18n="settings:profile_strength_strict"]').textContent();
        assert.ok(!/DeepSeek|GPT|\{\{/.test(strict), `${locale}: strict audit must not claim the profile-review model`);
        labels.push(label);
    }
    assert.equal(new Set(labels).size, 7);
    report.checks.push('Seven locale switches update model-aware labels without reload');
    await page.locator('[data-tab="settings"]').click();
    if (!await page.locator('#pipelinePreset').isVisible()) await page.locator('[onclick="toggleSettingsOptions()"]').click();
    await page.locator('#pipelinePreset').selectOption('compatible');
    await page.waitForFunction(() => document.querySelector('[data-i18n="transform:profile_prep_button"]').textContent.includes('DeepSeek Flash'));
    report.checks.push('Changing the configured pipeline updates profile-preparation model labels');
    await page.locator('#pipelinePreset').selectOption('legacy');
    await page.evaluate(async () => {
        const { ApiClient } = await import('/static/js/core/api-client.js');
        const { ProfilePrepManager: manager } = await import('/static/js/translation/profile-prep-manager.js?v=20261114-openai-primary');
        document.getElementById('llmProvider').value = 'openai';
        document.getElementById('openaiEndpoint').value = 'https://api.openai.com/v1/chat/completions';
        ApiClient.startBookProfilePreparation = form => {
            window.profileFixturePayload = Object.fromEntries(['provider', 'model', 'review_model', 'api_endpoint'].map(key => [key, form.get(key)]));
            return new Promise(() => {});
        };
        manager.pendingFiles = [new File(['A test paragraph.'], 'profile-fixture.txt', { type: 'text/plain' })];
        void manager.start();
    });
    await page.waitForFunction(() => !!window.profileFixturePayload);
    assert.deepEqual(await page.evaluate(() => window.profileFixturePayload), {
        provider: 'openai', model: 'gpt-6-luna', review_model: 'gpt-6.1-terra',
        api_endpoint: 'https://api.openai.com/v1/chat/completions',
    });
    for (const initial of [true, false]) {
        if (!initial) await page.evaluate(async () => {
            const { ProfilePrepManager: manager } = await import('/static/js/translation/profile-prep-manager.js?v=20261114-openai-primary');
            manager.updateFromJob({ status: 'running', model: 'gpt-6-luna', progress: 25 });
        });
        await page.locator('#uiLocaleSelect').selectOption('en');
        await page.waitForFunction(() => document.getElementById('profilePrepStatus').textContent.includes('reading'));
        const english = await page.locator('#profilePrepStatus').textContent();
        await page.locator('#uiLocaleSelect').selectOption('es');
        await page.waitForFunction(() => document.getElementById('profilePrepStatus').textContent.includes('leyendo'));
        assert.notEqual(await page.locator('#profilePrepStatus').textContent(), english);
    }
    report.checks.push('Initial and running profile status translates immediately without polling; OpenAI request fields match the labels');
    await page.evaluate(async () => {
        const { ProfilePrepManager: manager } = await import('/static/js/translation/profile-prep-manager.js?v=20261114-openai-primary');
        manager.setProgress(null, 'Fixture connection warning', 'error');
    });
    await page.locator('#uiLocaleSelect').selectOption('en');
    await page.waitForFunction(() => document.documentElement.lang === 'en');
    assert.equal(await page.locator('#profilePrepStatus').textContent(), 'Fixture connection warning');
    report.checks.push('Locale switch never replaces a newer connection warning with stale progress');
    await page.screenshot({ path: `${output}/profile-model-settings.png` });
    assert.deepEqual(report.errors, []);
} finally {
    await fs.writeFile(`${output}/report.json`, JSON.stringify(report, null, 2));
    await browser.close();
}
console.log(JSON.stringify(report, null, 2));
