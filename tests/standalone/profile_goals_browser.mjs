// Run with pipeline_settings_server.py. Requests are observed before provider work.
import assert from 'node:assert/strict';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.PIPELINE_QA_URL || 'http://127.0.0.1:5099';
assert.equal(new URL(base).hostname, '127.0.0.1');
assert.equal((await fetch(base + '/api/health')).headers.get('X-Verbaloom-QA'), 'isolated');
const browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME_EXECUTABLE ? { executablePath: process.env.CHROME_EXECUTABLE } : {}),
});
const cases = [
    ['translation', 'faithful_translation', 'Traducción fiel'],
    ['audiobook', 'audiobook', 'Audiolibro'],
    ['modernize', 'modernization', 'Modernizar'],
    ['simplify', 'explanatory_rewrite', 'Explicar'],
    ['literary_polish', 'literary_polish', 'Pulido literario'],
];
const file = { name: 'Shared title.txt', mimeType: 'text/plain', buffer: Buffer.from('An original test paragraph.') };
const profiles = [];
const requests = [];
const errors = [];
async function goalHints(locale) {
    const strings = await (await fetch(base + '/static/locales/' + locale + '/transform.json')).json();
    const hints = cases.map(([, goal]) => strings['profile_goal_hint_' + goal]);
    assert.equal(new Set(hints).size, 5);
    assert.ok(hints.every(hint => typeof hint === 'string' && hint.length > 10));
    return Object.fromEntries(cases.map(([, goal], index) => [goal, hints[index]]));
}
try {
    const page = await browser.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('dialog', dialog => dialog.accept());
    await page.route('**/api/book-profiles', route => route.fulfill({ json: { profiles } }));
    await page.route('**/api/book-profiles/prepare-jobs', async route => {
        const req = route.request();
        const form = await new Response(req.postDataBuffer(), { headers: req.headers() }).formData();
        const goal = form.get('profile_goal');
        requests.push([form.get('transform_mode'), goal]);
        const profile = {
            profile_id: 'fixture_' + goal, name: 'Shared title', profile_goal: goal,
            generated_profile: true, target_locale: form.get('target_locale'),
        };
        profiles.push(profile);
        await route.fulfill({ json: {
            prep_id: profile.profile_id, status: 'completed', progress: 100,
            profile_goal: goal, result: { profile: { ...profile, profile_name: profile.name } },
        } });
    });
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.locator('#uiLocaleSelect').selectOption('es');
    await page.waitForFunction(() => document.documentElement.lang === 'es');
    await page.locator('#topTabNav [data-tab="profile-prep"]').click();
    const chooser = page.getByLabel('Uso principal', { exact: true });
    assert.equal(await chooser.inputValue(), 'translation');
    assert.deepEqual(await chooser.locator('option').allTextContents(), cases.map(row => row[2]));
    const transformMode = await page.locator('#transformMode').inputValue();
    const spanishHints = await goalHints('es');
    const help = page.locator('#profilePrepPurposeHelp');
    assert.equal(await chooser.getAttribute('aria-describedby'), 'profilePrepPurposeHelp');
    assert.equal(await help.textContent(), spanishHints.faithful_translation);

    for (const [mode, goal, label] of cases) {
        await chooser.selectOption(mode);
        assert.equal(await help.textContent(), spanishHints[goal]);
        await page.locator('#profilePrepFileInput').setInputFiles(file);
        await page.locator('#profilePrepStartBtn').click();
        const id = 'fixture_' + goal;
        await page.waitForFunction(id => ['transformProfile', 'bookProfileSelect'].every(
            selector => document.getElementById(selector).value === id
        ) && document.getElementById('translateGlossarySelect').value === 'profile:' + id, id);
        for (const selector of ['#bookProfileSelect', '#transformProfile', '#translateGlossarySelect']) {
            assert.ok((await page.locator(selector + ' option:checked').textContent()).includes(label));
        }
        assert.equal(await page.locator('#transformMode').inputValue(), transformMode);
        assert.deepEqual(requests.at(-1), [mode, goal]);
        await page.reload({ waitUntil: 'networkidle' });
        await page.locator('#topTabNav [data-tab="profile-prep"]').click();
        assert.equal(await chooser.inputValue(), mode);
        assert.equal(await help.textContent(), spanishHints[goal]);
        await page.waitForFunction(id => document.getElementById('bookProfileSelect').value === id, id);
    }
    const last = 'fixture_literary_polish';
    for (const locale of ['fr', 'de', 'ja', 'ko', 'zh-CN', 'en']) {
        await page.locator('#uiLocaleSelect').selectOption(locale);
        await page.waitForFunction(locale => {
            if (document.documentElement.lang !== locale) return false;
            return Array.from(document.getElementById('profilePrepPurpose').options).every(option => {
                const id = 'fixture_' + option.dataset.profileGoal;
                return ['transformProfile', 'sampleProfileSelect', 'translateGlossarySelect'].every(selector => {
                    const value = selector === 'translateGlossarySelect' ? 'profile:' + id : id;
                    const saved = Array.from(document.getElementById(selector).options).find(item => item.value === value);
                    return saved?.textContent.includes(option.textContent);
                });
            });
        }, locale);
        const hints = await goalHints(locale);
        const purpose = page.locator('#profilePrepPurpose');
        assert.equal(await help.textContent(), hints.literary_polish);
        for (const [mode, goal] of cases) {
            await purpose.selectOption(mode);
            assert.equal(await help.textContent(), hints[goal]);
        }
    }
    assert.equal(await page.getByLabel('Primary use', { exact: true }).inputValue(), 'literary_polish');
    assert.equal(await page.locator('#bookProfileSelect').inputValue(), last);

    // Simulate an upload suggestion that disagrees with the profile selected for this job.
    await page.route('**/api/upload', async route => {
        const res = await route.fetch();
        const data = await res.json();
        assert.equal(res.ok(), true);
        await route.fulfill({ json: { ...data, profile_id: 'fixture_audiobook', profile_name: 'Other profile' } });
    });
    await page.locator('#topTabNav [data-tab="transform"]').click();
    await page.locator('#transformMode').selectOption('modernize');
    await page.locator('#transformProfile').selectOption(last);
    await page.locator('#transformFileInput').setInputFiles(file);
    const submitted = page.waitForRequest(req => req.url().endsWith('/api/translate') && req.method() === 'POST',
        { timeout: 12000 });
    await page.route('**/api/translate', route => route.fulfill({ status: 400, json: { error: 'End of isolated capture' } }));
    await page.locator('#transformStartBtn').click();
    const request = (await submitted).postDataJSON();
    assert.equal(request.prompt_options.profile_id, last);
    assert.equal(request.prompt_options.text_transform_mode, 'modernize');
    // A normal translation uses the chosen book profile and keeps its own operation.
    await page.locator('#fileListContainer .file-remove-btn').click();
    await page.unroute('**/api/upload');
    await page.locator('#topTabNav [data-tab="translate"]').click();
    await page.locator('#translateGlossarySelect').selectOption('profile:fixture_explanatory_rewrite');
    await page.locator('#sourceLang').selectOption('English');
    await page.locator('#fileInput').setInputFiles({ ...file, name: 'translation fixture.txt' });
    const translation = page.waitForRequest(req => req.url().endsWith('/api/translate') && req.method() === 'POST',
        { timeout: 12000 });
    await page.locator('#translateBtn').click();
    const translationRequest = (await translation).postDataJSON();
    assert.equal(translationRequest.prompt_options.profile_id, 'fixture_explanatory_rewrite');
    assert.ok(!translationRequest.prompt_options.text_transform_mode);
    assert.equal(requests.length, 5);
    assert.deepEqual(errors, []);
    console.log('Verified five preparation choices, contextual help in seven languages and canonically scoped requests, selected profile after each refresh, language changes, restored preference, normal translation and independently chosen transform.');
} finally {
    await browser.close();
}
