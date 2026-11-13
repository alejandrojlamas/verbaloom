import { ApiClient } from '../core/api-client.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { t, applyToDOM } from '../i18n/i18n.js';

const STORAGE = 'verbaloom.pipeline.v1';
const $ = id => document.getElementById(id);
const copy = value => JSON.parse(JSON.stringify(value));
const endpoints = { deepseek: 'https://api.deepseek.com/chat/completions', openai: 'https://api.openai.com/v1/chat/completions' };
let pipeline = null;
let catalog = null;
let revision = 0;
let pending = false;

function status(key) {
    const node = $('pipelineStatus');
    node.dataset.i18n = `settings:pipe_${key}`;
    node.textContent = t(node.dataset.i18n);
}

function accessStatus(result) {
    if (result.status === 'models_visible') return 'visible';
    const provider = (result.providers || []).find(item => item.status !== 'models_visible');
    const failure = provider?.models?.find(item => item.status !== 'model_visible')?.status
        || provider?.status || result.status;
    return {
        missing_credentials: 'missing_credentials', authentication_error: 'access_auth',
        permission_denied: 'access_denied', model_unavailable: 'access_model',
        rate_limited: 'access_rate', connection_error: 'access_connection',
    }[failure] || 'not_ready';
}

function notify() {
    window.dispatchEvent(new CustomEvent('pipelineChanged'));
}

function refreshProviderSurface() {
    const baseline = $('llmProvider')?.value;
    const providers = new Set([baseline, ...Object.values(pipeline?.roles || {}).map(role => role.provider)]);
    for (const [provider, id] of [['openai', 'openaiApiKeyGroup'], ['deepseek', 'deepseekSettings']]) {
        if ($(id)) $(id).style.display = providers.has(provider) ? 'block' : 'none';
    }
    for (const [id, key] of [['providerSelectionLabel', pipeline ? 'pipe_baseline_provider' : 'ai_provider'],
                             ['modelSelectionLabel', pipeline ? 'pipe_baseline_model' : 'model']]) {
        if (!$(id)) continue;
        $(id).dataset.i18n = `settings:${key}`;
        $(id).textContent = t($(id).dataset.i18n);
    }
}

function persist() {
    try {
        if (pipeline) localStorage.setItem(STORAGE, JSON.stringify(pipeline));
        else localStorage.removeItem(STORAGE);
        status(pipeline ? 'saved' : 'unchanged');
    } catch { status('storage_error'); }
    notify();
}

function field(parent, name, key, values, selected) {
    const label = document.createElement('label');
    const text = document.createElement('span');
    text.dataset.i18n = `settings:pipe_${key}`;
    text.textContent = t(text.dataset.i18n);
    label.append(text);
    const input = document.createElement(values ? 'select' : 'input');
    input.className = 'form-control';
    input.dataset.field = key;
    input.id = `pipeline-${name}-${key}`;
    label.htmlFor = input.id;
    if (values) {
        for (const value of values) {
            const option = new Option(value, value);
            if (key === 'effort') {
                option.dataset.i18n = `settings:pipe_${value}`;
                option.textContent = t(option.dataset.i18n);
            } else if (key === 'provider') option.textContent = value === 'openai' ? 'OpenAI' : 'DeepSeek';
            input.add(option);
        }
    } else {
        input.type = 'number'; input.min = '256'; input.step = '1';
        input.max = pipeline.roles[name].provider === 'openai' ? '128000' : '384000';
        input.required = true;
    }
    input.value = String(selected);
    label.append(input);
    parent.append(label);
}

function render() {
    refreshProviderSurface();
    $('pipelineEditor').hidden = !pipeline;
    $('pipelineRoleFields').replaceChildren();
    if (!pipeline || !catalog) return;
    $('pipelineReview').value = pipeline.review_policy;
    $('pipelineBudgetEnabled').checked = !!pipeline.budget;
    $('pipelineBudgetFields').hidden = !pipeline.budget;
    if (pipeline.budget) {
        $('pipelineBudgetUsd').value = pipeline.budget.limit_usd;
        $('pipelineMaxCalls').value = pipeline.budget.max_calls;
    }
    for (const name of catalog.roles) {
        const role = pipeline.roles[name];
        const row = document.createElement('fieldset');
        row.className = 'pipeline-role'; row.dataset.role = name;
        const legend = document.createElement('legend');
        legend.dataset.i18n = `settings:pipe_role_${name}`;
        legend.textContent = t(legend.dataset.i18n);
        row.append(legend);
        const fields = document.createElement('div'); fields.className = 'pipeline-fields';
        field(fields, name, 'provider', Object.keys(catalog.models), role.provider);
        field(fields, name, 'model', catalog.models[role.provider], role.model);
        field(fields, name, 'effort', role.provider === 'openai' ? ['none', 'low'] : ['none'], role.reasoning_effort);
        field(fields, name, 'output', null, role.max_output_tokens);
        row.append(fields); $('pipelineRoleFields').append(row);
    }
    applyToDOM($('pipelineSettings'));
}

function validInputs(report = true) {
    return [...$('pipelineEditor').querySelectorAll('input[type="number"]')]
        .filter(input => !input.closest('[hidden]')).every(input => report ? input.reportValidity() : input.validity.valid);
}

function changed(event) {
    if (!pipeline || pending) return;
    $('pipelineBudgetFields').hidden = !$('pipelineBudgetEnabled').checked;
    if (!validInputs()) { status('invalid'); notify(); return; }
    const row = event.target.closest('[data-role]');
    if (row) {
        const role = pipeline.roles[row.dataset.role];
        const key = event.target.dataset.field;
        if (key === 'provider') {
            role.provider = event.target.value;
            role.model = catalog.models[role.provider][0];
            role.reasoning_effort = 'none';
            role.max_output_tokens = Math.min(role.max_output_tokens, role.provider === 'openai' ? 128000 : 384000);
        } else if (key === 'effort') role.reasoning_effort = event.target.value;
        else if (key === 'output') role.max_output_tokens = Number(event.target.value);
        else if (key === 'model') role.model = event.target.value;
    }
    pipeline.review_policy = $('pipelineReview').value;
    pipeline.epub_repair_policy = pipeline.review_policy === 'source_audited_v1' ? 'localized_v1' : 'compatible';
    pipeline.budget = $('pipelineBudgetEnabled').checked ? {
        limit_usd: $('pipelineBudgetUsd').value, max_calls: Number($('pipelineMaxCalls').value),
    } : null;
    pipeline.preset = 'custom';
    $('pipelinePreset').value = 'custom';
    revision++; render(); persist();
}

export const PipelineSettings = {
    async initialize() {
        if (!$('pipelineSettings')) return;
        $('pipelinePreset').addEventListener('change', () => this.selectPreset());
        $('pipelineEditor').addEventListener('change', changed);
        $('pipelinePreflight').addEventListener('click', () => this.checkAccess());
        window.addEventListener('providerSettingsChanged', refreshProviderSurface);
        window.addEventListener('defaultConfigLoaded', refreshProviderSurface);
        let saved, invalidSaved = false;
        try { saved = JSON.parse(localStorage.getItem(STORAGE) || 'null'); } catch { invalidSaved = true; }
        const ticket = revision;
        if (saved || invalidSaved) { $('pipelinePreset').value = 'custom'; pending = true; status('loading'); }
        try {
            catalog = await ApiClient.getPipelineCatalog();
            if (invalidSaved) throw new Error('Invalid saved pipeline');
            if (saved) {
                const result = await ApiClient.configurePipeline({ pipeline: saved });
                if (ticket !== revision) return;
                pipeline = result.pipeline;
                $('pipelinePreset').value = pipeline.preset;
                render(); status('saved'); notify();
            }
        } catch { if (ticket === revision) status(saved || invalidSaved ? 'invalid' : 'unavailable'); }
        finally { if (ticket === revision) { pending = false; notify(); } }
    },

    async selectPreset() {
        const selected = $('pipelinePreset').value;
        const ticket = ++revision;
        pipeline = null; render();
        if (selected === 'legacy') { pending = false; persist(); return; }
        pending = true; status('loading'); notify();
        try {
            catalog ||= await ApiClient.getPipelineCatalog();
            const result = await ApiClient.configurePipeline({
                preset: selected,
                baseline: { llm_provider: $('llmProvider').value, model: $('model').value },
            });
            if (ticket !== revision) return;
            pipeline = result.pipeline; render(); persist();
        } catch { if (ticket === revision) status('invalid'); }
        finally { if (ticket === revision) { pending = false; notify(); } }
    },

    isEnabled() {
        return !!$('pipelinePreset') && $('pipelinePreset').value !== 'legacy';
    },

    snapshot({ reportValidity = true } = {}) {
        if (!this.isEnabled()) return null;
        if (pending || !pipeline || !validInputs(reportValidity)) throw new Error(t('settings:pipe_invalid'));
        return copy(pipeline);
    },

    primary(role = 'translate') {
        return pipeline?.roles?.[role] || null;
    },

    endpoint(role = 'translate') {
        return endpoints[this.primary(role)?.provider] || null;
    },

    validateCredentials(roles = null) {
        const snapshot = this.snapshot();
        if (!snapshot) return { valid: true };
        const selected = roles || Object.keys(snapshot.roles);
        for (const provider of new Set(selected.map(role => snapshot.roles[role].provider))) {
            const result = ApiKeyUtils.validateForProvider(provider, endpoints[provider]);
            if (!result.valid) return result;
        }
        return { valid: true };
    },

    apply(config) {
        const snapshot = this.snapshot();
        if (!snapshot) return config;
        const primary = snapshot.roles.translate;
        return { ...config, pipeline: snapshot, llm_provider: primary.provider, model: primary.model,
            llm_api_endpoint: endpoints[primary.provider], ...this.credentials(snapshot) };
    },

    credentials(snapshot, roles = Object.keys(snapshot.roles)) {
        return Object.fromEntries([...new Set(roles.map(role => snapshot.roles[role].provider))]
            .map(provider => [`${provider}_api_key`, ApiKeyUtils.getValueForProvider(provider)]));
    },

    async prepare() {
        const snapshot = this.snapshot();
        if (!snapshot) return null;
        const result = await ApiClient.configurePipeline({ pipeline: snapshot });
        return result.pipeline;
    },

    async verifyAccess(config, roles = null) {
        if (!config.pipeline) return null;
        const selected = roles || Object.keys(config.pipeline.roles);
        const request = { pipeline: config.pipeline, roles: selected };
        for (const provider of new Set(selected.map(role => config.pipeline.roles[role].provider))) {
            const field = `${provider}_api_key`;
            if (config[field]) request[field] = config[field];
        }
        const result = await ApiClient.preflightPipeline(request);
        const key = accessStatus(result);
        if (key !== 'visible') {
            const error = new Error(t(`settings:pipe_${key}`));
            error.pipelineStatus = key;
            throw error;
        }
        return result;
    },

    async checkAccess() {
        const button = $('pipelinePreflight');
        const ticket = revision;
        button.disabled = true; status('checking');
        try {
            const snapshot = await this.prepare();
            await this.verifyAccess({ pipeline: snapshot, ...this.credentials(snapshot) });
            if (ticket === revision) status('visible');
        } catch (error) { if (ticket === revision) status(error.pipelineStatus || 'not_ready'); }
        finally { button.disabled = false; }
    },
};
