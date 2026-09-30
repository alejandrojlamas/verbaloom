/**
 * Sample & Compare manager — orchestrates UI for the Sample tab.
 *
 * Owns:
 *  - upload state for the selected book file
 *  - a fixed set of editorial pipeline variants over the same model
 *  - kick-off + stop of a sample run via the backend
 *  - subscription to the `sample_update` WebSocket event for streaming cells
 *  - a cross-Run results cache so identical (item, llm, params) cells are
 *    never re-translated; changing a variant updates the displayed grid
 *    immediately and only the new column hits the backend on the next Run.
 *
 * Delegates rendering of the comparison grid to SampleTable. Inline diff
 * (translate+refine) lives in sample-diff.js.
 */

import { ApiClient } from '../core/api-client.js';
import { WebSocketManager } from '../core/websocket-manager.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { t, applyToDOM } from '../i18n/i18n.js';
import { SampleTable } from './sample-table.js';
import { SAMPLE_DEFAULT_N_SAMPLES, SAMPLE_DEFAULT_MAX_CHARS } from './sample-defaults.js';
import { SearchableSelectFactory } from '../ui/searchable-select.js';
import {
    PROVIDER_ORDER,
    PROVIDER_META,
    attachProviderSearchable,
    attachModelSearchable,
    populateModelSelectInto,
    setPlaceholderOption,
} from '../providers/provider-select-helpers.js';

// Track every SearchableSelect attached by this module so we can avoid double
// wiring when the tab is initialized more than once in a long browser session.
const sampleSearchableIds = new Set();
const SAMPLE_SESSION_STORAGE_KEY = 'verbaloom.sample.activeRun.v2';
const LEGACY_SAMPLE_SESSION_STORAGE_KEY = 'tbl.sample.activeRun.v2';
const SAMPLE_POLL_INTERVAL_MS = 2500;
const SAMPLE_CONTROL_REQUEST_TIMEOUT_MS = 20000;
let samplePollTimer = null;

const state = {
    file: null,          // File object
    uploadedPath: null,  // path on server after upload
    fileType: null,
    columns: [],         // generated pipeline variants sent to /api/sample/run
    modelConfig: {
        provider: 'deepseek',
        model: '',
        api_endpoint: '',
        custom_instruction_file: '',
    },
    mode: 'translate',
    currentSampleId: null,
    running: false,
    // Snapshot of the last Run — used by add/remove column to re-render the
    // result table without needing a fresh server roundtrip.
    lastItems: null,            // [{index, source_text, truncated}] or null before first Run
    lastRunContext: null,       // { mode, source_lang, target_lang, prompt_options, glossary_id }
    // Maps "row:col" -> cellKey for the *currently displayed* table, so that
    // streamed cell_done events know which cache key to update.
    currentRunKeys: new Map(),
    activeRunCells: new Set(),
    lastRunStatus: 'idle',
    lastProgressAt: 0,
    lastProgressCount: 0,
    snapshotViewLocked: false,
    initializingSamples: false,
    pendingRunAfterInitialize: false,
    // Last (N, max_chars) actually fed to /api/sample/initialize; the "Update
    // samples" button is enabled only when the current input values differ
    // from these.
    appliedNSamples: null,
    appliedMaxChars: null,
};

// Cross-Run result cache. Persists across Runs for the lifetime of the page.
// Key: canonical JSON of (source_text, model config, mode, langs, prompt/profile/glossary variant)
// Value: { translate?: {output, metrics}, refine?: {output, metrics} }
const resultsCache = new Map();

// Providers that take a configurable API endpoint in the column UI. Both
// default to the endpoint set in Settings (fetched once via /api/config).
const ENDPOINT_PROVIDERS = new Set(['openai', 'ollama']);
let settingsEndpoints = { ollama: '', openai: '' };

// Available custom-instruction presets (files in Custom_Instructions/), shared
// by every column's per-LLM picker. [{ filename, display_name }]
let customInstructionFiles = [];

// Available glossaries, shared by every column's per-LLM picker. [{ id, name }]
let glossaryList = [];
let profileList = [];

function $(id) {
    return document.getElementById(id);
}

function safeLocalStorage() {
    try {
        return window.localStorage;
    } catch (_err) {
        return null;
    }
}

function persistSampleSession() {
    const storage = safeLocalStorage();
    if (!storage || !state.currentSampleId) return;
    const payload = {
        sample_id: state.currentSampleId,
        uploadedPath: state.uploadedPath,
        fileType: state.fileType,
        thumbnail: state.thumbnail || null,
        file: state.file ? {
            name: state.file.name || 'file',
            size: state.file.size || 0,
        } : null,
        savedAt: Date.now(),
    };
    try {
        storage.setItem(SAMPLE_SESSION_STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
        console.warn('[sample] could not persist sample session', err);
    }
}

function readPersistedSampleSession() {
    const storage = safeLocalStorage();
    if (!storage) return null;
    try {
        let raw = storage.getItem(SAMPLE_SESSION_STORAGE_KEY);
        if (!raw) {
            raw = storage.getItem(LEGACY_SAMPLE_SESSION_STORAGE_KEY);
            if (raw) {
                storage.setItem(SAMPLE_SESSION_STORAGE_KEY, raw);
                storage.removeItem(LEGACY_SAMPLE_SESSION_STORAGE_KEY);
            }
        }
        return raw ? JSON.parse(raw) : null;
    } catch (err) {
        console.warn('[sample] could not read persisted sample session', err);
        return null;
    }
}

function clearPersistedSampleSession() {
    const storage = safeLocalStorage();
    if (!storage) return;
    try {
        storage.removeItem(SAMPLE_SESSION_STORAGE_KEY);
        storage.removeItem(LEGACY_SAMPLE_SESSION_STORAGE_KEY);
    } catch (_err) {
        // ignore storage failures
    }
}

/**
 * The endpoint a column should send (and use to list models). Only meaningful
 * for ENDPOINT_PROVIDERS; returns undefined otherwise so stale values from a
 * previous provider are never forwarded.
 */
function columnEndpoint(col) {
    return ENDPOINT_PROVIDERS.has(col.provider) ? (col.api_endpoint || undefined) : undefined;
}

/** Greyed-out hint shown when the endpoint field is empty, per provider. */
function endpointPlaceholder(provider) {
    if (provider === 'ollama') return settingsEndpoints.ollama || 'http://localhost:11434/api/generate';
    return settingsEndpoints.openai || 'https://api.openai.com/v1/chat/completions';
}

/**
 * Fetch the Settings endpoints once and seed any column still lacking one, so
 * new columns default to the same endpoint the user configured in Settings.
 */
async function loadSettingsEndpoints() {
    try {
        const cfg = await ApiClient.getConfig();
        settingsEndpoints = {
            ollama: cfg.ollama_api_endpoint || cfg.api_endpoint || '',
            openai: cfg.openai_api_endpoint || '',
        };
        state.modelConfig.provider = (cfg.llm_provider || state.modelConfig.provider || 'deepseek').trim().toLowerCase();
        state.modelConfig.model = (cfg.default_model || state.modelConfig.model || '').trim();
        if (!state.modelConfig.api_endpoint && settingsEndpoints[state.modelConfig.provider]) {
            state.modelConfig.api_endpoint = settingsEndpoints[state.modelConfig.provider];
        }
        syncModelControlsFromState();
        await loadAndPopulateBaseModel();
        renderColumns();
    } catch (err) {
        console.warn('[sample] could not load default endpoints from /api/config', err);
    }
}

/**
 * Fetch the custom-instruction presets once and re-render columns so each LLM's
 * picker is populated. Same source as the Translate tab's global picker.
 */
async function loadCustomInstructionFiles() {
    try {
        const data = await ApiClient.getCustomInstructions();
        customInstructionFiles = Array.isArray(data.files) ? data.files : [];
        populateInstructionSelect();
        renderColumns();
    } catch (err) {
        console.warn('[sample] could not load custom instruction presets', err);
    }
}

/**
 * Fetch the glossaries once and re-render columns so each LLM's glossary picker
 * is populated.
 */
async function loadGlossaries() {
    try {
        const data = await ApiClient.getGlossaries();
        glossaryList = Array.isArray(data.glossaries) ? data.glossaries : [];
        populateGlossarySelect();
        renderColumns();
    } catch (err) {
        console.warn('[sample] could not load glossaries', err);
    }
}

async function loadBookProfiles() {
    try {
        const data = await ApiClient.getBookProfiles();
        profileList = Array.isArray(data.profiles) ? data.profiles : [];
        populateProfileSelect();
        renderColumns();
    } catch (err) {
        console.warn('[sample] could not load book profiles', err);
    }
}

function providerLabel(value) {
    const meta = PROVIDER_META[value] || {};
    return meta.name || value || '—';
}

function optionHtml(value, label, selected = false) {
    return `<option value="${DomHelpers.escapeHtml(String(value || ''))}" ${selected ? 'selected' : ''}>${DomHelpers.escapeHtml(label || '')}</option>`;
}

function profileOptionLabel(profile) {
    const name = profile?.name || profile?.profile_name || profile?.profile_id || '';
    const approved = Number(profile?.approved_count ?? profile?.approved_entries ?? 0);
    const pending = Number(profile?.pending_count ?? profile?.pending_suggestions ?? 0);
    if (approved || pending) {
        return `${name} · ${approved} ${t('sample:profile_approved_short')} · ${pending} ${t('sample:profile_pending_short')}`;
    }
    return name;
}

function populateInstructionSelect() {
    const select = $('sampleInstructions');
    if (!select) return;
    const current = state.modelConfig.custom_instruction_file || select.value || '';
    select.innerHTML = [optionHtml('', t('settings:select_none'), !current)]
        .concat(customInstructionFiles.map((f) => optionHtml(
            f.filename,
            f.display_name || f.filename,
            current === f.filename,
        )))
        .join('');
    if (current && Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
    }
}

function populateGlossarySelect() {
    const select = $('sampleGlossarySelect');
    if (!select) return;
    const current = select.value || '';
    select.innerHTML = [optionHtml('', t('sample:glossary_none'), !current)]
        .concat(glossaryList.map((g) => optionHtml(
            String(g.id),
            `${g.name || `#${g.id}`} · ${Number(g.term_count || 0)} ${t('sample:terms_short')}`,
            String(current) === String(g.id),
        )))
        .join('');
    if (current && Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
    }
}

function populateProfileSelect() {
    const select = $('sampleProfileSelect');
    if (!select) return;
    const current = select.value || '';
    select.innerHTML = [optionHtml('', t('sample:profile_none'), !current)]
        .concat(profileList.map((profile) => optionHtml(
            profile.profile_id,
            profileOptionLabel(profile),
            current === profile.profile_id,
        )))
        .join('');
    if (current && Array.from(select.options).some((option) => option.value === current)) {
        select.value = current;
    }
}

function populateProviderSelect() {
    const select = $('sampleProvider');
    if (!select) return;
    const provider = state.modelConfig.provider || 'deepseek';
    select.innerHTML = PROVIDER_ORDER
        .map((value) => optionHtml(value, providerLabel(value), value === provider))
        .join('');
    if (Array.from(select.options).some((option) => option.value === provider)) {
        select.value = provider;
    }
}

function syncModelControlsFromState() {
    populateProviderSelect();
    populateInstructionSelect();
    const providerInst = SearchableSelectFactory.get('sampleProvider');
    const providerValue = state.modelConfig.provider || 'deepseek';
    if (providerInst && providerInst.getValue() !== providerValue) {
        providerInst.setValue(providerValue);
    }
    const endpointWrap = $('sampleEndpointWrap');
    const endpointInput = $('sampleEndpoint');
    if (endpointWrap) {
        endpointWrap.classList.toggle('hidden', !ENDPOINT_PROVIDERS.has(state.modelConfig.provider));
    }
    if (endpointInput) {
        endpointInput.value = state.modelConfig.api_endpoint || '';
        endpointInput.placeholder = endpointPlaceholder(state.modelConfig.provider);
    }
}

async function loadAndPopulateBaseModel() {
    const modelSelectEl = $('sampleModel');
    if (!modelSelectEl) return '';
    const picked = await loadAndPopulateModelsForColumn(state.modelConfig, modelSelectEl);
    const modelInst = SearchableSelectFactory.get('sampleModel');
    if (picked && modelInst && modelInst.getValue() !== picked) {
        modelInst.setValue(picked);
    }
    return picked;
}

function attachBaseModelSearchable() {
    const modelSelectEl = $('sampleModel');
    if (!modelSelectEl || sampleSearchableIds.has('sampleModel')) return;
    attachModelSearchable(modelSelectEl, {
        onChange: (value) => {
            state.snapshotViewLocked = false;
            state.modelConfig.model = value;
            renderColumns();
            refreshResultsFromCache();
        },
    });
    sampleSearchableIds.add('sampleModel');
}

function transformLabelFor(value) {
    const select = $('sampleTransformMode');
    const option = select ? Array.from(select.options).find((item) => item.value === value) : null;
    return option?.textContent?.trim() || value.replace(/_/g, ' ');
}

function transformDescriptionFor(value) {
    const map = {
        modernize: t('transform:mode_modernize_desc'),
        simplify: t('transform:mode_simplify_desc'),
        humanize: t('transform:mode_humanize_desc'),
        mexican_spanish: t('transform:mode_mexican_desc'),
        academic_clarity: t('transform:mode_academic_desc'),
        literary_polish: t('transform:mode_literary_desc'),
        ocr_structure: t('transform:mode_ocr_desc'),
    };
    return map[value] || '';
}

function selectedProfileStrength() {
    const value = ($('profileStrengthSelect')?.value || 'balanced').trim().toLowerCase();
    return ['light', 'balanced', 'strict'].includes(value) ? value : 'balanced';
}

function applyProfilePromptOptions(options, profileId, targetLanguage, transformMode = '') {
    const selectedProfile = (profileId || '').trim();
    if (!selectedProfile) return;

    options.editorial_mode = 'book_profile';
    options.profile_id = selectedProfile;
    options.profile_strength = selectedProfileStrength();
    if (/^spanish$/i.test(targetLanguage || '')) {
        options.target_locale = 'es-MX';
    }
    options.preserve_author_voice = true;
    options.use_profile_glossary = true;
    options.allow_common_glossary = true;
    options.allow_cross_profile_glossary = false;
    options.glossary_suggestions_enabled = true;
    options.auto_approve_glossary_suggestions = false;
    options.min_glossary_suggestion_confidence = 0.92;
    options.avoid_hardcoded_editorial_rules = true;

    if (transformMode) {
        options.modernization_strength = transformMode === 'modernize' ? 'high' : 'medium';
    } else {
        options.audit_dimensions = 'translation_editorial_full';
        options.translation_profile_mode = true;
        options.profile_audit_enabled = false;
        options.repair_until_pass = true;
        options.max_repair_rounds = 1;
    }
}

function basePromptOptions() {
    return {
        preserve_technical_content: $('preserveTechnicalContent')?.checked || false,
        text_cleanup: $('textCleanup')?.checked || false,
    };
}

function promptOptionsForVariant({ profileId = '', transformMode = '' } = {}) {
    const targetLanguage = $('sampleTargetLang')?.value || '';
    const options = basePromptOptions();
    applyProfilePromptOptions(options, profileId, targetLanguage, transformMode);

    if (!transformMode) return options;

    options.text_transform_mode = transformMode;
    options.text_transform_label = transformLabelFor(transformMode);
    options.refinement_instructions = transformDescriptionFor(transformMode);

    if (transformMode === 'modernize') {
        options.text_transform_profile = 'faithful_current_spanish';
        options.preserve_block_structure = true;
        options.transform_guard = 'strict';
        options.transform_auditor_model = profileId ? 'deepseek-v4-pro' : 'deepseek-flash';
        options.transform_repair_attempts = 2;
        options.suppress_attribution_footer = true;
        options.editorial_quality_guard = true;
        options.fidelity_supervisor = true;
        options.fidelity_supervisor_mode = profileId ? 'always' : 'alerted';
        options.fidelity_supervisor_model = profileId ? 'deepseek-v4-pro' : 'deepseek-flash';
        options.transform_fallback = 'best_candidate';
        if (profileId) {
            options.preserve_archaisms = false;
            options.preserve_iconic_formulas = true;
            options.audit_dimensions = 'editorial_full';
            options.min_dimension_score = 8.5;
            options.max_repair_rounds = 2;
            if (options.profile_strength === 'strict') {
                options.profile_audit_enabled = true;
                options.profile_audit_model = 'deepseek-v4-pro';
                options.profile_repair_model = 'deepseek-v4-pro';
                options.transform_auditor_model = 'deepseek-v4-pro';
                options.fidelity_supervisor_mode = 'always';
                options.fidelity_supervisor_model = 'deepseek-v4-pro';
            }
        }
    } else {
        options.editorial_quality_guard = false;
        options.source_aware_editorial_guard = false;
        options.fidelity_supervisor_mode = 'local';
    }
    return options;
}

/**
 * Stable JSON for prompt_options — used as part of the cache key. Only the
 * fields that actually affect the prompt are included.
 */
function normalizePromptOptions(po) {
    return JSON.stringify({
        ci: (po && po.custom_instructions) || '',
        ptc: !!(po && po.preserve_technical_content),
        tc: !!(po && po.text_cleanup),
        profile_id: (po && po.profile_id) || '',
        editorial_mode: (po && po.editorial_mode) || '',
        target_locale: (po && po.target_locale) || '',
        transform_mode: (po && po.text_transform_mode) || '',
        transform_label: (po && po.text_transform_label) || '',
        refinement_instructions: (po && po.refinement_instructions) || '',
        modernization_strength: (po && po.modernization_strength) || '',
    });
}

/**
 * Build a canonical cache key for a (source extract, LLM config, run context)
 * tuple. Two cells share a key iff their LLM outputs are expected to match.
 */
function buildCellKey(item, col, runCtx) {
    return JSON.stringify({
        src: item.source_text,
        mode: runCtx.mode,
        sl: runCtx.source_lang || '',
        tl: runCtx.target_lang || '',
        provider: col.provider || '',
        model: col.model || '',
        ep: columnEndpoint(col) || '',
        cif: col.custom_instruction_file || '',
        po: normalizePromptOptions(runCtx.prompt_options),
        cpo: normalizePromptOptions(col.prompt_options),
        gl: col.glossary_id || '',
        profile: col.profile_id || '',
        variant: col.variant_key || '',
        refine: !!col.refine_after_translate,
    });
}

/**
 * Which phases must have a 'done' entry for a cached cell to count as a hit?
 */
function requiredPhases(mode, col = null) {
    if (col && col.refine_after_translate) return ['translate', 'refine'];
    if (mode === 'translate_refine' && col && col.refine_after_translate === false) return ['translate'];
    if (mode === 'refine') return ['refine'];
    if (mode === 'translate_refine') return ['translate', 'refine'];
    return ['translate'];
}

function isFullCacheHit(entry, mode, col = null) {
    if (!entry) return false;
    return requiredPhases(mode, col).every((p) => entry[p] && entry[p].status === 'done');
}

function sampleSnapshot() {
    const tableSnapshot = SampleTable.getIntegrationSnapshot
        ? SampleTable.getIntegrationSnapshot()
        : null;
    const hasStateItems = Array.isArray(state.lastItems);
    const items = hasStateItems
        ? ((tableSnapshot && tableSnapshot.items && tableSnapshot.items.length) ? tableSnapshot.items : state.lastItems)
        : [];
    const columns = (tableSnapshot && tableSnapshot.columns && tableSnapshot.columns.length)
        ? tableSnapshot.columns
        : (Array.isArray(state.columns) ? state.columns : []);
    const mode = (tableSnapshot && tableSnapshot.mode)
        || (state.lastRunContext && state.lastRunContext.mode)
        || (columns.some((col) => col.refine_after_translate) ? 'translate_refine' : state.mode);
    const cellState = (tableSnapshot && tableSnapshot.cellState instanceof Map)
        ? tableSnapshot.cellState
        : new Map();
    return { items, columns, mode, cellState };
}

function cellRunState(entry, mode, col, cellKey) {
    const phases = requiredPhases(mode, col);
    if (phases.some((phase) => entry && entry[phase] && entry[phase].status === 'error')) {
        return 'error';
    }
    if (entry && phases.every((phase) => entry[phase] && entry[phase].status === 'done')) {
        return 'done';
    }
    if (state.activeRunCells.has(cellKey)) return 'active';
    if (entry && phases.some((phase) => entry[phase])) return 'active';
    return 'pending';
}

function progressSnapshot() {
    const { items, columns, mode, cellState } = sampleSnapshot();
    const total = items.length * columns.length;
    const progress = { total, done: 0, error: 0, active: 0, pending: 0, percent: 0 };
    if (!total) return progress;

    items.forEach((_item, rowIdx) => {
        columns.forEach((col, colIdx) => {
            const key = `${rowIdx}:${colIdx}`;
            const status = cellRunState(cellState.get(key), mode, col, key);
            progress[status] += 1;
        });
    });
    progress.percent = Math.round(((progress.done + progress.error) / total) * 100);
    return progress;
}

function isTerminalProgress(progress) {
    return !!progress && progress.total > 0 && (progress.done + progress.error) >= progress.total;
}

function reconcileTerminalProgress(progress = null) {
    const current = progress || progressSnapshot();
    if (!state.running || !isTerminalProgress(current)) return false;

    state.running = false;
    state.activeRunCells = new Set();
    state.lastRunStatus = current.error > 0 ? 'partial' : 'done';
    stopSamplePolling();
    syncRunningChrome(false);
    return true;
}

function markProgressHeartbeat(progress = null) {
    const current = progress || progressSnapshot();
    const completed = current.done + current.error;
    if (!state.lastProgressAt || completed !== state.lastProgressCount) {
        state.lastProgressAt = Date.now();
        state.lastProgressCount = completed;
    }
}

function renderRunStatus() {
    const box = $('sampleRunStatus');
    if (!box) return;
    const progress = progressSnapshot();
    const hasVisibleState = state.running || state.currentSampleId || progress.total > 0;
    if (!hasVisibleState) {
        box.classList.add('hidden');
        box.innerHTML = '';
        return;
    }

    let title = t('sample:status_idle');
    let icon = 'radio_button_unchecked';
    let tone = 'idle';
    const effectivelyRunning = state.running && !isTerminalProgress(progress);
    if (effectivelyRunning) {
        title = t('sample:status_running');
        icon = 'sync';
        tone = 'running';
    } else if (state.lastRunStatus === 'stopped') {
        title = t('sample:status_stopped');
        icon = 'pause_circle';
        tone = 'stopped';
    } else if (progress.error > 0) {
        title = t('sample:status_partial');
        icon = 'error';
        tone = 'warning';
    } else if (progress.total > 0 && progress.done >= progress.total) {
        title = t('sample:status_completed');
        icon = 'check_circle';
        tone = 'done';
    } else if (progress.total > 0) {
        title = t('sample:status_ready');
        icon = 'fact_check';
        tone = 'idle';
    }

    const detail = progress.error > 0
        ? t('sample:status_detail_errors', { done: progress.done, total: progress.total, error: progress.error })
        : t('sample:status_detail', { done: progress.done, total: progress.total });
    const waitingSeconds = effectivelyRunning && state.lastProgressAt
        ? Math.max(1, Math.floor((Date.now() - state.lastProgressAt) / 1000))
        : 0;
    const waitingText = waitingSeconds >= 10
        ? t('sample:status_waiting_elapsed', { seconds: waitingSeconds })
        : t('sample:status_waiting');
    const waiting = effectivelyRunning
        ? `<span class="sample-run-status-waiting">· ${DomHelpers.escapeHtml(waitingText)}</span>`
        : '';

    box.className = `sample-run-status sample-run-status-${tone}`;
    box.innerHTML = `
        <div class="sample-run-status-main">
            <span class="material-symbols-outlined sample-run-status-icon">${icon}</span>
            <div class="sample-run-status-copy">
                <strong>${DomHelpers.escapeHtml(title)}</strong>
                <span>${DomHelpers.escapeHtml(detail)}${waiting}</span>
            </div>
            <span class="sample-run-status-percent">${progress.percent}%</span>
        </div>
        <div class="sample-run-status-track" aria-hidden="true">
            <div class="sample-run-status-fill" style="width: ${Math.max(0, Math.min(100, progress.percent))}%"></div>
        </div>
    `;
}

function finalBlockForCell(entry, mode, col) {
    const phases = requiredPhases(mode, col);
    const preferred = phases.includes('refine') ? ['refine', 'translate'] : ['translate', 'refine'];
    for (const phase of preferred) {
        const block = entry && entry[phase];
        if (block && block.status === 'done' && block.output) {
            return { phase, status: 'done', text: block.output };
        }
    }
    for (const phase of phases) {
        const block = entry && entry[phase];
        if (block && block.status === 'error') {
            return { phase, status: 'error', text: block.error || 'Error' };
        }
    }
    return { phase: '', status: 'pending', text: '' };
}

function integratedTextForColumn(colIdx) {
    const { items, columns, mode, cellState } = sampleSnapshot();
    const col = columns[colIdx];
    if (!col) return '';
    return items.map((item, rowIdx) => {
        const result = finalBlockForCell(cellState.get(`${rowIdx}:${colIdx}`), mode, col);
        const label = `#${item.index}`;
        if (result.status === 'done') return `${label}\n${result.text || ''}`.trim();
        if (result.status === 'error') return `${label}\n[${t('sample:integrated_error')}: ${result.text}]`;
        return `${label}\n[${t('sample:integrated_pending')}]`;
    }).join('\n\n');
}

function renderIntegratedResults() {
    const section = $('sampleIntegratedSection');
    const root = $('sampleIntegratedResults');
    if (!section || !root) return;
    const { items, columns, mode, cellState } = sampleSnapshot();
    if (!items.length || !columns.length) {
        section.classList.add('hidden');
        root.innerHTML = '';
        return;
    }

    section.classList.remove('hidden');
    root.innerHTML = columns.map((col, colIdx) => {
        const fragments = items.map((item, rowIdx) => {
            const result = finalBlockForCell(cellState.get(`${rowIdx}:${colIdx}`), mode, col);
            const body = result.status === 'done'
                ? DomHelpers.escapeHtml(result.text)
                : DomHelpers.escapeHtml(result.status === 'error'
                    ? `${t('sample:integrated_error')}: ${result.text}`
                    : t('sample:integrated_pending'));
            return `
                <div class="sample-integrated-fragment sample-integrated-fragment-${result.status}">
                    <div class="sample-integrated-fragment-label">#${DomHelpers.escapeHtml(String(item.index))}</div>
                    <div class="sample-integrated-fragment-text">${body}</div>
                </div>
            `;
        }).join('');
        return `
            <article class="sample-integrated-card">
                <div class="sample-integrated-card-header">
                    <div>
                        <span>${DomHelpers.escapeHtml(t('sample:pipeline_variant'))} #${colIdx + 1}</span>
                        <strong>${DomHelpers.escapeHtml(col.label || `${col.provider || '?'} / ${col.model || '?'}`)}</strong>
                    </div>
                    <button type="button" class="sample-integrated-copy" data-sample-integrated-copy="${colIdx}">
                        <span class="material-symbols-outlined">content_copy</span>
                        <span>${DomHelpers.escapeHtml(t('sample:integrated_copy'))}</span>
                    </button>
                </div>
                ${col.variant_summary ? `<p class="sample-integrated-summary">${DomHelpers.escapeHtml(col.variant_summary)}</p>` : ''}
                <div class="sample-integrated-body">${fragments}</div>
            </article>
        `;
    }).join('');
}

function cellPhasesFromSnapshot(snapshot) {
    const cells = Array.isArray(snapshot?.cells) ? snapshot.cells : [];
    const phaseMap = new Map();
    cells.forEach((cell) => {
        if (!cell || cell.status === 'pending') return;
        const row = Number(cell.row);
        const col = Number(cell.col);
        if (!Number.isInteger(row) || !Number.isInteger(col)) return;
        const phase = cell.phase || 'translate';
        const key = `${row}:${col}`;
        const entry = phaseMap.get(key) || {};
        entry[phase] = {
            status: cell.status === 'error' ? 'error' : 'done',
            output: cell.output || '',
            metrics: cell.metrics || {},
            error: cell.error || '',
        };
        phaseMap.set(key, entry);
    });
    return phaseMap;
}

function incompleteCellsForSnapshot(snapshot, phaseMap) {
    const items = Array.isArray(snapshot?.items) ? snapshot.items : [];
    const columns = Array.isArray(snapshot?.columns) ? snapshot.columns : [];
    const mode = snapshot?.mode || 'translate';
    const active = new Set();
    items.forEach((_item, rowIdx) => {
        columns.forEach((col, colIdx) => {
            const key = `${rowIdx}:${colIdx}`;
            const entry = phaseMap.get(key);
            const failed = requiredPhases(mode, col).some((phase) => entry?.[phase]?.status === 'error');
            if (!failed && !isFullCacheHit(entry, mode, col)) {
                active.add(key);
            }
        });
    });
    return active;
}

function runContextFromSnapshot(snapshot) {
    const runCtx = snapshot?.run_context || {};
    return {
        mode: runCtx.mode || snapshot?.mode || 'translate',
        source_lang: runCtx.source_language || '',
        target_lang: runCtx.target_language || '',
        prompt_options: runCtx.prompt_options || {},
    };
}

function syncRunningChrome(running) {
    const runBtn = $('sampleRunBtn');
    const stopBtn = $('sampleStopBtn');
    const hasSamples = Array.isArray(state.lastItems) && state.lastItems.length > 0;
    if (runBtn) runBtn.disabled = running || state.initializingSamples || !hasSamples;
    if (stopBtn) stopBtn.classList.toggle('hidden', !running);
    const columnsEl = $('sampleColumns');
    if (columnsEl) columnsEl.classList.toggle('is-running', running);
    syncSampleEditButtons();
    syncUpdateButton();
}

function applySampleSnapshot(snapshot, { session = null, announce = false } = {}) {
    if (!snapshot || !snapshot.sample_id) return false;

    state.currentSampleId = snapshot.sample_id;
    state.lastItems = Array.isArray(snapshot.items) ? snapshot.items : [];
    state.columns = Array.isArray(snapshot.columns) ? snapshot.columns : [];
    state.lastRunContext = runContextFromSnapshot(snapshot);
    state.snapshotViewLocked = true;

    if (session) {
        state.uploadedPath = session.uploadedPath || state.uploadedPath;
        state.fileType = session.fileType || state.fileType;
        state.thumbnail = session.thumbnail || state.thumbnail;
        if (session.file && !state.file) {
            state.file = {
                name: session.file.name || 'file',
                size: session.file.size || 0,
            };
            updateFileCard();
            showFileCardMode(true);
        }
    }

    const phaseMap = cellPhasesFromSnapshot(snapshot);
    const status = snapshot.status || 'prepared';
    let running = status === 'running';
    let activeRunCells = running ? incompleteCellsForSnapshot(snapshot, phaseMap) : new Set();
    let inferredComplete = false;
    if (running && activeRunCells.size === 0 && state.lastItems.length > 0 && state.columns.length > 0) {
        running = false;
        inferredComplete = true;
    }
    state.running = running;
    state.lastRunStatus = (status === 'completed' || inferredComplete)
        ? 'done'
        : (status === 'stopped' ? 'stopped' : (running ? 'running' : 'idle'));
    state.activeRunCells = running ? activeRunCells : new Set();

    const results = $('sampleResults');
    if (results) {
        SampleTable.render(results, state.lastItems, state.columns, state.lastRunContext.mode, {
            prefilled: phaseMap,
            runningCells: state.activeRunCells,
        });
        applyToDOM(results);
    }
    syncRunningChrome(running);
    renderColumns({ force: true, useCurrent: true });
    markProgressHeartbeat();
    renderRunStatus();
    renderIntegratedResults();
    const copyBtn = $('sampleCopyMdBtn');
    if (copyBtn) copyBtn.disabled = !SampleTable.hasResults();

    if (announce) {
        showSampleMessage(t('sample:run_restored'), 'info');
    }

    if (running) {
        persistSampleSession();
        startSamplePolling();
    } else {
        stopSamplePolling();
        if (status === 'completed' || status === 'stopped') {
            persistSampleSession();
        }
    }
    return true;
}

async function fetchSampleSnapshot(sampleId) {
    const response = await fetch(`/api/sample/${encodeURIComponent(sampleId)}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(body.error || `HTTP ${response.status}`);
    }
    return body;
}

async function refreshSampleSnapshot({ announceMissing = false } = {}) {
    if (!state.currentSampleId) return false;
    try {
        const snapshot = await fetchSampleSnapshot(state.currentSampleId);
        return applySampleSnapshot(snapshot);
    } catch (err) {
        console.warn('[sample] snapshot refresh failed', err);
        stopSamplePolling();
        state.running = false;
        state.activeRunCells = new Set();
        syncRunningChrome(false);
        renderRunStatus();
        if (announceMissing) {
            showSampleMessage(t('sample:run_restore_failed', { error: err.message || String(err) }), 'error');
        }
        return false;
    }
}

function startSamplePolling() {
    stopSamplePolling();
    if (!state.currentSampleId || !state.running) return;
    window.setTimeout(() => {
        if (state.currentSampleId && state.running) {
            refreshSampleSnapshot();
        }
    }, 250);
    samplePollTimer = window.setInterval(() => {
        refreshSampleSnapshot();
    }, SAMPLE_POLL_INTERVAL_MS);
}

function stopSamplePolling() {
    if (samplePollTimer) {
        window.clearInterval(samplePollTimer);
        samplePollTimer = null;
    }
}

async function restoreSampleSession() {
    const session = readPersistedSampleSession();
    if (!session || !session.sample_id) return;
    try {
        const snapshot = await fetchSampleSnapshot(session.sample_id);
        applySampleSnapshot(snapshot, { session, announce: snapshot.status === 'running' });
    } catch (err) {
        console.warn('[sample] persisted sample run is not restorable', err);
        clearPersistedSampleSession();
    }
}

/**
 * Show a status message inline within the Sample tab.
 *
 * `MessageLogger.showMessage` targets `#messages`, which lives inside the
 * Translate tab — so its toasts are invisible while the user is on Sample.
 * We render here instead so failures don't appear silent.
 */
function showSampleMessage(text, type = 'info') {
    const box = $('sampleWarnings');
    if (!box) return;
    const div = document.createElement('div');
    div.className = `sample-warning sample-warning-${type}`;
    const icon = type === 'error' ? '✖' : (type === 'success' ? '✓' : '⚠');
    div.textContent = `${icon} ${text}`;
    box.innerHTML = '';
    box.appendChild(div);
}

async function fetchSampleJson(url, options = {}, timeoutMs = SAMPLE_CONTROL_REQUEST_TIMEOUT_MS) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
    try {
        const response = await fetch(url, { ...options, signal: controller.signal });
        const body = await response.json().catch(() => ({}));
        return { response, body };
    } finally {
        window.clearTimeout(timeout);
    }
}

/**
 * Render server warnings into the warnings box. Each warning is a structured
 * `{ code, params }` object (the backend no longer emits pre-formatted English
 * strings). We emit a `data-i18n` span so applyToDOM translates it now AND
 * re-translates it on a UI language switch — a raw string would freeze in
 * whatever locale was active when the warning arrived. Legacy plain strings
 * are still tolerated for safety.
 */
function renderWarnings(box, warnings) {
    if (!box) return;
    if (!Array.isArray(warnings) || warnings.length === 0) {
        box.innerHTML = '';
        return;
    }
    box.innerHTML = warnings.map((w) => {
        if (w && typeof w === 'object' && w.code) {
            const key = `sample:${w.code}`;
            const params = w.params || {};
            const paramsAttr = DomHelpers.escapeHtml(JSON.stringify(params));
            return `<div class="sample-warning">⚠ <span data-i18n="${key}" data-i18n-params='${paramsAttr}'>${DomHelpers.escapeHtml(t(key, params))}</span></div>`;
        }
        return `<div class="sample-warning">⚠ ${DomHelpers.escapeHtml(String(w))}</div>`;
    }).join('');
    applyToDOM(box);
}

function setButtonsRunningState(running) {
    state.running = running;
    if (running) {
        state.lastRunStatus = 'running';
    } else if (state.lastRunStatus === 'running') {
        state.lastRunStatus = 'idle';
    }
    const runBtn = $('sampleRunBtn');
    const stopBtn = $('sampleStopBtn');
    const hasSamples = Array.isArray(state.lastItems) && state.lastItems.length > 0;
    if (runBtn) runBtn.disabled = running || state.initializingSamples || !hasSamples;
    if (stopBtn) stopBtn.classList.toggle('hidden', !running);
    // Freeze the whole column editor while a Run is in flight: changing a
    // provider/model mid-run would call refreshResultsFromCache() and clobber
    // the in-flight run's currentRunKeys + re-render, losing shimmer state.
    const columnsEl = $('sampleColumns');
    if (columnsEl) columnsEl.classList.toggle('is-running', running);
    syncSampleEditButtons();
    syncUpdateButton();
    renderRunStatus();
}

/**
 * Disable the per-card "remove" and the "add a sample" buttons while a Run
 * is in flight — editing the sample set mid-run would race with arriving
 * WebSocket cell_done events.
 *
 * Called after every render of #sampleResults (whose markup is re-built
 * from scratch each time) so the disabled state always reflects state.running.
 */
function syncSampleEditButtons() {
    const running = state.running;
    const addBtn = $('sampleAddSampleBtn');
    if (addBtn) addBtn.disabled = running;
    document.querySelectorAll('#sampleResults .sample-card-remove').forEach((btn) => {
        btn.disabled = running;
    });
    // Visual feedback: pending cells in other columns can't be clicked while
    // one column is currently translating.
    document.querySelectorAll('#sampleResults .sample-cell-pending').forEach((el) => {
        el.classList.toggle('is-disabled', running);
    });
}

/**
 * Fetch + render models for a column. Uses the SAME backend path and the SAME
 * per-provider rendering (Gemini token tooltips, OpenRouter/Poe pricing
 * labels, Poe optgroups, …) as the Settings panel. With `__USE_ENV__` the
 * server resolves the API key from `.env`.
 *
 * Returns the first model's value when the column had no model yet, so the
 * caller can keep `col.model` in sync.
 */
async function loadAndPopulateModelsForColumn(col, modelSelectEl) {
    setPlaceholderOption(modelSelectEl, 'common:loading');
    try {
        const data = await ApiClient.getModels(col.provider, {
            apiKey: '__USE_ENV__',
            // Endpoint applies to ollama + openai columns; ignore a value left
            // over from a different provider.
            apiEndpoint: columnEndpoint(col),
        });
        const models = data.models || [];
        if (!models.length) {
            setPlaceholderOption(modelSelectEl, 'settings:search_models_no_models_available');
            col.model = '';
            return '';
        }
        populateModelSelectInto(modelSelectEl, models, col.model || data.default || '', col.provider);
        // populateModelSelectInto leaves the native <select> value pointing at
        // the matched option (or the first if nothing matched); mirror that
        // into the column state so the next Run uses the visible selection.
        const picked = modelSelectEl.value;
        col.model = picked;
        return picked;
    } catch (err) {
        console.error('[sample] model fetch failed', err);
        setPlaceholderOption(modelSelectEl, 'settings:search_models_error');
        col.model = '';
        return '';
    }
}

function selectedGlossaryId() {
    return $('sampleGlossarySelect')?.value || '';
}

function selectedProfileId() {
    return $('sampleProfileSelect')?.value || '';
}

function selectedTransformMode() {
    return $('sampleTransformMode')?.value || '';
}

function currentModelColumnBase() {
    return {
        provider: state.modelConfig.provider || 'deepseek',
        model: state.modelConfig.model || '',
        api_key: '__USE_ENV__',
        api_endpoint: columnEndpoint(state.modelConfig),
        custom_instruction_file: state.modelConfig.custom_instruction_file || '',
    };
}

function makePipelineVariant(key, label, summary, options = {}) {
    const profileId = options.profileId || '';
    const transformMode = options.transformMode || '';
    return {
        ...currentModelColumnBase(),
        variant_key: key,
        label,
        variant_summary: summary,
        glossary_id: options.glossaryId || null,
        profile_id: profileId || null,
        refine_after_translate: !!transformMode,
        prompt_options: promptOptionsForVariant({ profileId, transformMode }),
    };
}

function buildPipelineColumns() {
    const glossaryId = selectedGlossaryId();
    const profileId = selectedProfileId();
    const transformMode = selectedTransformMode();
    const columns = [
        makePipelineVariant(
            'base',
            t('sample:variant_base'),
            t('sample:variant_base_desc'),
        ),
    ];

    if (glossaryId) {
        columns.push(makePipelineVariant(
            'glossary',
            t('sample:variant_glossary'),
            t('sample:variant_glossary_desc'),
            { glossaryId },
        ));
    }

    if (profileId) {
        columns.push(makePipelineVariant(
            'profile',
            t('sample:variant_profile'),
            t('sample:variant_profile_desc'),
            { profileId },
        ));
    }

    if (glossaryId && profileId) {
        columns.push(makePipelineVariant(
            'profile_glossary',
            t('sample:variant_profile_glossary'),
            t('sample:variant_profile_glossary_desc'),
            { glossaryId, profileId },
        ));
    }

    if (transformMode) {
        const transformName = transformLabelFor(transformMode);
        columns.push(makePipelineVariant(
            'transformed',
            t('sample:variant_transform', { mode: transformName }),
            t('sample:variant_transform_desc'),
            { glossaryId, profileId, transformMode },
        ));
    }

    return columns;
}

function renderVariantCard(col, idx) {
    const badges = [];
    if (col.glossary_id) badges.push(t('sample:badge_glossary'));
    if (col.profile_id) badges.push(t('sample:badge_profile'));
    if (col.refine_after_translate) badges.push(t('sample:badge_transform'));
    if (idx === 0) badges.push(t('sample:badge_baseline'));
    return `
        <article class="sample-variant-card" data-variant="${DomHelpers.escapeHtml(col.variant_key || '')}">
            <div class="sample-variant-icon">
                <span class="material-symbols-outlined">${idx === 0 ? 'radio_button_checked' : 'difference'}</span>
            </div>
            <div class="sample-variant-body">
                <strong>${DomHelpers.escapeHtml(col.label || `#${idx + 1}`)}</strong>
                <p>${DomHelpers.escapeHtml(col.variant_summary || '')}</p>
                <div class="sample-variant-badges">
                    ${badges.map((badge) => `<span>${DomHelpers.escapeHtml(badge)}</span>`).join('')}
                </div>
            </div>
        </article>
    `;
}

function renderColumns(options = {}) {
    const container = $('sampleColumns');
    if (!container) return;
    if ((state.running || state.snapshotViewLocked) && !options.force) {
        return;
    }
    if (!options.useCurrent) {
        state.columns = buildPipelineColumns();
    }
    container.innerHTML = state.columns.map((col, idx) => renderVariantCard(col, idx)).join('');
    applyToDOM(container);
    const countEl = $('sampleVariantCount');
    if (countEl) countEl.textContent = String(state.columns.length);
}

/**
 * Re-render the sample table from current `state.lastItems` × `state.columns`,
 * pulling whatever is cached from `resultsCache` when a run context is known.
 *
 * Called in three situations:
 *   - right after /initialize, before any Run (no lastRunContext, all
 *     skeletons but the cards are shown)
 *   - after add/remove sample or add/remove column (some prefilled, some
 *     skeletons)
 *   - inside runSample once the server returns items (most prefilled if the
 *     user keeps the same config)
 *
 * No-op if no file has been initialized yet (`state.lastItems === null`).
 */
function refreshResultsFromCache() {
    if (state.lastItems === null) return;
    if (state.running || state.snapshotViewLocked) return;
    const results = $('sampleResults');
    if (!results) return;

    const prefilled = new Map();
    state.currentRunKeys = new Map();

    if (state.lastRunContext) {
        state.lastItems.forEach((item, rowIdx) => {
            state.columns.forEach((col, colIdx) => {
                const key = buildCellKey(item, col, state.lastRunContext);
                state.currentRunKeys.set(`${rowIdx}:${colIdx}`, key);
                const cached = resultsCache.get(key);
                if (cached) {
                    const merged = {};
                    if (cached.translate) merged.translate = { status: 'done', ...cached.translate };
                    if (cached.refine) merged.refine = { status: 'done', ...cached.refine };
                    if (Object.keys(merged).length > 0) {
                        prefilled.set(`${rowIdx}:${colIdx}`, merged);
                    }
                }
            });
        });
    }

    const mode = (state.lastRunContext && state.lastRunContext.mode) || state.mode;
    SampleTable.render(results, state.lastItems, state.columns, mode, { prefilled });
    applyToDOM(results);
    syncSampleEditButtons();
    renderRunStatus();
    renderIntegratedResults();
    const copyBtn = $('sampleCopyMdBtn');
    if (copyBtn) copyBtn.disabled = !SampleTable.hasResults();
}

/**
 * Set the Sample-tab source-language <select> by language name, matching
 * options case-insensitively. Returns false if the language isn't an option.
 */
function setSampleSourceLang(languageValue) {
    const select = $('sampleSourceLang');
    if (!select || !languageValue || languageValue === 'Other') return false;
    for (const opt of select.options) {
        if (opt.value && opt.value.toLowerCase() === languageValue.toLowerCase()) {
            select.value = opt.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }
    }
    return false;
}

/**
 * Set the Sample-tab target-language <select> by language name, matching
 * options case-insensitively. Returns false if the language isn't an option.
 */
function setSampleTargetLang(languageValue) {
    const select = $('sampleTargetLang');
    if (!select || !languageValue || languageValue === 'Other') return false;
    for (const opt of select.options) {
        if (opt.value && opt.value.toLowerCase() === languageValue.toLowerCase()) {
            select.value = opt.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }
    }
    return false;
}

/**
 * Auto-detect the source language of the uploaded file and reflect it in the
 * source-language picker, so the user sees the detection immediately on drop
 * instead of leaving it on "Auto-detect". Best-effort: only applies on a
 * confident match (>= 0.7); silent on failure (the Run still auto-detects).
 */
async function detectAndSetSourceLanguage() {
    if (!state.uploadedPath) return;
    try {
        const result = await ApiClient.detectLanguage(state.uploadedPath);
        if (result && result.success && result.detected_language && (result.language_confidence || 0) >= 0.7) {
            const matched = setSampleSourceLang(result.detected_language);
            if (!matched) {
                showSampleMessage(
                    t('translation:lang_detected_not_in_list', { lang: result.detected_language }),
                    'info',
                );
            }
        }
    } catch (err) {
        console.warn('[sample] language detection failed', err);
    }
}

async function uploadFileIfNeeded() {
    if (state.uploadedPath) return state.uploadedPath;
    if (!state.file) {
        throw new Error(t('sample:error_no_file'));
    }
    const result = await ApiClient.uploadFile(state.file);
    state.uploadedPath = result.file_path;
    state.fileType = result.file_type;
    state.thumbnail = result.thumbnail || null;
    updateFileCard();
    return result.file_path;
}

/**
 * Refresh the selected-file card: cover (EPUB thumbnail or file-type icon),
 * filename, and a "size · type" details line. Called when a file is picked
 * and again once /api/upload returns a thumbnail.
 */
function updateFileCard() {
    if (!state.file) return;
    const nameEl = $('sampleFileName');
    const detailsEl = $('sampleFileDetails');
    const coverEl = $('sampleFileCover');
    if (nameEl) nameEl.textContent = state.file.name;

    if (detailsEl) {
        const sizeKb = state.file.size != null ? `${(state.file.size / 1024).toFixed(1)} KB` : '';
        const ext = (state.fileType || (state.file.name.split('.').pop() || '')).toUpperCase();
        detailsEl.textContent = [ext, sizeKb].filter(Boolean).join(' · ');
    }

    if (coverEl) {
        coverEl.innerHTML = '';
        if (state.thumbnail) {
            const img = document.createElement('img');
            img.src = `/api/thumbnails/${encodeURIComponent(state.thumbnail)}`;
            img.alt = '';
            img.onerror = () => {
                coverEl.innerHTML = `<span class="material-symbols-outlined">${iconForFileType(state.fileType)}</span>`;
            };
            coverEl.appendChild(img);
        } else {
            coverEl.innerHTML = `<span class="material-symbols-outlined">${iconForFileType(state.fileType)}</span>`;
        }
    }
}

function iconForFileType(ft) {
    const ext = (ft || '').toLowerCase();
    if (ext === 'epub') return 'menu_book';
    if (ext === 'srt')  return 'closed_caption';
    if (ext === 'pdf')  return 'picture_as_pdf';
    if (ext === 'docx') return 'description';
    if (ext === 'txt')  return 'article';
    return 'description';
}

/**
 * Toggle between the dropzone (no file picked yet) and the rich file card
 * (file picked). Hiding the dropzone makes the selected book unmistakable
 * and prevents accidental drag-drop of a different file mid-session.
 */
function showFileCardMode(hasFile) {
    const dropzone = $('sampleFileUpload');
    const card = $('sampleFileInfo');
    if (hasFile) {
        if (dropzone) dropzone.classList.add('hidden');
        DomHelpers.show(card);
    } else {
        if (dropzone) dropzone.classList.remove('hidden');
        DomHelpers.hide(card);
    }
}

/**
 * Triggered after the user picks/drops a file. Uploads it (if needed) and
 * calls /api/sample/initialize so the sample cards appear immediately, before
 * any LLM call. The user can then curate them (X / Add) at no token cost.
 */
async function initializeSamples() {
    if (!state.file) return;
    state.initializingSamples = true;
    state.pendingRunAfterInitialize = false;
    stopSamplePolling();
    clearPersistedSampleSession();
    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p class="sample-empty" data-i18n="sample:initializing">${t('sample:initializing')}</p>`;
    }
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentSampleId = null;
    state.currentRunKeys = new Map();
    state.activeRunCells = new Set();
    state.lastRunStatus = 'idle';
    state.lastProgressAt = 0;
    state.lastProgressCount = 0;
    state.snapshotViewLocked = false;
    syncRunningChrome(false);
    renderRunStatus();
    renderIntegratedResults();

    try {
        await uploadFileIfNeeded();
    } catch (err) {
        console.error('[sample] upload failed', err);
        showSampleMessage(err.message || String(err), 'error');
        state.initializingSamples = false;
        state.pendingRunAfterInitialize = false;
        syncRunningChrome(false);
        return;
    }
    // Reflect the detected source language in the picker right away; runs in
    // parallel so it never delays the sample cards.
    detectAndSetSourceLanguage();
    await _runInitialize({ preserveContext: false });
    state.initializingSamples = false;
    syncRunningChrome(false);
    if (state.pendingRunAfterInitialize && Array.isArray(state.lastItems) && state.lastItems.length > 0) {
        state.pendingRunAfterInitialize = false;
        runSample();
    }
}

/**
 * Re-sample the already-uploaded document with the current (N, max_chars)
 * inputs. Triggered by the "Update samples" button. Unlike a fresh upload,
 * this preserves `state.lastRunContext` so cells whose source_text still
 * matches a cached translation reappear instantly — only the *changed*
 * positions show as pending.
 */
async function refreshSampleSet() {
    if (state.running) return;
    if (!state.file || !state.uploadedPath) return;
    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    await _runInitialize({ preserveContext: true });
}

async function _runInitialize({ preserveContext }) {
    const warningsBox = $('sampleWarnings');
    const nSamples = parseInt($('sampleNSamples')?.value, 10) || SAMPLE_DEFAULT_N_SAMPLES;
    const maxChars = parseInt($('sampleMaxChars')?.value, 10) || SAMPLE_DEFAULT_MAX_CHARS;
    const mode = state.columns.some((col) => col.refine_after_translate) ? 'translate_refine' : 'translate';

    try {
        const r = await fetch('/api/sample/initialize', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: state.uploadedPath,
                file_type: state.fileType,
                n_samples: nSamples,
                max_chars: maxChars,
            }),
        });
        const body = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
        state.lastItems = body.items || [];
        state.appliedNSamples = nSamples;
        state.appliedMaxChars = maxChars;
        if (!preserveContext) {
            state.lastRunContext = null;
        }
        renderWarnings(warningsBox, body.warnings);
        refreshResultsFromCache();
        syncUpdateButton();
    } catch (err) {
        console.error('[sample] initialize failed', err);
        showSampleMessage(t('sample:initialize_failed', { error: err.message || String(err) }), 'error');
        state.pendingRunAfterInitialize = false;
        if (!preserveContext) {
            state.lastItems = [];
            const results = $('sampleResults');
            if (results) {
                results.innerHTML = `<p class="sample-empty sample-empty-error">${DomHelpers.escapeHtml(t('sample:initialize_failed', { error: err.message || String(err) }))}</p>`;
            }
        }
        syncUpdateButton();
        renderRunStatus();
        renderIntegratedResults();
    }
}

/**
 * Enable the "Update samples" button only when there's a loaded file, no Run
 * is in flight, AND at least one of the two number inputs differs from the
 * value that was last fed to /initialize.
 */
function syncUpdateButton() {
    const btn = $('sampleUpdateBtn');
    if (!btn) return;
    const nVal = parseInt($('sampleNSamples')?.value, 10);
    const mVal = parseInt($('sampleMaxChars')?.value, 10);
    const hasFile = !!state.uploadedPath;
    const dirty = (
        Number.isFinite(nVal) && Number.isFinite(mVal) &&
        (nVal !== state.appliedNSamples || mVal !== state.appliedMaxChars)
    );
    btn.disabled = !hasFile || state.running || !dirty;
}

function removeSample(rowIdx) {
    if (!Array.isArray(state.lastItems)) return;
    if (rowIdx < 0 || rowIdx >= state.lastItems.length) return;
    state.lastItems.splice(rowIdx, 1);
    refreshResultsFromCache();
}

async function addSample() {
    if (!state.uploadedPath || !state.fileType) {
        showSampleMessage(t('sample:error_no_file'), 'error');
        return;
    }
    const maxChars = parseInt($('sampleMaxChars')?.value, 10) || SAMPLE_DEFAULT_MAX_CHARS;
    const excludeIndices = Array.isArray(state.lastItems)
        ? state.lastItems.map((it) => it.index)
        : [];

    const addBtn = $('sampleAddSampleBtn');
    if (addBtn) addBtn.disabled = true;

    try {
        const r = await fetch('/api/sample/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                file_path: state.uploadedPath,
                file_type: state.fileType,
                max_chars: maxChars,
                exclude_indices: excludeIndices,
            }),
        });
        const body = await r.json().catch(() => ({}));
        if (r.status === 409) {
            showSampleMessage(t('sample:no_more_indices'), 'info');
            return;
        }
        if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);

        if (!Array.isArray(state.lastItems)) state.lastItems = [];
        state.lastItems.push(body.item);
        refreshResultsFromCache();
    } catch (err) {
        console.error('[sample] extract failed', err);
        showSampleMessage(t('sample:add_sample_failed', { error: err.message || String(err) }), 'error');
    } finally {
        const btn = $('sampleAddSampleBtn');
        if (btn) btn.disabled = false;
    }
}

function buildRunPayload({ defer = false } = {}) {
    state.columns = buildPipelineColumns();
    const sourceLang = $('sampleSourceLang')?.value || '';
    const targetLang = $('sampleTargetLang')?.value || '';
    const nSamples = parseInt($('sampleNSamples')?.value, 10) || SAMPLE_DEFAULT_N_SAMPLES;
    const maxChars = parseInt($('sampleMaxChars')?.value, 10) || SAMPLE_DEFAULT_MAX_CHARS;
    const runMode = state.columns.some((col) => col.refine_after_translate) ? 'translate_refine' : 'translate';

    const columns = state.columns.map((col) => ({
        provider: col.provider,
        model: col.model,
        api_key: '__USE_ENV__',
        // ollama + openai columns carry a custom endpoint; '' falls back to the
        // server config in _instantiate_provider.
        api_endpoint: columnEndpoint(col),
        // Per-LLM custom-instruction preset (file in Custom_Instructions/),
        // resolved server-side per column.
        custom_instruction_file: col.custom_instruction_file || '',
        // Per-LLM glossary; resolved + filtered per cell server-side.
        glossary_id: col.glossary_id || null,
        profile_id: col.profile_id || null,
        variant_key: col.variant_key || '',
        label: col.label || '',
        variant_summary: col.variant_summary || '',
        refine_after_translate: !!col.refine_after_translate,
        prompt_options: col.prompt_options || {},
    }));

    // Most editorial knobs now live per variant. Keep these run-wide so legacy
    // callers and the backend defaults continue to work.
    const promptOptions = basePromptOptions();

    const payload = {
        file_path: state.uploadedPath,
        file_type: state.fileType,
        source_language: sourceLang,
        target_language: targetLang,
        mode: runMode,
        n_samples: nSamples,
        max_chars: maxChars,
        columns,
        prompt_options: promptOptions,
        defer_dispatch: defer,
    };

    // The sample set is owned by the client once /initialize has run; pass it
    // along so the server doesn't re-shuffle behind the user's back.
    if (Array.isArray(state.lastItems)) {
        payload.items = state.lastItems.map((it) => ({
            index: it.index,
            source_text: it.source_text,
            truncated: !!it.truncated,
        }));
    }

    return payload;
}

/**
 * Kick off an LLM run.
 *
 * `opts.onlyColumn` (number) restricts the run to a single LLM column —
 * useful for the "click a grey cell to translate this column only" shortcut,
 * which avoids unload/reload churn on local providers like Ollama. When
 * omitted, every column is eligible (the global Run button).
 */
async function runSample(opts = {}) {
    if (state.running) return;
    if (state.initializingSamples) {
        state.pendingRunAfterInitialize = true;
        showSampleMessage(t('sample:initializing'), 'info');
        return;
    }
    state.snapshotViewLocked = false;
    renderColumns();
    const onlyColumn = (typeof opts.onlyColumn === 'number' && opts.onlyColumn >= 0)
        ? opts.onlyColumn
        : null;

    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';

    if (state.columns.length === 0) {
        showSampleMessage(t('sample:error_no_variants'), 'error');
        return;
    }
    if (onlyColumn !== null) {
        if (onlyColumn >= state.columns.length) return;
        const targetCol = state.columns[onlyColumn];
        if (!targetCol || !targetCol.model) {
            showSampleMessage(t('sample:error_llm_missing_model', { index: onlyColumn + 1 }), 'error');
            return;
        }
    } else {
        const missingModelIdx = state.columns.findIndex((c) => !c.model);
        if (missingModelIdx !== -1) {
            showSampleMessage(t('sample:error_llm_missing_model', { index: missingModelIdx + 1 }), 'error');
            return;
        }
    }
    if (!state.file) {
        showSampleMessage(t('sample:error_no_file'), 'error');
        return;
    }
    if (!Array.isArray(state.lastItems) || state.lastItems.length === 0) {
        showSampleMessage(t('sample:error_no_samples'), 'error');
        return;
    }

    try {
        setButtonsRunningState(true);
        await uploadFileIfNeeded();
    } catch (err) {
        console.error('[sample] upload failed', err);
        showSampleMessage(err.message || String(err), 'error');
        setButtonsRunningState(false);
        return;
    }

    // Phase 1 — server samples items and creates the state entry, but does NOT
    // yet spend any LLM tokens. We need the actual extracts before we can know
    // which cells are cached.
    const payload = buildRunPayload({ defer: true });

    let resp;
    try {
        const { response: r, body } = await fetchSampleJson('/api/sample/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        resp = body;
        if (!r.ok) {
            throw new Error(resp.error || `HTTP ${r.status}`);
        }
    } catch (err) {
        console.error('[sample] /api/sample/run failed', err);
        showSampleMessage(t('sample:error_run', { error: err.message || String(err) }), 'error');
        setButtonsRunningState(false);
        return;
    }

    state.currentSampleId = resp.sample_id;
    state.lastItems = resp.items;
    state.lastRunContext = {
        mode: resp.mode,
        source_lang: payload.source_language,
        target_lang: payload.target_language,
        prompt_options: payload.prompt_options,
        // glossary is per-column now (see buildCellKey), not run-wide.
    };
    persistSampleSession();
    startSamplePolling();

    // Phase 2 — compute cache hits, render the table (cached cells show their
    // content immediately, cells the server will work on now show the animated
    // shimmer, the rest fall back to the static "Click Run" hint), and tell
    // the server which (row, col) pairs to skip.
    const prefilled = new Map();
    const runningCells = new Set();
    const skipCells = [];
    state.currentRunKeys = new Map();

    resp.items.forEach((item, rowIdx) => {
        resp.columns.forEach((col, colIdx) => {
            const key = buildCellKey(item, col, state.lastRunContext);
            state.currentRunKeys.set(`${rowIdx}:${colIdx}`, key);
            const cached = resultsCache.get(key);
            const cellKey = `${rowIdx}:${colIdx}`;

            const mergedFromCache = () => {
                const merged = {};
                if (cached.translate) merged.translate = { status: 'done', ...cached.translate };
                if (cached.refine) merged.refine = { status: 'done', ...cached.refine };
                return merged;
            };

            // Off-target columns in a partial Run stay frozen — show what's
            // cached, leave the rest as the static pending hint, never run.
            if (onlyColumn !== null && colIdx !== onlyColumn) {
                if (isFullCacheHit(cached, resp.mode, col)) {
                    prefilled.set(cellKey, mergedFromCache());
                }
                skipCells.push([rowIdx, colIdx]);
                return;
            }

            if (isFullCacheHit(cached, resp.mode, col)) {
                prefilled.set(cellKey, mergedFromCache());
                skipCells.push([rowIdx, colIdx]);
            } else {
                runningCells.add(cellKey);
            }
        });
    });
    state.activeRunCells = new Set(runningCells);
    state.lastRunStatus = 'running';
    state.lastProgressAt = Date.now();
    state.lastProgressCount = 0;

    SampleTable.render($('sampleResults'), resp.items, resp.columns, resp.mode, { prefilled, runningCells });
    applyToDOM($('sampleResults'));
    syncSampleEditButtons();
    $('sampleCopyMdBtn').disabled = false;
    renderRunStatus();
    renderIntegratedResults();

    renderWarnings(warningsBox, resp.warnings);

    // Phase 3 — kick off the LLM work for non-cached cells. Server will emit
    // sample_done once finished (including when every cell is skipped).
    try {
        const { response: r, body: errBody } = await fetchSampleJson(`/api/sample/${encodeURIComponent(resp.sample_id)}/dispatch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ skip_cells: skipCells }),
        });
        if (!r.ok) {
            throw new Error(errBody.error || `HTTP ${r.status}`);
        }
        persistSampleSession();
        startSamplePolling();
    } catch (err) {
        console.error('[sample] /dispatch failed', err);
        await refreshSampleSnapshot().catch(() => false);
        if (state.lastRunStatus === 'running') {
            showSampleMessage(t('sample:status_waiting_elapsed', { seconds: Math.floor(SAMPLE_CONTROL_REQUEST_TIMEOUT_MS / 1000) }), 'info');
            startSamplePolling();
            return;
        }
        showSampleMessage(t('sample:error_run', { error: err.message || String(err) }), 'error');
        stopSamplePolling();
        setButtonsRunningState(false);
    }
}

async function stopSample() {
    if (!state.currentSampleId) return;
    try {
        await fetch(`/api/sample/${encodeURIComponent(state.currentSampleId)}/stop`, { method: 'POST' });
    } catch (err) {
        console.warn('stop sample failed', err);
    }
}

function copyAsMarkdown() {
    const md = SampleTable.toMarkdown();
    if (!md) return;
    navigator.clipboard.writeText(md).then(
        () => showSampleMessage(t('sample:copied_to_clipboard'), 'success'),
        (err) => showSampleMessage(t('sample:copy_failed', { error: err.message || String(err) }), 'error'),
    );
}

function reset() {
    state.file = null;
    state.uploadedPath = null;
    state.fileType = null;
    state.thumbnail = null;
    state.currentSampleId = null;
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentRunKeys = new Map();
    state.activeRunCells = new Set();
    state.lastRunStatus = 'idle';
    state.lastProgressAt = 0;
    state.lastProgressCount = 0;
    state.snapshotViewLocked = false;
    stopSamplePolling();
    clearPersistedSampleSession();
    showFileCardMode(false);
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p data-i18n="sample:no_results_yet">${t('sample:no_results_yet')}</p>`;
    }
    $('sampleCopyMdBtn').disabled = true;
    renderRunStatus();
    renderIntegratedResults();
}

function handleSampleUpdate(payload) {
    if (!payload || payload.sample_id !== state.currentSampleId) return;

    if (payload.type === 'cell_done' || payload.type === 'cell_error') {
        SampleTable.updateCell(payload);
        state.activeRunCells.delete(`${payload.row}:${payload.col}`);
        if (payload.type === 'cell_done') {
            const key = state.currentRunKeys.get(`${payload.row}:${payload.col}`);
            if (key) {
                const entry = resultsCache.get(key) || {};
                // `status: 'done'` is required by isFullCacheHit() so the next
                // Run treats this cell as a cache hit and skips the LLM call.
                entry[payload.phase] = {
                    status: 'done',
                    output: payload.output,
                    metrics: payload.metrics,
                };
                resultsCache.set(key, entry);
            }
        }
        markProgressHeartbeat();
        reconcileTerminalProgress();
        renderRunStatus();
        renderIntegratedResults();
        persistSampleSession();
        return;
    }

    if (payload.type === 'sample_done' || payload.type === 'sample_stopped') {
        state.activeRunCells = new Set();
        state.lastRunStatus = payload.type === 'sample_done' ? 'done' : 'stopped';
        stopSamplePolling();
        setButtonsRunningState(false);
        renderRunStatus();
        renderIntegratedResults();
        persistSampleSession();
        showSampleMessage(
            payload.type === 'sample_done'
                ? t('sample:run_done')
                : t('sample:run_stopped'),
            payload.type === 'sample_done' ? 'success' : 'info',
        );
    }
}

function onFileSelected(file) {
    state.file = file;
    state.uploadedPath = null;
    state.fileType = null;
    state.thumbnail = null;
    state.activeRunCells = new Set();
    state.lastRunStatus = 'idle';
    state.lastProgressAt = 0;
    state.lastProgressCount = 0;
    state.snapshotViewLocked = false;
    stopSamplePolling();
    clearPersistedSampleSession();
    updateFileCard();
    showFileCardMode(true);
    initializeSamples();
}

/**
 * Adopt a file that is ALREADY uploaded to the server — handed over from the
 * Translate tab's quick-test "Compare" action. Skips the drop + upload step
 * entirely: the path is set directly and the sample cards are initialized.
 *
 * Languages are seeded from the caller (the user's per-file choice in the
 * Translate queue) instead of auto-detected, so the carry-over is faithful.
 */
async function loadServerFile(info) {
    if (!info || !info.filePath) return;
    stopSamplePolling();
    clearPersistedSampleSession();
    state.file = { name: info.name || 'file', size: info.size || 0 };
    state.uploadedPath = info.filePath;
    state.fileType = info.fileType || null;
    state.thumbnail = info.thumbnail || null;
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentSampleId = null;
    state.currentRunKeys = new Map();
    state.appliedNSamples = null;
    state.appliedMaxChars = null;
    state.activeRunCells = new Set();
    state.lastRunStatus = 'idle';
    state.lastProgressAt = 0;
    state.lastProgressCount = 0;
    state.snapshotViewLocked = false;

    updateFileCard();
    showFileCardMode(true);

    if (info.sourceLanguage) setSampleSourceLang(info.sourceLanguage);
    if (info.targetLanguage) setSampleTargetLang(info.targetLanguage);

    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p class="sample-empty" data-i18n="sample:initializing">${t('sample:initializing')}</p>`;
    }
    renderRunStatus();
    renderIntegratedResults();
    await _runInitialize({ preserveContext: false });
}

/**
 * Clear the selected file: reset all sample state, switch the UI back to the
 * dropzone, and wipe the result table. Triggered by the X button on the file
 * card. No-op while a Run is in flight.
 */
function clearSelectedFile() {
    if (state.running) return;
    stopSamplePolling();
    clearPersistedSampleSession();
    state.file = null;
    state.uploadedPath = null;
    state.fileType = null;
    state.thumbnail = null;
    state.lastItems = null;
    state.lastRunContext = null;
    state.currentSampleId = null;
    state.currentRunKeys = new Map();
    state.appliedNSamples = null;
    state.appliedMaxChars = null;
    state.activeRunCells = new Set();
    state.lastRunStatus = 'idle';
    state.lastProgressAt = 0;
    state.lastProgressCount = 0;
    state.snapshotViewLocked = false;
    const fileInput = $('sampleFileInput');
    if (fileInput) fileInput.value = '';
    showFileCardMode(false);
    const warningsBox = $('sampleWarnings');
    if (warningsBox) warningsBox.innerHTML = '';
    const results = $('sampleResults');
    if (results) {
        results.innerHTML = `<p data-i18n="sample:no_results_yet">${t('sample:no_results_yet')}</p>`;
        applyToDOM(results);
    }
    const copyBtn = $('sampleCopyMdBtn');
    if (copyBtn) copyBtn.disabled = true;
    renderRunStatus();
    renderIntegratedResults();
    syncUpdateButton();
}

function wireFileInput() {
    const input = $('sampleFileInput');
    const uploadZone = $('sampleFileUpload');

    if (input) {
        input.addEventListener('change', (e) => {
            const f = e.target.files && e.target.files[0];
            if (!f) return;
            onFileSelected(f);
        });
    }

    if (uploadZone) {
        ['dragover', 'dragenter'].forEach((evt) => {
            uploadZone.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadZone.classList.add('drag-over');
            });
        });
        ['dragleave', 'drop'].forEach((evt) => {
            uploadZone.addEventListener(evt, (e) => {
                e.preventDefault();
                uploadZone.classList.remove('drag-over');
            });
        });
        uploadZone.addEventListener('drop', (e) => {
            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (!f) return;
            onFileSelected(f);
        });
    }
}

function initModelControls() {
    syncModelControlsFromState();
    const providerSelectEl = $('sampleProvider');
    const endpointInput = $('sampleEndpoint');
    const instructionSelect = $('sampleInstructions');

    if (providerSelectEl && !sampleSearchableIds.has('sampleProvider')) {
        attachProviderSearchable(providerSelectEl, {
            onChange: async (newProvider) => {
                state.snapshotViewLocked = false;
                state.modelConfig.provider = newProvider;
                state.modelConfig.model = '';
                state.modelConfig.api_endpoint = settingsEndpoints[newProvider] || '';
                syncModelControlsFromState();
                await loadAndPopulateBaseModel();
                renderColumns();
                refreshResultsFromCache();
            },
        });
        sampleSearchableIds.add('sampleProvider');
    }

    if (endpointInput) {
        endpointInput.addEventListener('change', async () => {
            state.snapshotViewLocked = false;
            state.modelConfig.api_endpoint = endpointInput.value.trim();
            await loadAndPopulateBaseModel();
            renderColumns();
            refreshResultsFromCache();
        });
    }

    if (instructionSelect) {
        instructionSelect.addEventListener('change', () => {
            state.snapshotViewLocked = false;
            state.modelConfig.custom_instruction_file = instructionSelect.value || '';
            renderColumns();
            refreshResultsFromCache();
        });
    }

    loadAndPopulateBaseModel().then(() => attachBaseModelSearchable());
}

function wireVariantSelectors() {
    ['sampleGlossarySelect', 'sampleProfileSelect', 'sampleTransformMode', 'sampleTargetLang', 'sampleSourceLang'].forEach((id) => {
        $(id)?.addEventListener('change', () => {
            state.snapshotViewLocked = false;
            renderColumns();
            refreshResultsFromCache();
        });
    });
    window.addEventListener('bookProfilesChanged', () => {
        loadBookProfiles();
    });
}

function wireResultsDelegation() {
    const results = $('sampleResults');
    if (!results) return;
    // SampleTable rebuilds the DOM on every render, so we delegate clicks
    // from the stable #sampleResults container instead of binding per-card.
    results.addEventListener('click', (e) => {
        const removeBtn = e.target.closest('.sample-card-remove');
        if (removeBtn) {
            const row = parseInt(removeBtn.dataset.row, 10);
            if (!Number.isNaN(row)) removeSample(row);
            return;
        }
        if (e.target.closest('#sampleAddSampleBtn')) {
            addSample();
            return;
        }
        // Click a grey "pending" cell to translate that LLM column only —
        // keeps the model loaded for every sample in one pass.
        const pending = e.target.closest('.sample-cell-pending');
        if (pending && !state.running) {
            const block = pending.closest('.sample-llm-block');
            if (block) {
                const colIdx = parseInt(block.dataset.col, 10);
                if (!Number.isNaN(colIdx)) {
                    runSample({ onlyColumn: colIdx });
                }
            }
        }
    });
}

function wireIntegratedResults() {
    const root = $('sampleIntegratedResults');
    if (!root) return;
    root.addEventListener('click', (e) => {
        const copyBtn = e.target.closest('[data-sample-integrated-copy]');
        if (!copyBtn) return;
        const colIdx = parseInt(copyBtn.dataset.sampleIntegratedCopy, 10);
        if (Number.isNaN(colIdx)) return;
        const text = integratedTextForColumn(colIdx);
        if (!text) return;
        navigator.clipboard.writeText(text).then(
            () => showSampleMessage(t('sample:integrated_copied'), 'success'),
            (err) => showSampleMessage(t('sample:copy_failed', { error: err.message || String(err) }), 'error'),
        );
    });
}

function wireButtons() {
    $('sampleRunBtn')?.addEventListener('click', runSample);
    $('sampleStopBtn')?.addEventListener('click', stopSample);
    $('sampleCopyMdBtn')?.addEventListener('click', copyAsMarkdown);
    $('sampleFileRemoveBtn')?.addEventListener('click', clearSelectedFile);
    $('sampleUpdateBtn')?.addEventListener('click', refreshSampleSet);

    // Live-enable the Update button as the user tweaks N or max_chars.
    ['sampleNSamples', 'sampleMaxChars'].forEach((id) => {
        $(id)?.addEventListener('input', syncUpdateButton);
    });

    wireResultsDelegation();
    wireIntegratedResults();
}

function rerenderOnLocale() {
    window.addEventListener('localeChanged', () => {
        renderColumns();
        if (SampleTable.hasResults()) {
            applyToDOM($('sampleResults'));
        }
        renderRunStatus();
        renderIntegratedResults();
    });
}

export const SampleManager = {
    init() {
        // Seed the N / max-chars inputs from the shared defaults so the constant
        // is the single source — the HTML no longer hard-codes its own value.
        const nInput = $('sampleNSamples');
        if (nInput && !nInput.value) nInput.value = String(SAMPLE_DEFAULT_N_SAMPLES);
        const maxInput = $('sampleMaxChars');
        if (maxInput && !maxInput.value) maxInput.value = String(SAMPLE_DEFAULT_MAX_CHARS);
        renderColumns();
        renderRunStatus();
        renderIntegratedResults();
        initModelControls();
        wireFileInput();
        wireVariantSelectors();
        wireButtons();
        rerenderOnLocale();
        WebSocketManager.on('sample_update', handleSampleUpdate);
        // Seed default endpoints from Settings; backfills + re-renders columns.
        loadSettingsEndpoints();
        // Load custom-instruction presets and glossaries for the per-LLM pickers.
        loadCustomInstructionFiles();
        loadGlossaries();
        loadBookProfiles();
        restoreSampleSession();
    },
    run: runSample,
    stop: stopSample,
    copyAsMarkdown,
    reset,
    loadServerFile,
};
