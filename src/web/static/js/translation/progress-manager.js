/**
 * Progress Manager - Translation progress tracking and display
 *
 * Manages progress bar updates and statistics display for active translations.
 * Supports different file types (text, EPUB, SRT) with appropriate stat labels.
 */

import { DomHelpers } from '../ui/dom-helpers.js';
import { t } from '../i18n/i18n.js';
import { navigateToSetting } from '../ui/settings-summary.js';

/**
 * Format elapsed time in a human-readable format
 * - Under 60s: shows seconds (e.g., "45.2s")
 * - Under 1h: shows minutes and seconds (e.g., "5m 23s")
 * - 1h+: shows hours, minutes and seconds (e.g., "1h 23m 45s")
 * @param {number} seconds - Elapsed time in seconds
 * @returns {string} Formatted time string
 */
export function formatElapsedTime(seconds) {
    if (seconds < 60) {
        return seconds.toFixed(1) + 's';
    }

    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}h ${minutes}m ${secs}s`;
    }

    return `${minutes}m ${secs}s`;
}

/**
 * Format a deliberately low-precision ETA. Exact seconds imply certainty that
 * long LLM jobs do not have, especially while auditing or retrying a chunk.
 */
export function formatEstimatedDuration(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return null;
    if (value < 60) return '<1m';

    const totalMinutes = Math.max(1, Math.round(value / 60));
    if (totalMinutes < 60) return `${totalMinutes}m`;

    const hours = Math.floor(totalMinutes / 60);
    const minutes = Math.round((totalMinutes % 60) / 5) * 5;
    if (minutes === 60) return `${hours + 1}h`;
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
}

export function estimatedTimeLabel(stats = {}) {
    if (stats.eta_status === 'waiting') return t('translation:eta_waiting');
    if (stats.eta_status === 'finalizing') return t('translation:eta_finalizing');
    if (stats.eta_status === 'complete') return '0s';

    const estimate = formatEstimatedDuration(stats.eta_seconds);
    if (!estimate) return t('translation:eta_calculating');

    const lower = formatEstimatedDuration(stats.eta_lower_seconds);
    const upper = formatEstimatedDuration(stats.eta_upper_seconds);
    if (lower && upper && lower !== upper) return `${lower}–${upper}`;
    return `≈ ${estimate}`;
}

function updateEstimatedTimeRemaining(stats) {
    DomHelpers.setText('estimatedTimeRemaining', estimatedTimeLabel(stats));
}

/**
 * Update progress bar
 * @param {number} percent - Progress percentage (0-100)
 */
function updateProgressBar(percent) {
    const progressBar = DomHelpers.getElement('progressBar');

    if (progressBar) {
        progressBar.style.width = percent + '%';
        progressBar.textContent = Math.round(percent) + '%';
    }
}

const TRANSFORM_LABEL_TO_MODE = [
    ['explicar', 'simplify', 'Explain'],
    ['modernizar', 'modernize', 'Modernize'],
    ['humanizar', 'humanize', 'Humanize'],
    ['adaptar a mexicano', 'mexican_spanish', 'Adapt to Mexican Spanish'],
    ['audiolibro', 'audiobook', 'Audiobook'],
];

function labelForTransformMode(mode) {
    const normalized = String(mode || '').trim().toLowerCase();
    const found = TRANSFORM_LABEL_TO_MODE.find(([, candidateMode]) => candidateMode === normalized);
    return found ? found[2] : '';
}

function inferTransformFromFilename(stats = {}) {
    const outputName = String(stats.output_filename || stats.outputFilename || '').toLowerCase();
    for (const [needle, mode, label] of TRANSFORM_LABEL_TO_MODE) {
        if (outputName.includes(`(${needle}`)) {
            return { mode, label };
        }
    }
    return { mode: '', label: '' };
}

/**
 * Resolve the localized operation label ("Translating" / "Refining (2/2)" / etc.)
 * for the current phase of the workflow.
 *
 * @param {Object} stats - Server stats. May contain enable_refinement,
 *   current_phase, refine_only.
 * @returns {string} Label to show in the progress section title.
 */
export function resolveOperationLabel(stats) {
    if (!stats) return t('translation:translating');

    const inferredTransform = inferTransformFromFilename(stats);
    const transformMode = String(stats.text_transform_mode || stats.transform_mode || inferredTransform.mode || '').trim();
    const transformLabel = String(
        stats.text_transform_label
        || stats.transform_label
        || inferredTransform.label
        || labelForTransformMode(transformMode)
        || ''
    ).trim();
    if (transformMode || transformLabel || stats.operation === 'transform') {
        return transformLabel ? `Transforming · ${transformLabel}` : 'Transforming';
    }

    if (stats.inline_refinement) {
        return t('translation:full_flow_running');
    }

    if (stats.enable_refinement) {
        return stats.current_phase === 2
            ? t('translation:refining_step', { step: 2, total: 2, defaultValue: 'Refining (2/2)' })
            : t('translation:translating_step', { step: 1, total: 2, defaultValue: 'Translating (1/2)' });
    }

    if (stats.refine_only) {
        return t('translation:refining');
    }

    return t('translation:translating');
}

/**
 * Update the operation label inside the current-file title without rebuilding
 * the surrounding DOM (icon, filename, languages). The element is created by
 * `updateTranslationTitle` in the file controllers and tagged with id
 * `progressOperationLabel`.
 *
 * @param {Object} stats - Server stats
 */
function updateOperationLabel(stats) {
    const labelEl = DomHelpers.getElement('progressOperationLabel');
    if (!labelEl) return;
    labelEl.textContent = resolveOperationLabel(stats);
}

function resolveLiveStatusText(stats) {
    if (!stats) return '';
    if (stats.live_status) return String(stats.live_status);

    const completed = Number(stats.completed_chunks || 0);
    const total = Number(stats.total_chunks || 0);
    if (total > 0) {
        const next = Math.min(completed + 1, total);
        return `Chunk ${next}/${total}`;
    }
    return '';
}

function updateLiveStatus(stats) {
    const statusEl = DomHelpers.getElement('progressLiveStatus');
    if (!statusEl) return;

    const text = resolveLiveStatusText(stats);
    statusEl.textContent = text;
    statusEl.hidden = !text;
    statusEl.style.color = stats && stats.live_status_kind === 'stale'
        ? '#f59e0b'
        : 'var(--text-muted-light)';
}

// The Fallbacks card has a single active state — "critical" (red). Any
// fallback at all is treated as critical because placeholder loss already
// degrades the output; there is no intermediate "warning" tier any more.
//   false → nothing to surface (card neutral, panel hidden)
//   true  → red card + red panel with the full recommendation block
function isAlertActive(stats, fallbacks) {
    if (fallbacks > 0) return true;
    // Backend's one-shot quality flag and our client-side threshold check
    // are kept as additional ways to flip the card on even when fallbacks
    // are still 0 but placeholder errors are accumulating fast.
    if (stats && stats.quality_warning_fired) return true;
    return exceedsQualityThreshold(stats);
}

// Mirror of TranslationMetrics._QUALITY_* thresholds, kept here so the UI can
// detect the failure state from the cumulative stats payload even when no
// single XHTML file ever reached the per-file 5-chunk minimum (e.g. an EPUB
// split into many short chapters).
const QUALITY_MIN_PROCESSED = 5;
const QUALITY_RETRY_RATE_THRESHOLD = 0.30;
const QUALITY_FALLBACK_RATE_THRESHOLD = 0.10;
const QUALITY_AVG_ERRORS_THRESHOLD = 1.0;

function exceedsQualityThreshold(stats) {
    if (!stats) return false;
    const processed = stats.processed_chunks || 0;
    if (processed < QUALITY_MIN_PROCESSED) return false;
    const fallbackCount = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
    const notFirstTry = (stats.successful_after_retry || 0) + fallbackCount;
    const retryRate = notFirstTry / processed;
    const fallbackRate = fallbackCount / processed;
    const avgErrors = (stats.placeholder_errors || 0) / processed;
    return (
        retryRate > QUALITY_RETRY_RATE_THRESHOLD
        || fallbackRate > QUALITY_FALLBACK_RATE_THRESHOLD
        || avgErrors > QUALITY_AVG_ERRORS_THRESHOLD
    );
}

/**
 * Compute the rate metrics the critical intro line surfaces. We re-derive them
 * client-side from the cumulative stats payload so the numbers stay in sync
 * across multi-file EPUB runs (the per-file backend warning text would drift).
 */
export function deriveRateContext(stats) {
    const processed = stats.processed_chunks || stats.completed_chunks || 0;
    if (processed <= 0) {
        return { retryPct: 0, fallbackPct: 0, avgErrors: '0.0', processed: 0 };
    }
    const fallbackCount = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
    const notFirstTry = (stats.successful_after_retry || 0) + fallbackCount;
    const placeholderErrors = stats.placeholder_errors || 0;
    return {
        retryPct: Math.round((notFirstTry / processed) * 100),
        fallbackPct: Math.round((fallbackCount / processed) * 100),
        avgErrors: (placeholderErrors / processed).toFixed(1),
        processed,
    };
}

/**
 * Build a list item that interpolates a "{{link}}" placeholder in the given
 * template with a clickable link that jumps to the matching setting.
 */
function buildTipWithLink(template, linkText, action) {
    const li = document.createElement('li');
    const idx = template.indexOf('{{link}}');
    if (idx === -1) {
        // Defensive: if the i18n string lost its placeholder, fall back to
        // appending the link at the end so the user still has the affordance.
        li.appendChild(document.createTextNode(template + ' '));
    } else {
        li.appendChild(document.createTextNode(template.slice(0, idx)));
    }
    const link = document.createElement('a');
    link.href = '#';
    link.className = 'recommendation-link';
    link.textContent = linkText;
    link.addEventListener('click', (event) => {
        event.preventDefault();
        navigateToSetting(action);
    });
    li.appendChild(link);
    if (idx !== -1) {
        li.appendChild(document.createTextNode(template.slice(idx + '{{link}}'.length)));
    }
    return li;
}

/**
 * Build the recommendation content (intro line + tip list) into the given
 * container element. Shared between the live in-progress panel and the
 * post-translation completion card so the wording and ordering stay in sync.
 *
 * Caller owns the container styling (color palette, hidden state, etc.).
 *
 * @param {HTMLElement} container - Empty (or to-be-emptied) target element.
 * @param {Object} [context] - Numbers used by the intro line
 *   (retryPct, fallbackPct, avgErrors, processed).
 * @param {string} [introKey] - i18n key for the intro line. Defaults to the
 *   live-progress critical wording; the completion card overrides it with a
 *   past-tense variant.
 */
export function buildRecommendationContent(container, context, introKey) {
    if (!container) return;
    container.textContent = '';

    const ctx = context || { retryPct: 0, fallbackPct: 0, avgErrors: '0.0', processed: 0 };
    const intro = document.createElement('strong');
    intro.textContent = t(introKey || 'translation:fallback_panel_intro_critical', ctx);
    container.appendChild(intro);

    const list = document.createElement('ul');
    list.className = 'recommendation-list';

    const llmTip = document.createElement('li');
    llmTip.textContent = t('translation:fallback_panel_tip_llm');
    list.appendChild(llmTip);

    list.appendChild(buildTipWithLink(
        t('translation:fallback_panel_tip_plain_text_mode'),
        t('translation:fallback_panel_link_plain_text_mode'),
        'plainText',
    ));

    container.appendChild(list);
}

function renderRecommendationPanel(context) {
    buildRecommendationContent(
        DomHelpers.getElement('fallbackRecommendationPanel'),
        context,
    );
}

/**
 * Open/close the inline recommendation panel. When opening (or refreshing
 * while already open) we re-render with the latest context so the rate
 * numbers in the intro stay live.
 */
function toggleRecommendationPanel({ forceState, context } = {}) {
    const card = DomHelpers.getElement('fallbackStatCard');
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (!card || !panel) return;
    const willOpen = forceState !== undefined
        ? forceState
        : panel.hasAttribute('hidden');
    if (willOpen) {
        renderRecommendationPanel(context);
        panel.removeAttribute('hidden');
        card.setAttribute('aria-expanded', 'true');
    } else {
        panel.setAttribute('hidden', '');
        card.setAttribute('aria-expanded', 'false');
    }
}

// Latest derived rate context (retry %, fallback %, avg errors), so the click
// handler can re-render the panel without re-receiving the stats payload.
let _lastSeverityContext = null;

let _fallbackCardClickBound = false;

function bindFallbackCardClick() {
    if (_fallbackCardClickBound) return;
    const card = DomHelpers.getElement('fallbackStatCard');
    if (!card) return;
    card.addEventListener('click', () => {
        // Only react when the card is in the alert state.
        if (!card.classList.contains('stat-card-critical')) return;
        toggleRecommendationPanel({ context: _lastSeverityContext });
    });
    _fallbackCardClickBound = true;
}

/**
 * Apply the alert highlight to the Fallbacks card and keep the inline panel
 * in sync. Only one active state exists ("critical", red palette):
 *  - inactive → card neutral, panel hidden
 *  - active   → red card, red panel with live retry/fallback/avg-error rates
 *
 * The panel stays open if the user already opened it; we just refresh its
 * content. When dropping back to inactive, any open panel is closed.
 */
function updateFallbackHighlight(count, stats) {
    const card = DomHelpers.getElement('fallbackStatCard');
    if (!card) return;

    const active = isAlertActive(stats, count);
    _lastSeverityContext = active ? deriveRateContext(stats || {}) : null;

    card.classList.toggle('stat-card-critical', active);
    card.title = t(active
        ? 'translation:stat_fallbacks_tooltip_critical'
        : 'translation:stat_fallbacks_tooltip');

    if (!active) {
        toggleRecommendationPanel({ forceState: false });
        return;
    }

    bindFallbackCardClick();
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (panel && !panel.hasAttribute('hidden')) {
        renderRecommendationPanel(_lastSeverityContext);
    }
}

// Re-translate the Fallbacks card tooltip and the open recommendation panel
// when the UI locale changes. Both pieces are rendered imperatively (no
// data-i18n marker), so applyToDOM doesn't touch them — without this listener
// they would stay in the boot locale until the next stats update.
window.addEventListener('localeChanged', () => {
    const card = DomHelpers.getElement('fallbackStatCard');
    if (card) {
        const isCritical = card.classList.contains('stat-card-critical');
        card.title = t(isCritical
            ? 'translation:stat_fallbacks_tooltip_critical'
            : 'translation:stat_fallbacks_tooltip');
    }
    const panel = DomHelpers.getElement('fallbackRecommendationPanel');
    if (panel && !panel.hasAttribute('hidden') && _lastSeverityContext) {
        renderRecommendationPanel(_lastSeverityContext);
    }
});

/**
 * Update statistics display based on file type
 * All file types (txt, epub, srt) show stats uniformly
 * @param {Object} stats - Statistics object from server
 * @param {string} fileType - File type ('txt', 'epub', 'srt')
 */
function updateStatistics(stats, fileType) {
    if (!stats) return;

    DomHelpers.show('statsGrid');
    renderPromptContextSummary(stats.prompt_context || null);

    DomHelpers.setText('totalChunks', stats.total_chunks || '0');
    DomHelpers.setText('completedChunks', stats.completed_chunks || '0');
    DomHelpers.setText('failedChunks', stats.failed_chunks || '0');

    if (fileType === 'srt') {
        // SRT does not use placeholders → Fallbacks stays at 0.
        DomHelpers.setText('fallbackChunks', '0');
        updateFallbackHighlight(0, stats);
    } else {
        const fallbacks = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
        DomHelpers.setText('fallbackChunks', String(fallbacks));
        updateFallbackHighlight(fallbacks, stats);
    }

    if (stats.elapsed_time !== undefined) {
        DomHelpers.setText('elapsedTime', formatElapsedTime(stats.elapsed_time));
        updateEstimatedTimeRemaining(stats);
    }
}

function renderPromptContextSummary(promptContext) {
    const el = DomHelpers.getElement('promptContextSummary');
    if (!el) return;
    if (!promptContext || typeof promptContext !== 'object') {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    const activeProfile = promptContext.active_profile || {};
    const profileGlossary = promptContext.profile_glossary || {};
    const manualGlossary = promptContext.manual_glossary || {};
    const lastChunk = promptContext.last_chunk || {};
    const chips = [];
    if (activeProfile.profile_id) {
        chips.push({
            icon: 'auto_stories',
            text: t('translation:prompt_context_profile', { profile: activeProfile.profile_id }),
        });
    }
    if (profileGlossary.total_terms !== undefined) {
        chips.push({
            icon: 'travel_explore',
            text: t('translation:prompt_context_profile_glossary', {
                matched: profileGlossary.matched_terms || 0,
                total: profileGlossary.total_terms || 0,
                rendered: profileGlossary.rendered_terms || 0,
            }),
        });
    }
    if (manualGlossary.total_terms !== undefined) {
        chips.push({
            icon: 'menu_book',
            text: t('translation:prompt_context_manual_glossary', {
                matched: manualGlossary.matched_terms || 0,
                total: manualGlossary.total_terms || 0,
            }),
        });
    }
    if (lastChunk.chunk_sequence) {
        const chunkProfile = lastChunk.profile_glossary || {};
        const chunkManual = lastChunk.manual_glossary || {};
        chips.push({
            icon: 'data_object',
            text: t('translation:prompt_context_last_chunk', {
                chunk: lastChunk.chunk_sequence,
                profile: chunkProfile.rendered_terms || chunkProfile.matched_terms || 0,
                manual: chunkManual.matched_terms || 0,
            }),
        });
    }
    if (!chips.length) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    el.classList.remove('hidden');
    el.innerHTML = chips.map((chip) => `
        <span class="prompt-context-chip">
            <span class="material-symbols-outlined" aria-hidden="true">${chip.icon}</span>
            <span>${chip.text}</span>
        </span>
    `).join('');
}

export const ProgressManager = {
    /**
     * Update progress display
     * @param {number} percent - Progress percentage (0-100)
     */
    updateProgress(percent) {
        updateProgressBar(percent);
    },

    /**
     * Update statistics display
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     * @param {Object} stats - Statistics object from server
     */
    updateStats(fileType, stats) {
        updateStatistics(stats, fileType);
    },

    updateLiveStatus(stats) {
        updateLiveStatus(stats);
    },

    /**
     * Update progress and statistics together
     * @param {Object} data - Update data from server
     * @param {Object} data.stats - Statistics object
     * @param {string} fileType - File type ('txt', 'epub', 'srt')
     */
    update(data, fileType) {
        if (!data.stats) return;

        const stats = data.stats;
        const completed = stats.completed_chunks || 0;
        const total = stats.total_chunks || 0;

        // The bar value is server-authoritative: the backend emits a single
        // canonical `percent` (see src/core/progress) with the phase mapping
        // and monotonic floor already applied. Display it verbatim; the chunk
        // ratio is only a defensive fallback should `percent` ever be absent.
        const chunkPercent = total > 0 ? (completed / total) * 100 : 0;
        const globalPercent = typeof stats.percent === 'number'
            ? Math.max(stats.percent, chunkPercent)
            : chunkPercent;
        updateProgressBar(globalPercent);

        updateOperationLabel(stats);
        updateLiveStatus(stats);
        updateStatistics(stats, fileType);
    },

    /**
     * Reset progress display to initial state
     */
    reset() {
        updateProgressBar(0);
        DomHelpers.setText('totalChunks', '0');
        DomHelpers.setText('completedChunks', '0');
        DomHelpers.setText('failedChunks', '0');
        DomHelpers.setText('fallbackChunks', '0');
        _lastSeverityContext = null;
        updateFallbackHighlight(0);
        DomHelpers.setText('elapsedTime', '0s');
        DomHelpers.setText('estimatedTimeRemaining', '--');
        updateLiveStatus({ live_status: '' });
        renderPromptContextSummary(null);
        DomHelpers.hide('statsGrid');
    },

    /**
     * Show progress section
     */
    show() {
        DomHelpers.show('progressSection');
    },

    /**
     * Hide progress section
     */
    hide() {
        DomHelpers.hide('progressSection');
    },

    /**
     * Set progress to complete (100%)
     */
    complete() {
        updateProgressBar(100);
    },

    /**
     * Get current progress percentage
     * @returns {number} Current progress (0-100)
     */
    getCurrentProgress() {
        const progressBar = DomHelpers.getElement('progressBar');

        if (!progressBar) return 0;

        const widthStyle = progressBar.style.width;
        const match = widthStyle.match(/(\d+(?:\.\d+)?)/);

        return match ? parseFloat(match[1]) : 0;
    }
};
