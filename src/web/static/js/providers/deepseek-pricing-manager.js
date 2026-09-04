import { ApiClient } from '../core/api-client.js';
import { StateManager } from '../core/state-manager.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { getCurrentLocale, t } from '../i18n/i18n.js';

const BLOCKED_ACTION_IDS = new Set([
    'translateBtn',
    'transformStartBtn',
    'profilePrepStartBtn',
    'sampleRunBtn',
    'sampleUpdateBtn',
    'quickTestCompareBtn',
]);

let availability = null;
let refreshTimer = null;
let countdownTimer = null;
let initialized = false;

function selectedProvider() {
    return String(DomHelpers.getValue('llmProvider') || '').trim().toLowerCase();
}

function isSelectedDeepSeekBlocked() {
    return selectedProvider() === 'deepseek' && Boolean(availability?.disabled);
}

function formatLocalDate(isoValue) {
    if (!isoValue) return t('common:no_data');
    const parsed = new Date(isoValue);
    if (Number.isNaN(parsed.getTime())) return isoValue;
    return new Intl.DateTimeFormat(getCurrentLocale(), {
        timeZone: availability?.display_timezone || 'America/Mexico_City',
        weekday: 'long',
        day: 'numeric',
        month: 'short',
        hour: 'numeric',
        minute: '2-digit',
    }).format(parsed);
}

function formatRemaining(ms) {
    const totalSeconds = Math.max(0, Math.ceil(ms / 1000));
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) return `${hours}h ${String(minutes).padStart(2, '0')}m`;
    return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

function updateCountdown() {
    const countdown = DomHelpers.getElement('deepseekPricingCountdown');
    if (!countdown || !availability?.next_available_at_utc) return;
    const remaining = new Date(availability.next_available_at_utc).getTime() - Date.now();
    countdown.textContent = t('common:deepseek_pricing_countdown', {
        time: formatRemaining(remaining),
    });
    if (remaining <= 0) {
        void DeepSeekPricingManager.refresh();
    }
}

function render() {
    const notice = DomHelpers.getElement('deepseekPricingNotice');
    const next = DomHelpers.getElement('deepseekPricingNext');
    const source = DomHelpers.getElement('deepseekPricingSource');
    const blocked = isSelectedDeepSeekBlocked();

    document.body.classList.toggle('deepseek-pricing-blocked', blocked);
    BLOCKED_ACTION_IDS.forEach((id) => {
        const control = DomHelpers.getElement(id);
        if (control) control.setAttribute('aria-disabled', blocked ? 'true' : 'false');
    });

    if (notice) notice.classList.toggle('hidden', !blocked);
    if (next) next.textContent = formatLocalDate(availability?.next_available_at_utc);
    if (source && availability?.source_url) source.href = availability.source_url;

    if (countdownTimer) clearInterval(countdownTimer);
    countdownTimer = blocked ? setInterval(updateCountdown, 1000) : null;
    if (blocked) updateCountdown();
}

function scheduleRefresh() {
    if (refreshTimer) clearTimeout(refreshTimer);
    let delay = 60_000;
    const boundary = availability?.disabled
        ? availability?.next_available_at_utc
        : availability?.next_peak_at_utc;
    if (boundary) {
        const untilBoundary = new Date(boundary).getTime() - Date.now() + 3000;
        if (untilBoundary > 0) delay = Math.min(delay, untilBoundary);
    }
    refreshTimer = setTimeout(() => void DeepSeekPricingManager.refresh(), Math.max(1000, delay));
}

function announceBlocked() {
    const when = formatLocalDate(availability?.next_available_at_utc);
    MessageLogger.showMessage(
        t('common:deepseek_pricing_action_blocked', { time: when }),
        'info',
    );
}

export const DeepSeekPricingManager = {
    init() {
        if (initialized) return;
        initialized = true;
        DomHelpers.getElement('llmProvider')?.addEventListener('change', render);
        window.addEventListener('localeChanged', render);
        document.addEventListener('click', (event) => {
            const target = event.target?.closest?.('button');
            if (!target || !BLOCKED_ACTION_IDS.has(target.id)) return;
            if (!isSelectedDeepSeekBlocked()) return;
            event.preventDefault();
            event.stopImmediatePropagation();
            announceBlocked();
        }, true);
        void this.refresh();
    },

    async refresh() {
        try {
            availability = await ApiClient.getDeepSeekAvailability();
            StateManager.setState('providers.deepseekPricing', availability);
            render();
            scheduleRefresh();
            return availability;
        } catch (error) {
            console.warn('[deepseek-pricing] availability check failed', error);
            scheduleRefresh();
            return availability;
        }
    },

    async ensureProviderAvailable(provider) {
        if (String(provider || '').toLowerCase() !== 'deepseek') return true;
        await this.refresh();
        if (!availability?.disabled) return true;
        announceBlocked();
        return false;
    },

    handleBlockedResponse(payload) {
        if (payload?.availability) {
            availability = payload.availability;
            StateManager.setState('providers.deepseekPricing', availability);
            render();
            scheduleRefresh();
        }
        announceBlocked();
    },

    isBlocked() {
        return isSelectedDeepSeekBlocked();
    },
};
