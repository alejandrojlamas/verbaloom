/**
 * Toast notifications
 *
 * Lightweight stacked toasts in the bottom-right corner. Used by the
 * glossary UI to replace `alert()`. Errors persist longer and expose a
 * close button; success/info auto-dismiss.
 *
 * API:
 *   toast.success(message, options?)
 *   toast.error(message, options?)
 *   toast.warn(message, options?)
 *   toast.info(message, options?)
 *
 * options:
 *   - duration (ms): override default. 0 means persistent.
 */

import { t } from '../i18n/i18n.js';

const CONTAINER_ID = '__app_toast_container';
const DEFAULT_DURATION = {
    success: 4000,
    info: 4000,
    warn: 6000,
    error: 0,  // persistent until user closes (or 8s fallback)
};

function ensureContainer() {
    let el = document.getElementById(CONTAINER_ID);
    if (el) return el;
    el = document.createElement('div');
    el.id = CONTAINER_ID;
    el.setAttribute('aria-live', 'polite');
    el.setAttribute('aria-atomic', 'false');
    el.style.cssText = `
        position: fixed;
        bottom: 1.25rem;
        right: 1.25rem;
        z-index: 10000;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
        max-width: min(420px, calc(100vw - 2.5rem));
        pointer-events: none;
    `;
    document.body.appendChild(el);
    return el;
}

function buildToast(type, message, options = {}) {
    const container = ensureContainer();
    const node = document.createElement('div');
    node.className = `app-toast app-toast-${type}`;
    node.setAttribute('role', type === 'error' ? 'alert' : 'status');
    node.style.pointerEvents = 'auto';

    const icon = document.createElement('span');
    icon.className = 'material-symbols-outlined app-toast-icon';
    icon.setAttribute('aria-hidden', 'true');
    icon.textContent = ({
        success: 'check_circle',
        error: 'error',
        warn: 'warning',
        info: 'info',
    })[type] || 'info';

    const text = document.createElement('div');
    text.className = 'app-toast-text';
    text.textContent = String(message == null ? '' : message);

    node.appendChild(icon);
    node.appendChild(text);

    const closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'app-toast-close';
    closeBtn.setAttribute('aria-label', t('common:toast_dismiss'));
    closeBtn.innerHTML = '<span class="material-symbols-outlined" aria-hidden="true">close</span>';

    let timeoutId = null;
    const dismiss = () => {
        if (timeoutId) clearTimeout(timeoutId);
        if (!node.parentNode) return;
        node.classList.add('app-toast-leaving');
        setTimeout(() => {
            if (node.parentNode) node.parentNode.removeChild(node);
        }, 180);
    };
    closeBtn.addEventListener('click', dismiss);
    node.appendChild(closeBtn);

    container.appendChild(node);
    requestAnimationFrame(() => node.classList.add('app-toast-show'));

    const baseDuration = DEFAULT_DURATION[type] != null ? DEFAULT_DURATION[type] : 4000;
    const duration = options.duration != null ? options.duration : baseDuration;
    if (duration > 0) {
        timeoutId = setTimeout(dismiss, duration);
    } else if (type === 'error') {
        timeoutId = setTimeout(dismiss, 12000);
    }

    return { dismiss };
}

export const toast = {
    success(msg, opts) { return buildToast('success', msg, opts); },
    error(msg, opts)   { return buildToast('error',   msg, opts); },
    warn(msg, opts)    { return buildToast('warn',    msg, opts); },
    info(msg, opts)    { return buildToast('info',    msg, opts); },
};

if (typeof window !== 'undefined') {
    window.toast = toast;
}
