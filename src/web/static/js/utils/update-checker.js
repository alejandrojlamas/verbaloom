/**
 * UpdateChecker - shows an update banner when a newer GitHub release exists
 * and drives the self-update flow (git pull + pip install + server restart).
 *
 * Endpoints used:
 *   GET  /api/version/check
 *   POST /api/version/update
 *   GET  /api/version/update/status (polled while update runs)
 */

import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

const DISMISS_STORAGE_KEY = 'tbl_update_dismissed_version';
const POLL_INTERVAL_MS = 1500;
const RESTART_WAIT_MAX_MS = 90_000;

export const UpdateChecker = {
    _pollTimer: null,
    _restartWaiter: null,
    _initialHealthSessionId: null,
    _lastCheck: null,

    async initialize() {
        this._wireBannerHandlers();
        this._wireOverlayHandlers();
        // Re-apply translations to the banner (version sentence + install
        // button label) when the UI locale changes, by replaying the last
        // /api/version/check response.
        window.addEventListener('localeChanged', () => {
            if (this._lastCheck) this._applyCheckResult(this._lastCheck);
        });
        try {
            await this.checkOnce();
        } catch (e) {
            console.warn('Initial version check failed:', e);
        }
    },

    _wireBannerHandlers() {
        const installBtn = document.getElementById('updateBannerInstallBtn');
        const closeBtn = document.getElementById('updateBannerCloseBtn');
        if (installBtn) {
            installBtn.addEventListener('click', () => this.handleInstallClick());
        }
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.dismissBanner());
        }
    },

    handleInstallClick() {
        // For PyInstaller bundles (or source zips without .git), auto-update via
        // git pull is impossible. Send the user to the GitHub release page so
        // they can download the new binary; otherwise run the in-place updater.
        const data = this._lastCheck;
        const kind = data && data.install_kind;
        const isFrozen = typeof kind === 'string' && (kind.startsWith('frozen') || kind === 'source');
        if (isFrozen) {
            const url = (data && data.release_url) || null;
            if (url) {
                window.open(url, '_blank', 'noopener');
            }
            this.dismissBanner();
            return;
        }
        this.startUpdate();
    },

    _wireOverlayHandlers() {
        const closeBtn = document.getElementById('updateOverlayCloseBtn');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hideOverlay());
        }
    },

    async checkOnce() {
        const resp = await fetch('/api/version/check');
        if (!resp.ok) return null;
        const data = await resp.json();
        this._applyCheckResult(data);
        return data;
    },

    _applyCheckResult(data) {
        this._lastCheck = data || null;
        const banner = document.getElementById('updateBanner');
        if (!banner) return;

        if (!data || !data.update_available || !data.latest) {
            banner.classList.add('hidden');
            return;
        }

        const dismissed = localStorage.getItem(DISMISS_STORAGE_KEY);
        if (dismissed && dismissed === String(data.latest)) {
            banner.classList.add('hidden');
            return;
        }

        const versionEl = document.getElementById('updateBannerVersion');
        const linkEl = document.getElementById('updateBannerLink');
        const installBtn = document.getElementById('updateBannerInstallBtn');

        if (versionEl) {
            versionEl.textContent = ` ${t('common:update_version_available', { latest: data.latest, current: data.current })}`;
        }
        if (linkEl && data.release_url) {
            linkEl.href = data.release_url;
        }
        if (installBtn) {
            // For packaged builds (PyInstaller .exe / .app) and source zips
            // without .git, the in-place updater can't run. Repurpose the
            // button as a "Download" link that opens the GitHub release page.
            const kind = data.install_kind || (data.git_repo === false ? 'source' : 'git');
            const isFrozen = typeof kind === 'string' && (kind.startsWith('frozen') || kind === 'source');
            if (isFrozen) {
                installBtn.disabled = false;
                installBtn.title = t('common:update_download_title');
                this._setInstallBtnText(installBtn, t('common:update_download'));
            } else {
                installBtn.disabled = false;
                installBtn.title = '';
                this._setInstallBtnText(installBtn, t('common:update_now'));
            }
        }

        banner.classList.remove('hidden');
    },

    _setInstallBtnText(btn, text) {
        // Replace the trailing text node (after the <span> icon) without
        // touching the icon itself.
        for (let i = btn.childNodes.length - 1; i >= 0; i--) {
            const node = btn.childNodes[i];
            if (node.nodeType === Node.TEXT_NODE) {
                node.textContent = ` ${text}`;
                return;
            }
        }
        btn.appendChild(document.createTextNode(` ${text}`));
    },

    dismissBanner() {
        const banner = document.getElementById('updateBanner');
        const versionEl = document.getElementById('updateBannerVersion');
        if (banner) banner.classList.add('hidden');
        if (versionEl) {
            const match = /v([0-9][\w.\-]*)/.exec(versionEl.textContent || '');
            if (match) {
                localStorage.setItem(DISMISS_STORAGE_KEY, match[1]);
            }
        }
    },

    async startUpdate() {
        const installBtn = document.getElementById('updateBannerInstallBtn');
        if (installBtn) installBtn.disabled = true;

        this.showOverlay();
        this._setOverlayStep(t('common:update_status_send'));

        try {
            const resp = await fetch('/api/version/update', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ install_deps: true, restart: true }),
            });
            const data = await resp.json().catch(() => ({}));

            if (!resp.ok) {
                const reason = data.message || data.error || `HTTP ${resp.status}`;
                this._setOverlayError(reason);
                if (installBtn) installBtn.disabled = false;
                MessageLogger.addLog(t('common:update_refused', { reason }));
                return;
            }

            MessageLogger.addLog(t('common:update_started'));
            this._captureCurrentSessionId();
            this._startPolling();
        } catch (e) {
            this._setOverlayError(t('common:update_contact_failed', { error: e.message }));
            if (installBtn) installBtn.disabled = false;
        }
    },

    async _captureCurrentSessionId() {
        try {
            const r = await fetch('/api/health');
            const j = await r.json();
            this._initialHealthSessionId = String(j.session_id || j.startup_time || '');
        } catch {
            this._initialHealthSessionId = null;
        }
    },

    _startPolling() {
        this._stopPolling();
        const tick = async () => {
            try {
                const r = await fetch('/api/version/update/status');
                if (!r.ok) return;
                const status = await r.json();
                this._applyStatus(status);

                if (status.state === 'failed') {
                    this._stopPolling();
                    return;
                }
                if (status.state === 'completed') {
                    if (status.requires_restart) {
                        this._setOverlayStep(t('common:update_status_restart'));
                        this._stopPolling();
                        this._waitForRestart();
                    } else {
                        this._stopPolling();
                        this._setOverlayStep(t('common:update_status_complete_manual'));
                        this._showOverlayCloseButton();
                    }
                }
            } catch (e) {
                // Network blip during restart is expected; let _waitForRestart handle it.
            }
        };
        tick();
        this._pollTimer = setInterval(tick, POLL_INTERVAL_MS);
    },

    _stopPolling() {
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
    },

    _waitForRestart() {
        const deadline = Date.now() + RESTART_WAIT_MAX_MS;
        const probe = async () => {
            try {
                const r = await fetch('/api/health', { cache: 'no-store' });
                if (r.ok) {
                    const j = await r.json();
                    const newSession = String(j.session_id || j.startup_time || '');
                    if (newSession && newSession !== this._initialHealthSessionId) {
                        this._setOverlayStep(t('common:update_status_restarted', { version: j.version }));
                        clearInterval(this._restartWaiter);
                        this._restartWaiter = null;
                        setTimeout(() => window.location.reload(), 700);
                        return;
                    }
                }
            } catch {
                // server is down during restart, keep probing
            }
            if (Date.now() > deadline) {
                clearInterval(this._restartWaiter);
                this._restartWaiter = null;
                this._setOverlayError(t('common:update_status_no_restart'));
                this._showOverlayCloseButton();
            }
        };
        this._restartWaiter = setInterval(probe, 2000);
        probe();
    },

    _applyStatus(status) {
        if (!status) return;
        if (status.step) {
            this._setOverlayStep(status.step);
        }
        const log = document.getElementById('updateOverlayLog');
        if (log && Array.isArray(status.output)) {
            log.textContent = status.output.join('\n');
            log.scrollTop = log.scrollHeight;
        }
        if (status.state === 'failed') {
            this._setOverlayError(status.error || t('common:update_status_failed'));
            this._showOverlayCloseButton();
        }
    },

    showOverlay() {
        const ov = document.getElementById('updateOverlay');
        const err = document.getElementById('updateOverlayError');
        const log = document.getElementById('updateOverlayLog');
        const closeBtn = document.getElementById('updateOverlayCloseBtn');
        if (ov) ov.classList.remove('hidden');
        if (err) { err.style.display = 'none'; err.textContent = ''; }
        if (log) log.textContent = '';
        if (closeBtn) closeBtn.style.display = 'none';
    },

    hideOverlay() {
        const ov = document.getElementById('updateOverlay');
        if (ov) ov.classList.add('hidden');
    },

    _setOverlayStep(text) {
        const el = document.getElementById('updateOverlayStep');
        if (el) el.textContent = text;
    },

    _setOverlayError(message) {
        const el = document.getElementById('updateOverlayError');
        if (el) {
            el.textContent = message;
            el.style.display = 'block';
        }
    },

    _showOverlayCloseButton() {
        const btn = document.getElementById('updateOverlayCloseBtn');
        if (btn) btn.style.display = 'inline-flex';
    },
};
