import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { t } from '../i18n/i18n.js';
import { TextTransformManager } from './text-transform-manager.js?v=20260705-transform-flow';

export const ProfilePrepManager = {
    pendingFiles: [],
    active: false,
    lastLogsSignature: '',
    lastJobSnapshot: null,
    pollFailureCount: 0,
    pollFailureStartedAt: 0,
    POLL_INTERVAL_MS: 1200,
    POLL_RECONNECT_MAX_DELAY_MS: 5000,
    MAX_POLL_FAILURE_MS: 180000,
    ACTIVE_JOB_STORAGE_KEY: 'tbl.activeProfilePreparationId',

    init() {
        this.bindEvents();
        this.renderPendingFiles();
        this.resetLiveView();
        void this.restoreTrackedJob();
    },

    bindEvents() {
        const upload = DomHelpers.getElement('profilePrepFileUpload');
        const input = DomHelpers.getElement('profilePrepFileInput');
        const start = DomHelpers.getElement('profilePrepStartBtn');
        const clear = DomHelpers.getElement('profilePrepClearBtn');

        if (upload) {
            upload.addEventListener('click', () => input?.click());
            upload.addEventListener('dragover', (event) => {
                event.preventDefault();
                upload.classList.add('dragging');
            });
            upload.addEventListener('dragleave', () => upload.classList.remove('dragging'));
            upload.addEventListener('drop', (event) => {
                event.preventDefault();
                upload.classList.remove('dragging');
                this.setPendingFiles(Array.from(event.dataTransfer?.files || []));
            });
        }

        if (input) {
            input.addEventListener('change', (event) => {
                this.setPendingFiles(Array.from(event.target.files || []));
                input.value = '';
            });
        }

        if (start) {
            start.addEventListener('click', () => this.start());
        }

        if (clear) {
            clear.addEventListener('click', () => {
                if (this.active) return;
                this.setPendingFiles([]);
                this.resetLiveView();
            });
        }
    },

    setPendingFiles(files) {
        this.pendingFiles = files.filter(Boolean);
        this.renderPendingFiles();
    },

    renderPendingFiles() {
        const list = DomHelpers.getElement('profilePrepPendingFiles');
        const start = DomHelpers.getElement('profilePrepStartBtn');
        const clear = DomHelpers.getElement('profilePrepClearBtn');
        if (!list) return;

        list.innerHTML = '';
        if (this.pendingFiles.length === 0) {
            list.innerHTML = `<div class="transform-empty">${DomHelpers.escapeHtml(t('transform:no_files'))}</div>`;
        } else {
            for (const file of this.pendingFiles) {
                const item = document.createElement('div');
                item.className = 'transform-file-row';
                item.innerHTML = `
                    <span class="material-symbols-outlined">description</span>
                    <span>${DomHelpers.escapeHtml(file.name)}</span>
                    <small>${(file.size / 1024).toFixed(1)} KB</small>
                `;
                list.appendChild(item);
            }
        }

        const disabled = this.pendingFiles.length === 0 || this.active;
        if (start) start.disabled = disabled;
        if (clear) clear.disabled = this.pendingFiles.length === 0 || this.active;
    },

    async start() {
        if (this.pendingFiles.length === 0) {
            MessageLogger.showMessage(t('transform:select_file_first'), 'info');
            return;
        }
        if (this.active) return;

        const language = DomHelpers.getValue('profilePrepLanguage') || DomHelpers.getValue('targetLang') || 'Spanish';
        const purpose = DomHelpers.getValue('profilePrepPurpose') || 'translation';
        const formData = new FormData();
        for (const file of this.pendingFiles) {
            formData.append('files', file, file.name);
        }
        formData.append('language', language);
        formData.append('target_locale', /^spanish$/i.test(language) ? 'es-MX' : '');
        formData.append('transform_mode', purpose);
        formData.append('profile_goal', purpose);
        formData.append('provider', 'deepseek');
        formData.append('model', 'deepseek-v4-flash');
        formData.append('review_model', 'deepseek-v4-pro');
        formData.append('llm_full_coverage', 'true');

        const deepseekKey = ApiKeyUtils.getValue('deepseekApiKey');
        if (deepseekKey) {
            formData.append('api_key', deepseekKey);
        }

        this.active = true;
        this.lastLogsSignature = '';
        this.lastJobSnapshot = null;
        this.pollFailureCount = 0;
        this.pollFailureStartedAt = 0;
        this.showLiveView();
        this.renderPendingFiles();
        this.setProgress(0, t('transform:profile_prep_running'));
        MessageLogger.showMessage(t('transform:profile_prep_started'), 'info');

        try {
            const started = await ApiClient.startBookProfilePreparation(formData);
            this.persistTrackedJob(started.prep_id);
            await this.followJob(started);
        } catch (error) {
            if (error.profileJobFailed) {
                this.clearTrackedJob();
            }
            this.setProgress(null, t('transform:profile_prep_error', { error: error.message }), 'error');
            MessageLogger.showMessage(t('transform:profile_prep_error', { error: error.message }), 'error');
        } finally {
            this.active = false;
            this.renderPendingFiles();
        }
    },

    async restoreTrackedJob() {
        if (this.active) return;

        let prepId = this.trackedJobId();
        let job = null;
        if (prepId) {
            try {
                job = await ApiClient.getBookProfilePreparation(prepId);
            } catch (error) {
                if (Number(error?.status || 0) === 404) {
                    this.clearTrackedJob(prepId);
                    prepId = '';
                } else {
                    return;
                }
            }
        }

        if (!job) {
            try {
                const response = await ApiClient.getBookProfilePreparations({
                    statuses: 'queued,running',
                    limit: 1,
                });
                job = Array.isArray(response?.jobs) ? response.jobs[0] : null;
            } catch (_error) {
                return;
            }
        }
        if (!job?.prep_id) return;

        this.persistTrackedJob(job.prep_id);
        this.active = true;
        this.lastLogsSignature = '';
        this.pollFailureCount = 0;
        this.pollFailureStartedAt = 0;
        this.showLiveView();
        this.renderPendingFiles();
        this.updateFromJob(job);
        MessageLogger.showMessage(
            t('transform:profile_prep_restored', {
                defaultValue: 'Seguimiento del perfil editorial restaurado.',
            }),
            'info',
        );

        try {
            await this.followJob(job);
        } catch (error) {
            if (error.profileJobFailed) {
                this.clearTrackedJob(job.prep_id);
            }
            this.setProgress(null, t('transform:profile_prep_error', { error: error.message }), 'error');
            MessageLogger.showMessage(t('transform:profile_prep_error', { error: error.message }), 'error');
        } finally {
            this.active = false;
            this.renderPendingFiles();
        }
    },

    async followJob(initialJob) {
        const prepId = initialJob?.prep_id;
        if (!prepId) throw new Error('Missing profile preparation job id.');
        this.updateFromJob(initialJob);
        if (initialJob.status === 'failed') {
            const jobError = new Error(initialJob.error || initialJob.message || 'Profile preparation failed.');
            jobError.profileJobFailed = true;
            throw jobError;
        }
        const job = initialJob.status === 'completed'
            ? initialJob
            : await this.poll(prepId);
        this.finish(job);
        this.clearTrackedJob(prepId);
        return job;
    },

    async poll(prepId) {
        if (!prepId) {
            throw new Error('Missing profile preparation job id.');
        }
        while (true) {
            await this.sleep(this.POLL_INTERVAL_MS);
            try {
                const job = await ApiClient.getBookProfilePreparation(prepId);
                this.pollFailureCount = 0;
                this.pollFailureStartedAt = 0;
                this.updateFromJob(job);
                if (job.status === 'completed') return job;
                if (job.status === 'failed') {
                    const jobError = new Error(job.error || job.message || 'Profile preparation failed.');
                    jobError.profileJobFailed = true;
                    throw jobError;
                }
            } catch (error) {
                if (!this.isTransientPollError(error)) {
                    throw error;
                }

                const now = Date.now();
                if (!this.pollFailureStartedAt) {
                    this.pollFailureStartedAt = now;
                }
                this.pollFailureCount += 1;
                const elapsedMs = now - this.pollFailureStartedAt;
                if (elapsedMs > this.MAX_POLL_FAILURE_MS) {
                    throw new Error(t('transform:profile_prep_reconnect_failed', {
                        error: error.message || '',
                        defaultValue: `La conexión con el servidor se perdió durante la preparación del perfil. El proceso puede seguir activo; actualiza la página para revisar si aparece como terminado. Último error: ${error.message || ''}`,
                    }));
                }

                this.showTransientPollIssue(error, elapsedMs);
                const retryDelay = Math.min(
                    this.POLL_RECONNECT_MAX_DELAY_MS,
                    700 + (this.pollFailureCount * 800),
                );
                await this.sleep(retryDelay);
            }
        }
    },

    updateFromJob(job) {
        if (!job) return;
        this.lastJobSnapshot = job;
        this.showLiveView();
        const percent = Number.isFinite(Number(job.progress)) ? Number(job.progress) : 0;
        const message = this.formatProgressMessage(job) || job.message || t('transform:profile_prep_running');
        const className = job.status === 'failed'
            ? 'error'
            : (job.status === 'completed' ? 'success' : '');
        this.setProgress(percent, t('transform:profile_prep_progress', {
            percent,
            message,
            defaultValue: `${percent}% · ${message}`,
        }), className);
        this.renderJobStats(job);
        this.renderLogs(job.logs || []);
    },

    finish(job) {
        this.pollFailureCount = 0;
        this.pollFailureStartedAt = 0;
        const profile = job?.result?.profile || {};
        const profileForSelect = {
            profile_id: profile.profile_id,
            name: profile.profile_name,
            approved_count: profile.approved_entries,
            pending_count: profile.pending_suggestions,
            editorial_artifact_counts: profile.editorial_artifact_counts || {},
            generated_profile: true,
        };
        TextTransformManager.upsertProfileOption(profileForSelect, true);
        window.dispatchEvent(new CustomEvent('bookProfilesChanged'));

        const signals = this.editorialSignalCount(profile);
        this.setProgress(100, t('transform:profile_prep_success', {
            approved: profile.approved_entries || 0,
            pending: profile.pending_suggestions || 0,
            signals,
        }), 'success');
        const result = DomHelpers.getElement('profilePrepResult');
        if (result) {
            result.textContent = t('transform:profile_prep_saved_result', {
                name: profile.profile_name || profile.profile_id || '',
                approved: profile.approved_entries || 0,
                pending: profile.pending_suggestions || 0,
                signals,
            });
        }
        MessageLogger.showMessage(t('transform:profile_prep_success_toast'), 'success');
    },

    trackedJobId() {
        try {
            return String(window.localStorage.getItem(this.ACTIVE_JOB_STORAGE_KEY) || '').trim();
        } catch (_error) {
            return '';
        }
    },

    persistTrackedJob(prepId) {
        if (!prepId) return;
        try {
            window.localStorage.setItem(this.ACTIVE_JOB_STORAGE_KEY, String(prepId));
        } catch (_error) {
            // The backend list endpoint remains the fallback when storage is unavailable.
        }
    },

    clearTrackedJob(prepId = '') {
        try {
            const current = window.localStorage.getItem(this.ACTIVE_JOB_STORAGE_KEY) || '';
            if (!prepId || current === String(prepId)) {
                window.localStorage.removeItem(this.ACTIVE_JOB_STORAGE_KEY);
            }
        } catch (_error) {
            // Ignore private-mode/storage failures; server-side recovery still works.
        }
    },

    isTransientPollError(error) {
        if (!error) return false;
        if (error.profileJobFailed) return false;
        if (error.network) return true;
        const status = Number(error.status || 0);
        if ([408, 425, 429, 500, 502, 503, 504].includes(status)) return true;
        const message = String(error.message || '').toLowerCase();
        return message.includes('failed to fetch')
            || message.includes('network')
            || message.includes('load failed')
            || message.includes('fetch');
    },

    showTransientPollIssue(error, elapsedMs) {
        const last = this.lastJobSnapshot;
        const percent = Number.isFinite(Number(last?.progress)) ? Number(last.progress) : 0;
        const seconds = Math.max(1, Math.round(elapsedMs / 1000));
        const message = t('transform:profile_prep_reconnecting', {
            seconds,
            error: error?.message || '',
            defaultValue: `Conexión inestable; reintentando consulta del perfil (${seconds}s). El análisis sigue en segundo plano.`,
        });
        this.setProgress(percent, message);
        if (last) {
            this.renderJobStats(last);
            this.renderLogs(last.logs || []);
        }
    },

    renderJobStats(job) {
        const logs = Array.isArray(job.logs) ? job.logs : [];
        const latest = logs.length ? logs[logs.length - 1] : {};
        const resultProfile = job?.result?.profile || {};
        const stage = job.current_stage || latest.stage || job.status || '';
        const isTermReview = stage === 'term_review' || latest.term_batch_total;
        const isLlmDiscovery = stage === 'llm_discovery';
        const chunkIndex = isTermReview
            ? (latest.term_batch_index ?? latest.chunk_index ?? '')
            : (latest.chunk_index ?? '');
        const chunkTotal = isTermReview
            ? (latest.term_batch_total ?? latest.chunk_total ?? '')
            : (latest.chunk_total ?? job.max_llm_chunks ?? '');
        const suggestions = latest.llm_suggestions ?? resultProfile.pending_suggestions ?? 0;
        const termsReviewed = latest.terms_reviewed
            ?? latest.reviewed_terms
            ?? resultProfile.term_review?.reviewed_terms
            ?? 0;
        const termsTotal = latest.terms_total ?? '';
        const signals = latest.editorial_artifacts
            ?? this.editorialSignalCount(resultProfile)
            ?? 0;

        this.setText('profilePrepStage', this.stageLabel(stage));
        this.setText(
            'profilePrepChunksLabel',
            t(isTermReview
                ? 'transform:profile_prep_stat_term_batches'
                : (isLlmDiscovery ? 'transform:profile_prep_stat_book_chunks' : 'transform:profile_prep_stat_chunks')),
        );
        this.setText(
            'profilePrepChunks',
            chunkIndex || chunkTotal ? `${chunkIndex || 0}/${chunkTotal || 0}` : '0/0',
        );
        this.setText(
            'profilePrepSuggestionsLabel',
            t(isTermReview ? 'transform:profile_prep_stat_terms' : 'transform:profile_prep_stat_suggestions'),
        );
        this.setText('profilePrepSuggestions', isTermReview
            ? `${termsReviewed || 0}/${termsTotal || 0}`
            : String(suggestions || 0));
        this.setText('profilePrepSignals', String(signals || 0));
        const discoveryModel = job.model || 'deepseek-v4-flash';
        const reviewModel = job.review_model || job.result?.profile?.review_model || 'deepseek-v4-pro';
        this.setText('profilePrepModelLabel', `${job.provider || 'deepseek'} · ${discoveryModel} · reviewer ${reviewModel}`);
    },

    formatProgressMessage(job) {
        const logs = Array.isArray(job?.logs) ? job.logs : [];
        const latest = logs.length ? logs[logs.length - 1] : {};
        const stage = job?.current_stage || latest.stage || job?.status || '';
        if (stage === 'term_review' || latest.term_batch_total) {
            const batchIndex = latest.term_batch_index ?? latest.chunk_index ?? 0;
            const batchTotal = latest.term_batch_total ?? latest.chunk_total ?? 0;
            const termsReviewed = latest.terms_reviewed ?? latest.reviewed_terms ?? 0;
            const termsTotal = latest.terms_total ?? 0;
            return t('transform:profile_prep_status_term_review', {
                current: batchIndex || 0,
                total: batchTotal || 0,
                termsCurrent: termsReviewed || 0,
                termsTotal: termsTotal || 0,
                defaultValue: `Revisión de glosario: lote ${batchIndex || 0}/${batchTotal || 0} · ${termsReviewed || 0}/${termsTotal || 0} términos`,
            });
        }
        if (stage === 'llm_discovery') {
            const chunkIndex = latest.chunk_index ?? 0;
            const chunkTotal = latest.chunk_total ?? job?.max_llm_chunks ?? 0;
            const suggestions = latest.llm_suggestions ?? 0;
            return t('transform:profile_prep_status_book_reading', {
                current: chunkIndex || 0,
                total: chunkTotal || 0,
                suggestions: suggestions || 0,
                defaultValue: `Lectura del libro: fragmento ${chunkIndex || 0}/${chunkTotal || 0} · ${suggestions || 0} sugerencias`,
            });
        }
        return '';
    },

    stageLabel(stage) {
        const keyByStage = {
            queued: 'profile_prep_stage_queued',
            profile_setup: 'profile_prep_stage_profile_setup',
            local_scan: 'profile_prep_stage_local_scan',
            editorial_map: 'profile_prep_stage_editorial_map',
            term_review: 'profile_prep_stage_term_review',
            local_glossary: 'profile_prep_stage_local_glossary',
            llm_discovery: 'profile_prep_stage_llm_discovery',
            llm_discovery_skipped: 'profile_prep_stage_llm_discovery_skipped',
            editorial_artifacts_saved: 'profile_prep_stage_editorial_artifacts_saved',
            profile_saved: 'profile_prep_stage_profile_saved',
            completed: 'profile_prep_stage_completed',
            failed: 'profile_prep_stage_failed',
        };
        const key = keyByStage[stage];
        if (!key) return stage || '—';
        return t(`transform:${key}`, { defaultValue: stage || '—' });
    },

    renderLogs(logs) {
        const list = DomHelpers.getElement('profilePrepLog');
        if (!list) return;
        const cleanLogs = Array.isArray(logs) ? logs.slice(-20) : [];
        const signature = JSON.stringify(cleanLogs.map((item) => [
            item.time,
            item.stage,
            item.message,
            item.progress,
            item.chunk_index,
            item.chunk_total,
            item.term_batch_index,
            item.term_batch_total,
            item.terms_reviewed,
            item.terms_total,
        ]));
        if (signature === this.lastLogsSignature) return;
        this.lastLogsSignature = signature;

        if (cleanLogs.length === 0) {
            list.innerHTML = `<div class="profile-prep-log-item"><span>${DomHelpers.escapeHtml(t('transform:profile_prep_no_logs'))}</span></div>`;
            return;
        }

        list.innerHTML = cleanLogs.reverse().map((item) => {
            const details = [];
            if (item.term_batch_index || item.term_batch_total) {
                details.push(t('transform:profile_prep_log_term_batches', {
                    current: item.term_batch_index || 0,
                    total: item.term_batch_total || 0,
                }));
            } else if (item.chunk_index || item.chunk_total) {
                details.push(t('transform:profile_prep_log_chunks', {
                    current: item.chunk_index || 0,
                    total: item.chunk_total || 0,
                }));
            }
            if (item.terms_reviewed !== undefined || item.terms_total !== undefined) {
                details.push(t('transform:profile_prep_log_terms', {
                    current: item.terms_reviewed || 0,
                    total: item.terms_total || 0,
                }));
            }
            if (item.llm_suggestions !== undefined) {
                details.push(t('transform:profile_prep_log_suggestions', {
                    count: item.llm_suggestions || 0,
                }));
            }
            if (item.editorial_artifacts !== undefined) {
                details.push(t('transform:profile_prep_log_artifacts', {
                    count: item.editorial_artifacts || 0,
                }));
            }
            if (item.reviewed_terms !== undefined) {
                details.push(`${item.reviewed_terms || 0} revisados`);
            }
            if (item.auto_approved_translations !== undefined) {
                details.push(`${item.auto_approved_translations || 0} traducciones`);
            }
            if (item.rejected_noise !== undefined) {
                details.push(`${item.rejected_noise || 0} ruido`);
            }
            const meta = [this.stageLabel(item.stage), `${item.progress ?? 0}%`, ...details].filter(Boolean).join(' · ');
            return `
                <div class="profile-prep-log-item">
                    <small>${DomHelpers.escapeHtml(meta)}</small>
                    <span>${DomHelpers.escapeHtml(item.message || '')}</span>
                </div>
            `;
        }).join('');
    },

    setProgress(percent, message, className = '') {
        const status = DomHelpers.getElement('profilePrepStatus');
        const progress = DomHelpers.getElement('profilePrepProgress');
        const bar = DomHelpers.getElement('profilePrepBar');
        if (status) {
            status.className = `profile-prep-status ${className}`.trim();
            status.textContent = message || '';
        }
        if (progress) {
            const shouldShow = percent !== null && percent !== undefined;
            progress.classList.toggle('hidden', !shouldShow);
            if (shouldShow) {
                const clean = Math.max(0, Math.min(100, Number(percent) || 0));
                progress.setAttribute('aria-valuenow', String(clean));
                if (bar) bar.style.width = `${clean}%`;
            }
        }
    },

    showLiveView() {
        DomHelpers.getElement('profilePrepLiveCard')?.classList.remove('hidden');
    },

    resetLiveView() {
        this.lastLogsSignature = '';
        this.setProgress(null, '');
        this.setText('profilePrepStage', '—');
        this.setText('profilePrepChunks', '0/0');
        this.setText('profilePrepChunksLabel', t('transform:profile_prep_stat_chunks'));
        this.setText('profilePrepSuggestionsLabel', t('transform:profile_prep_stat_suggestions'));
        this.setText('profilePrepSuggestions', '0');
        this.setText('profilePrepSignals', '0');
        this.setText('profilePrepResult', '');
        this.renderLogs([]);
    },

    setText(id, value) {
        const element = DomHelpers.getElement(id);
        if (element) element.textContent = value;
    },

    editorialSignalCount(profile) {
        const counts = profile?.editorial_artifact_counts || profile?.editorial_artifacts || {};
        if (!counts || typeof counts !== 'object') return 0;
        return Object.values(counts).reduce((sum, value) => {
            const number = Number(value || 0);
            return sum + (Number.isFinite(number) ? number : 0);
        }, 0);
    },

    sleep(ms) {
        return new Promise((resolve) => setTimeout(resolve, ms));
    },
};
