/**
 * Translation Tracker - Track active translations and handle WebSocket updates
 *
 * Manages active translation state, WebSocket event handling,
 * translation completion, error handling, and batch queue progression.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { StatusManager } from '../utils/status-manager.js';
import { FileUpload } from '../files/file-upload.js';
import { FileActions } from '../files/file-actions.js';
import { ProgressManager, formatElapsedTime, deriveRateContext, buildRecommendationContent } from './progress-manager.js?v=20260705-transform-flow';
import { renderTranslationTitle, getFileIcon, createGenericEPUBIcon } from './progress-title.js?v=20260705-transform-flow';
import { LifecycleManager } from '../utils/lifecycle-manager.js';
import { t } from '../i18n/i18n.js';

// Storage configuration with versioning
const STORAGE_VERSION = 1;
const STORAGE_KEY_PREFIX = 'verbaloom_translation_state';
const TRANSLATION_STATE_STORAGE_KEY = `${STORAGE_KEY_PREFIX}_v${STORAGE_VERSION}`;
const TERMINAL_STATUSES = new Set(['completed', 'error', 'interrupted', 'rate_limited', 'partial']);
const ACTIVE_STATUSES = new Set(['running', 'queued', 'pricing_wait', 'provider_wait']);
const TRANSFORM_MODE_LABELS = {
    modernize: 'Modernize',
    simplify: 'Explain',
    humanize: 'Humanize',
    mexican_spanish: 'Adapt to Mexican Spanish',
    audiobook: 'Audiobook',
};

function inferTransformMetadata(job = {}) {
    const config = job.config || {};
    const promptOptions = config.prompt_options || {};
    const outputName = String(job.output_filename || config.output_filename || '');
    const normalizedOutput = outputName.toLowerCase();
    let mode = String(
        job.text_transform_mode
        || job.transform_mode
        || promptOptions.text_transform_mode
        || ''
    ).trim().toLowerCase();
    let label = String(
        job.text_transform_label
        || job.transform_label
        || promptOptions.text_transform_label
        || ''
    ).trim();

    if (!mode) {
        if (normalizedOutput.includes('(explicar')) mode = 'simplify';
        else if (normalizedOutput.includes('(modernizar')) mode = 'modernize';
        else if (normalizedOutput.includes('(humanizar')) mode = 'humanize';
        else if (normalizedOutput.includes('(audiolibro')) mode = 'audiobook';
    }
    if (!label && mode) {
        label = TRANSFORM_MODE_LABELS[mode] || mode.replace(/_/g, ' ');
    }

    const profileId = String(
        job.profile_id
        || job.book_profile_id
        || promptOptions.profile_id
        || ''
    ).trim();

    return {
        mode,
        label,
        profileId,
        isTransform: Boolean(mode),
    };
}

/**
 * Validate translation state structure
 * @param {any} data - Data to validate
 * @returns {boolean} True if valid
 */
function validateTranslationState(data) {
    if (!data || typeof data !== 'object') return false;

    // Check required fields
    if (!('version' in data)) return false;
    if (!('currentJob' in data)) return false;
    if (!('isBatchActive' in data)) return false;
    if (!('activeJobs' in data)) return false;
    if (!('hasActive' in data)) return false;

    // Validate types
    if (typeof data.isBatchActive !== 'boolean') return false;
    if (typeof data.hasActive !== 'boolean') return false;
    if (!Array.isArray(data.activeJobs)) return false;

    // Validate currentJob if present
    if (data.currentJob !== null) {
        if (typeof data.currentJob !== 'object') return false;
        if (!('translationId' in data.currentJob)) return false;
        if (!('fileRef' in data.currentJob)) return false;
    }

    return true;
}

export const TranslationTracker = {
    // Debounce timer for saving state
    _saveStateTimer: null,
    _saveStateDebounceMs: 100,
    _initializationPromise: null,
    _manualRepairStatsByTranslation: new Map(),
    _manualRepairStatsTtlMs: 45000,
    _liveProgressPollTimer: null,
    _liveProgressPollInFlight: false,
    _liveProgressPollIntervalMs: 5000,
    _liveProgressLastCompleted: new Map(),
    _liveProgressLastChangeAt: new Map(),
    _liveActivityByTranslation: new Map(),

    /**
     * Initialize translation tracker
     */
    async initialize() {
        if (this._initializationComplete === true) {
            return;
        }
        if (this._initializationPromise) {
            return this._initializationPromise;
        }

        this._initializationPromise = this._runInitialization();
        try {
            await this._initializationPromise;
        } finally {
            this._initializationPromise = null;
        }
    },

    async _runInitialization() {
        // Clean up old storage versions
        this.cleanupOldStorageVersions();

        // Setup event listeners FIRST (they need to be ready before any state changes)
        this.setupEventListeners();

        // CRITICAL: Check server session BEFORE restoring state
        // This prevents restoring state from a previous server session
        try {
            const serverWasRestarted = await LifecycleManager.getServerSessionCheck();

            if (serverWasRestarted) {
                this.initializeDefaultTranslationState();
                await Promise.all([
                    this.updateActiveTranslationsState(),
                    this.restoreActiveTranslation()
                ]);
            } else {
                this.restoreTranslationStateSync();

                await Promise.all([
                    this.updateActiveTranslationsState(),
                    this.reconcileStateWithServer()
                ]);
            }
        } catch (error) {
            console.error('Failed to initialize translation state:', error);
            MessageLogger.addLog(t('translation:session_init_failed'));

            // Fallback: restore from localStorage anyway
            this.restoreTranslationStateSync();
        }

        // Mark initialization as complete
        this._initializationComplete = true;
    },

    /**
     * Check if initialization is complete
     * @returns {boolean} True if initialization is complete
     */
    isInitialized() {
        return this._initializationComplete === true;
    },

    /**
     * Clean up old localStorage versions
     */
    cleanupOldStorageVersions() {
        try {
            // Silently carry an active pre-VerbaLoom session forward.
            const legacyKeys = ['tbl_translation_state_v1', 'tbl_translation_state'];
            for (const legacyKey of legacyKeys) {
                const legacyState = localStorage.getItem(legacyKey);
                if (legacyState && !localStorage.getItem(TRANSLATION_STATE_STORAGE_KEY)) {
                    try {
                        const parsed = JSON.parse(legacyState);
                        parsed.version = STORAGE_VERSION;
                        localStorage.setItem(TRANSLATION_STATE_STORAGE_KEY, JSON.stringify(parsed));
                    } catch (error) {
                        console.warn('Could not migrate legacy translation state:', error);
                    }
                }
                localStorage.removeItem(legacyKey);
            }

            // Remove any other versions (future-proofing)
            for (let i = 0; i < STORAGE_VERSION; i++) {
                const oldVersionKey = `${STORAGE_KEY_PREFIX}_v${i}`;
                if (localStorage.getItem(oldVersionKey)) {
                    localStorage.removeItem(oldVersionKey);
                }
            }
        } catch (error) {
            console.warn('Failed to cleanup old storage versions:', error);
        }
    },

    /**
     * Restore translation state from localStorage synchronously
     * This ensures the UI shows the translation state immediately on page load
     */
    restoreTranslationStateSync() {
        try {
            const stored = localStorage.getItem(TRANSLATION_STATE_STORAGE_KEY);

            if (!stored) {
                this.initializeDefaultTranslationState();
                return;
            }

            const savedState = JSON.parse(stored);

            if (!validateTranslationState(savedState)) {
                MessageLogger.addLog(t('translation:session_corrupted_log'));
                this.initializeDefaultTranslationState();
                this.clearTranslationState();
                return;
            }

            if (savedState.version !== STORAGE_VERSION) {
                this.initializeDefaultTranslationState();
                this.clearTranslationState();
                return;
            }

            if (savedState.isBatchActive && savedState.currentJob) {
                StateManager.setState('translation.currentJob', savedState.currentJob);
                StateManager.setState('translation.isBatchActive', savedState.isBatchActive);
                StateManager.setState('translation.activeJobs', savedState.activeJobs || []);
                StateManager.setState('translation.hasActive', savedState.hasActive || false);

                DomHelpers.show('progressSection');
                DomHelpers.show('interruptBtn');

                const translateBtn = DomHelpers.getElement('translateBtn');
                if (translateBtn) {
                    translateBtn.disabled = true;
                    translateBtn.innerHTML = t('translation:batch_in_progress');
                }

                MessageLogger.addLog(t('translation:session_restored_log'));
            } else {
                this.initializeDefaultTranslationState();
            }
        } catch (error) {
            console.error('Failed to restore translation state from localStorage:', error);
            MessageLogger.addLog(t('translation:session_could_not_restore'));
            this.initializeDefaultTranslationState();
        }
    },

    /**
     * Reconcile local state with server state
     * Checks if localStorage state matches server reality
     */
    async reconcileStateWithServer() {
        try {
            const currentJob = StateManager.getState('translation.currentJob');

            // If we have a local job, verify it exists on server
            if (currentJob && currentJob.translationId) {
                try {
                    const serverState = await ApiClient.getTranslationStatus(currentJob.translationId);

                    if (TERMINAL_STATUSES.has(serverState.status)) {

                        MessageLogger.addLog(t('translation:session_sync_log', { status: serverState.status }));
                        this.resetUIToIdle();
                    } else if (ACTIVE_STATUSES.has(serverState.status)) {
                        // Calculate progress from stats if available
                        if (serverState.stats) {
                            this.updateStats(
                                currentJob.fileRef.fileType || currentJob.fileRef.type || 'txt',
                                serverState.stats
                            );
                        }
                    } else {
                        MessageLogger.addLog(t('translation:session_sync_log', { status: serverState.status || 'missing' }));
                        this.resetUIToIdle();
                    }
                } catch (error) {
                    if (error.status === 404 || (error.message && error.message.includes('404'))) {
                        MessageLogger.addLog(t('translation:session_job_missing_log'));
                        this.resetUIToIdle();
                    }
                }
            }

            await this.restoreActiveTranslation();

        } catch (error) {
            console.warn('Failed to reconcile state with server:', error);
        }
    },

    /**
     * Initialize default translation state (when no saved state exists)
     */
    initializeDefaultTranslationState() {
        StateManager.setState('translation.currentJob', null);
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.activeJobs', []);
        StateManager.setState('translation.hasActive', false);
    },

    /**
     * Save translation state to localStorage (debounced)
     */
    saveTranslationState() {
        // Clear existing timer
        if (this._saveStateTimer) {
            clearTimeout(this._saveStateTimer);
        }

        // Debounce to avoid multiple rapid saves
        this._saveStateTimer = setTimeout(() => {
            this._performSaveTranslationState();
        }, this._saveStateDebounceMs);
    },

    /**
     * Perform the actual save to localStorage
     * @private
     */
    _performSaveTranslationState() {
        try {
            const state = {
                version: STORAGE_VERSION,
                currentJob: StateManager.getState('translation.currentJob'),
                isBatchActive: StateManager.getState('translation.isBatchActive'),
                activeJobs: StateManager.getState('translation.activeJobs'),
                hasActive: StateManager.getState('translation.hasActive'),
                timestamp: Date.now()
            };

            localStorage.setItem(TRANSLATION_STATE_STORAGE_KEY, JSON.stringify(state));
        } catch (error) {
            console.error('Failed to save translation state to localStorage:', error);

            // Check if it's a quota exceeded error
            if (error.name === 'QuotaExceededError') {
                MessageLogger.addLog(t('translation:session_state_save_quota'));
            } else {
                MessageLogger.addLog(t('translation:session_state_save_failed'));
            }
        }
    },

    /**
     * Clear translation state from localStorage
     */
    clearTranslationState() {
        try {
            // Clear any pending save
            if (this._saveStateTimer) {
                clearTimeout(this._saveStateTimer);
                this._saveStateTimer = null;
            }

            localStorage.removeItem(TRANSLATION_STATE_STORAGE_KEY);
        } catch (error) {
            console.error('Failed to clear translation state from localStorage:', error);
        }
    },

    /**
     * Restore active translation state if there's one running on the server
     */
    async restoreActiveTranslation() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                job => ACTIVE_STATUSES.has(job.status)
            );

            if (activeJobs.length === 0) return;

            // Find matching file in our queue
            const filesToProcess = StateManager.getState('files.toProcess') || [];

            for (const job of activeJobs) {
                const transformMeta = inferTransformMetadata(job);
                let matchingFile = filesToProcess.find(f =>
                    f.translationId === job.translation_id ||
                    f.filePath === job.input_file ||
                    f.name === job.input_file?.split('/').pop()
                );

                // If no matching file found, create a virtual file reference from server data
                // This allows restoration after browser refresh even if filesToProcess is empty
                const virtualName = (
                    job.input_filename
                    || job.output_filename
                    || job.config?.output_filename
                    || `Translation ${job.translation_id}`
                );
                if (!matchingFile) {
                    matchingFile = {
                        name: virtualName,
                        translationId: job.translation_id,
                        status: 'Processing',
                        type: job.file_type || 'txt',
                        fileType: job.file_type || 'txt',
                        operation: transformMeta.isTransform ? 'transform' : (job.refine_only ? 'refine' : 'translate'),
                        transformMode: transformMeta.mode,
                        transformLabel: transformMeta.label,
                        profileId: transformMeta.profileId,
                        sourceLanguage: job.source_language || job.config?.source_language || '',
                        targetLanguage: job.target_language || job.config?.target_language || '',
                        outputFilename: job.output_filename || job.config?.output_filename || '',
                        isVirtual: true
                    };
                } else if (transformMeta.isTransform) {
                    matchingFile.operation = 'transform';
                    matchingFile.transformMode = matchingFile.transformMode || transformMeta.mode;
                    matchingFile.transformLabel = matchingFile.transformLabel || transformMeta.label;
                    matchingFile.profileId = matchingFile.profileId || transformMeta.profileId;
                }

                if (matchingFile) {
                    StateManager.setState('translation.currentJob', {
                        fileRef: matchingFile,
                        translationId: job.translation_id
                    });
                    StateManager.setState('translation.isBatchActive', true);

                    DomHelpers.show('progressSection');
                    this.updateTranslationTitle(matchingFile);

                    // Calculate progress from stats (job contains total_chunks, completed_chunks, etc.)
                    if (job.total_chunks > 0) {
                        const stats = this._statsFromActiveJob(job);
                        this.updateStats(
                            matchingFile.fileType || matchingFile.type || job.file_type || 'txt',
                            stats
                        );
                    }

                    if (job.last_translation) {
                        MessageLogger.updateTranslationPreview(job.last_translation);
                    }

                    this.startLiveProgressPolling();

                    const translateBtn = DomHelpers.getElement('translateBtn');
                    if (translateBtn) {
                        translateBtn.disabled = true;
                        translateBtn.innerHTML = t('translation:batch_in_progress');
                    }
                    DomHelpers.show('interruptBtn');

                    if (!matchingFile.isVirtual) {
                        this.updateFileStatusInList(matchingFile.name, 'Processing', job.translation_id);
                    }

                    break;
                }
            }
        } catch (error) {
            console.warn('Failed to restore active translation:', error);
        }
    },

    setupEventListeners() {
        StateManager.subscribe('translation.currentJob', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.isBatchActive', () => {
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.hasActive', () => {
            this.updateResumeButtonsState();
            this.saveTranslationState();
        });

        StateManager.subscribe('translation.activeJobs', () => {
            this.saveTranslationState();
        });
    },

    /**
     * Handle translation update from WebSocket
     * @param {Object} data - Translation update data
     */
    handleTranslationUpdate(data) {
        const currentJob = StateManager.getState('translation.currentJob');

        if (!currentJob || data.translation_id !== currentJob.translationId) {
            if (data.translation_id && !currentJob) {
                if (data.status === 'completed' || data.status === 'error' || data.status === 'interrupted' || data.status === 'rate_limited') {
                    this.resetUIToIdle();
                }
            }
            return;
        }

        const currentFile = currentJob.fileRef;

        this.startLiveProgressPolling();
        this._rememberLiveActivity(data);

        if (data.log) {
            MessageLogger.addLog(`[${currentFile.name}] ${data.log}`);
        }

        // Progress is now calculated from stats in ProgressManager.update()
        // No need to call updateProgress() separately
        let displayedStats = null;
        if (data.stats) {
            const stats = this.applyManualRepairStats(currentJob.translationId, data.stats, data.manual_repair === true);
            displayedStats = stats;
            this.updateStats(currentFile.fileType, stats);
        }

        if (data.last_translation) {
            MessageLogger.updateTranslationPreview(data.last_translation);
        } else if (data.log_entry
            && (data.log_entry.type === 'llm_response' || data.log_entry.type === 'refinement_response')
            && data.log_entry.data && data.log_entry.data.response_preview) {
            MessageLogger.updateTranslationPreview(data.log_entry.data.response_preview);
        }

        if (data.status === 'completed') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_completed_msg', { name: currentFile.name }),
                'success',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'interrupted') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_interrupted_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'rate_limited') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_rate_limited_msg', { name: currentFile.name }),
                'info',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'provider_wait') {
            DomHelpers.show('progressSection');
            DomHelpers.show('statsGrid');
            DomHelpers.show('interruptBtn');
            this.updateTranslationTitle(currentFile);
            this.updateFileStatusInList(
                currentFile.name,
                t('translation:translation_provider_waiting_short')
            );
            const waitMessage = data.log || t(
                'translation:translation_provider_waiting_msg',
                { name: currentFile.name }
            );
            ProgressManager.updateLiveStatus({
                live_status: waitMessage,
                live_status_kind: 'scheduled_pause',
            });
            MessageLogger.showMessage(waitMessage, 'info');
            this.updateActiveTranslationsState();
        } else if (data.status === 'pricing_wait') {
            DomHelpers.show('progressSection');
            DomHelpers.show('statsGrid');
            DomHelpers.show('interruptBtn');
            this.updateTranslationTitle(currentFile);
            this.updateFileStatusInList(
                currentFile.name,
                t('common:deepseek_pricing_title')
            );
            const resumeAt = data.resume_at_local || data.stats?.pricing_resume_at_local || '';
            ProgressManager.updateLiveStatus({
                live_status: t('common:deepseek_pricing_job_waiting', {
                    name: currentFile.name,
                    time: resumeAt,
                }),
                live_status_kind: 'scheduled_pause',
            });
            MessageLogger.showMessage(
                t('common:deepseek_pricing_job_waiting', {
                    name: currentFile.name,
                    time: resumeAt,
                }),
                'info'
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'error') {
            MessageLogger.resetProgressTracking();
            this.finishCurrentFileTranslation(
                t('translation:translation_error_msg', { name: currentFile.name, error: data.error || t('translation:translation_unknown_error') }),
                'error',
                data
            );
            this.updateActiveTranslationsState();
        } else if (data.status === 'running') {
            MessageLogger.resetProgressTracking();
            DomHelpers.show('progressSection');
            DomHelpers.show('statsGrid');
            this.updateTranslationTitle(currentFile);
            if (displayedStats) {
                this.updateStats(currentFile.fileType, displayedStats);
            }
            this.resetOpenRouterCostDisplay();

            MessageLogger.showMessage(t('translation:translation_in_progress', { name: currentFile.name }), 'info');
            this.updateFileStatusInList(currentFile.name, 'Processing');
        }
    },

    /**
     * Update translation title with file icon/thumbnail and name
     * @param {Object} file - File object
     */
    updateTranslationTitle(file) {
        renderTranslationTitle(file);
    },

    /**
     * Update statistics display
     * @param {string} fileType - File type (txt, epub, srt)
     * @param {Object} stats - Statistics object
     */
    updateStats(fileType, stats) {
        ProgressManager.update({ stats: stats }, fileType);
        this.updateOpenRouterCost(stats);
    },

    startLiveProgressPolling() {
        if (this._liveProgressPollTimer) return;
        this._pollLiveProgress();
        this._liveProgressPollTimer = setInterval(
            () => this._pollLiveProgress(),
            this._liveProgressPollIntervalMs,
        );
    },

    stopLiveProgressPolling() {
        if (this._liveProgressPollTimer) {
            clearInterval(this._liveProgressPollTimer);
            this._liveProgressPollTimer = null;
        }
        this._liveProgressPollInFlight = false;
    },

    async _pollLiveProgress() {
        if (this._liveProgressPollInFlight) return;

        const currentJob = StateManager.getState('translation.currentJob');
        if (!currentJob || !currentJob.translationId) {
            this.stopLiveProgressPolling();
            return;
        }

        this._liveProgressPollInFlight = true;
        try {
            const response = await ApiClient.getActiveTranslations();
            const jobs = response.translations || [];
            const activeJobs = jobs.filter(job => ACTIVE_STATUSES.has(job.status));
            const wasActive = StateManager.getState('translation.hasActive');
            const hasActive = activeJobs.length > 0;
            StateManager.setState('translation.hasActive', hasActive);
            StateManager.setState('translation.activeJobs', activeJobs);
            if (wasActive !== hasActive) {
                this.updateResumeButtonsState();
            }

            const job = jobs.find(item => item.translation_id === currentJob.translationId);
            if (!job || !ACTIVE_STATUSES.has(job.status)) {
                this.stopLiveProgressPolling();
                return;
            }

            this._applyPolledJobProgress(job, currentJob.fileRef);
        } catch (error) {
            ProgressManager.updateLiveStatus({
                live_status: 'Progress synchronization is pending; the process may still be active.',
                live_status_kind: 'stale',
            });
        } finally {
            this._liveProgressPollInFlight = false;
        }
    },

    _applyPolledJobProgress(job, fileRef) {
        if (!job || !fileRef) return;
        const fileType = fileRef.fileType || fileRef.type || job.file_type || 'txt';
        const stats = this.applyManualRepairStats(
            job.translation_id,
            this._statsFromActiveJob(job),
            false,
        );
        this.updateStats(fileType, stats);

        if (job.last_translation) {
            MessageLogger.updateTranslationPreview(job.last_translation);
        }
    },

    _statsFromActiveJob(job) {
        const completed = Number(job.completed_chunks || 0);
        const previousCompleted = this._liveProgressLastCompleted.get(job.translation_id);
        const now = Date.now();
        if (previousCompleted !== completed) {
            this._liveProgressLastCompleted.set(job.translation_id, completed);
            this._liveProgressLastChangeAt.set(job.translation_id, now);
        } else if (!this._liveProgressLastChangeAt.has(job.translation_id)) {
            this._liveProgressLastChangeAt.set(job.translation_id, now);
        }

        const transformMeta = inferTransformMetadata(job);
        return {
            total_chunks: job.total_chunks || 0,
            completed_chunks: completed,
            failed_chunks: job.failed_chunks || 0,
            elapsed_time: job.elapsed_time ?? job.elapsed_seconds,
            elapsed_seconds: job.elapsed_seconds,
            eta_seconds: job.eta_seconds,
            progress_percent: job.progress_percent,
            percent: job.percent,
            phase: job.phase,
            current_phase: job.current_phase,
            enable_refinement: job.enable_refinement || false,
            refinement_enabled: job.refinement_enabled || false,
            refine_only: job.refine_only || false,
            text_transform_mode: transformMeta.mode,
            text_transform_label: transformMeta.label,
            operation: transformMeta.isTransform ? 'transform' : undefined,
            live_status: job.live_status || this._deriveLiveStatus(job),
            live_status_kind: job.live_status_kind || this._deriveLiveStatusKind(job),
            live_activity_event: job.live_activity_event,
            last_activity_at: job.last_activity_at,
            failure_recovery_cycle: job.failure_recovery_cycle || 0,
            failure_recovery_stuck_count: job.failure_recovery_stuck_count || 0,
        };
    },

    _deriveLiveStatus(job) {
        const completed = Number(job.completed_chunks || 0);
        const total = Number(job.total_chunks || 0);
        const nextChunk = total > 0 ? Math.min(completed + 1, total) : null;
        const activity = this._recentLiveActivity(job.translation_id);
        const chunkText = total > 0 ? `chunk ${nextChunk}/${total}` : 'current chunk';

        if (job.status === 'queued') {
            return `Queued · ${chunkText}`;
        }
        if (job.status === 'pricing_wait') {
            return t('common:deepseek_pricing_job_waiting', {
                name: job.output_filename || 'DeepSeek',
                time: job.resume_at_local || '',
            });
        }
        if (job.status === 'provider_wait') {
            return t('translation:translation_provider_waiting_msg', {
                name: job.output_filename || 'LLM',
            });
        }

        const lastChange = this._liveProgressLastChangeAt.get(job.translation_id) || Date.now();
        const secondsSinceChange = Math.floor((Date.now() - lastChange) / 1000);
        const syncText = secondsSinceChange > 20
            ? `no new chunk for ${secondsSinceChange}s`
            : 'synced now';

        if (activity) {
            return `${activity.label} · ${chunkText} · ${syncText}`;
        }
        if (secondsSinceChange > 20) {
            return `Working on ${chunkText}; audit or repair may take a while · ${syncText}`;
        }
        return `Working on ${chunkText} · ${syncText}`;
    },

    _deriveLiveStatusKind(job) {
        const lastChange = this._liveProgressLastChangeAt.get(job.translation_id);
        if (!lastChange) return 'normal';
        return Date.now() - lastChange > 90000 ? 'stale' : 'normal';
    },

    _rememberLiveActivity(data) {
        if (!data || !data.translation_id) return;
        const label = this._activityLabelFromUpdate(data);
        if (!label) return;
        this._liveActivityByTranslation.set(data.translation_id, {
            label,
            timestamp: Date.now(),
        });
    },

    _recentLiveActivity(translationId) {
        const activity = this._liveActivityByTranslation.get(translationId);
        if (!activity) return null;
        if (Date.now() - activity.timestamp > 120000) return null;
        return activity;
    },

    _activityLabelFromUpdate(data) {
        const entryType = String(data.log_entry?.type || '');
        const text = `${entryType} ${data.log || ''}`.toLowerCase();

        if (text.includes('fidelity supervisor')) return 'Auditing fidelity';
        if (text.includes('profile repair')) return 'Repairing editorial profile';
        if (text.includes('profile audit')) return 'Auditing editorial profile';
        if (text.includes('post-run repair') || text.includes('postprocess_repair')) {
            return 'Reprocessing flagged chunks';
        }
        if (text.includes('glossary')) return 'Applying glossary';
        if (text.includes('modernize') || text.includes('refinement_request')) {
            return 'Modernizing chunk';
        }
        if (text.includes('refinement_response')) return 'Reviewing editorial response';
        if (text.includes('llm_request') || text.includes('sending request to llm')) {
            return 'Querying model';
        }
        if (text.includes('llm_response') || text.includes('response received')) {
            return 'Processing response';
        }
        if (text.includes('checkpoint')) return 'Saving progress';
        if (text.includes('rate limited')) return 'Waiting for provider';
        if (data.status === 'provider_wait') return 'Waiting for provider';
        if (data.status === 'running') return 'Processing';
        return '';
    },

    /**
     * Keep manually repaired checkpoint counts visible while the active worker
     * still emits its older in-memory failed counter.
     */
    applyManualRepairStats(translationId, stats, isManualRepair = false) {
        const now = Date.now();
        if (isManualRepair) {
            this._manualRepairStatsByTranslation.set(translationId, {
                failedChunks: Number(stats.checkpoint_failed_chunks ?? stats.failed_chunks ?? 0),
                expiresAt: now + this._manualRepairStatsTtlMs,
            });
            return stats;
        }

        const repair = this._manualRepairStatsByTranslation.get(translationId);
        if (!repair) return stats;
        if (repair.expiresAt < now) {
            this._manualRepairStatsByTranslation.delete(translationId);
            return stats;
        }

        const incomingFailed = Number(stats.failed_chunks ?? 0);
        if (incomingFailed <= repair.failedChunks) return stats;
        return {
            ...stats,
            failed_chunks: repair.failedChunks,
            checkpoint_failed_chunks: repair.failedChunks,
            manual_repair_visible: true,
        };
    },

    /**
     * Update OpenRouter cost display
     * @param {Object} stats - Statistics object containing cost data
     */
    updateOpenRouterCost(stats) {
        const costGrid = DomHelpers.getElement('openrouterCostGrid');
        if (!costGrid) return;

        const cost = stats.openrouter_cost || 0;
        const promptTokens = stats.openrouter_prompt_tokens || 0;
        const completionTokens = stats.openrouter_completion_tokens || 0;
        const totalTokens = promptTokens + completionTokens;

        // Show cost grid if there's any cost or token data
        if (cost > 0 || totalTokens > 0) {
            DomHelpers.show('openrouterCostGrid');
            DomHelpers.setText('openrouterCost', '$' + cost.toFixed(4));
            DomHelpers.setText('openrouterTokens', totalTokens.toLocaleString());
        }
    },

    /**
     * Reset OpenRouter cost display for a new translation
     */
    resetOpenRouterCostDisplay() {
        DomHelpers.hide('openrouterCostGrid');
        DomHelpers.setText('openrouterCost', '$0.0000');
        DomHelpers.setText('openrouterTokens', '0');
    },

    /**
     * Update file status in UI list
     * @param {string} fileName - File name
     * @param {string} newStatus - New status text
     * @param {string} [translationId] - Translation ID
     */
    updateFileStatusInList(fileName, newStatus, translationId = null) {
        const fileListItem = DomHelpers.getOne(`#fileListContainer li[data-filename="${fileName}"] .file-status`);
        if (fileListItem) {
            DomHelpers.setText(fileListItem, `(${newStatus})`);
        }

        // Update in state
        const filesToProcess = StateManager.getState('files.toProcess');
        const fileObj = filesToProcess.find(f => f.name === fileName);
        if (fileObj) {
            fileObj.status = newStatus;
            if (translationId) {
                fileObj.translationId = translationId;
            }
            StateManager.setState('files.toProcess', filesToProcess);
            // Persist to localStorage
            FileUpload.notifyFileListChanged();
        }
    },

    /**
     * Finish current file translation and update UI
     * @param {string} statusMessage - Status message to display
     * @param {string} messageType - Message type (success, error, info)
     * @param {Object} resultData - Translation result data
     */
    finishCurrentFileTranslation(statusMessage, messageType, resultData) {
        const currentJob = StateManager.getState('translation.currentJob');
        if (!currentJob) return;

        const currentFile = currentJob.fileRef;
        currentFile.status = resultData.status || 'unknown_error';
        currentFile.result = resultData.result;

        MessageLogger.showMessage(statusMessage, messageType);
        this.updateFileStatusInList(
            currentFile.name,
            resultData.status === 'completed' ? 'Completed' :
            resultData.status === 'interrupted' ? 'Interrupted' :
            resultData.status === 'rate_limited' ? 'Rate Limited' : 'Error'
        );

        if (resultData.status === 'completed') {
            this.renderCompletionCard(currentFile, resultData);
        }

        StateManager.setState('translation.currentJob', null);

        if (resultData.status === 'completed') {
            this.processNextFileInQueue();
        } else if (resultData.status === 'interrupted') {
            MessageLogger.addLog(t('translation:batch_stopped_user_log'));
            this.resetUIToIdle();
        } else if (resultData.status === 'rate_limited') {
            MessageLogger.addLog(t('translation:batch_paused_log'));
            this.resetUIToIdle();
        } else {
            this.processNextFileInQueue();
        }
    },

    /**
     * Render a persistent success card for a completed file, with quick actions
     * to locate it on disk.
     * @param {Object} file - The file that just finished
     * @param {Object} resultData - Final payload from the server (output_filename, output_dir)
     */
    renderCompletionCard(file, resultData) {
        const container = DomHelpers.getElement('completionCardsContainer');
        if (!container) return;

        const card = document.createElement('div');
        card.className = 'completion-card';
        this._populateCompletionCard(card, file, resultData);
        container.appendChild(card);
        this._ensureCompletionCardsLocaleListener();

        DomHelpers.hide('progressSection');
    },

    /**
     * Fill (or rebuild) an existing completion card with localized content.
     * Pulled out of `renderCompletionCard` so the same DOM tree can be
     * re-rendered on `localeChanged` without dropping the card from the page.
     *
     * Stashes the source payload on the element itself so the locale listener
     * can rebuild without coordinating extra storage.
     */
    _populateCompletionCard(card, file, resultData) {
        card._tblPayload = { file, resultData };

        const outputFilename = resultData.output_filename || file.outputFilename || file.name;
        const safeFilename = DomHelpers.escapeHtml(outputFilename);
        const statsHtml = this._buildCompletionStatsHtml(file, resultData);
        const dismissLabel = t('translation:completion_card_dismiss');

        card.innerHTML = '';

        const topRow = document.createElement('div');
        topRow.className = 'completion-card__top';
        topRow.appendChild(this._buildCompletionThumb(file));

        const main = document.createElement('div');
        main.className = 'completion-card__main';
        main.innerHTML = `
            <div class="completion-card__header">
                <h3 class="completion-card__title">
                    <span class="material-symbols-outlined">check_circle</span>
                    <span>${t('translation:translation_completed_card_title')}${statsHtml}</span>
                </h3>
                <button type="button" class="completion-card__close" title="${dismissLabel}" aria-label="${dismissLabel}">
                    <span class="material-symbols-outlined">close</span>
                </button>
            </div>
            <div class="completion-card__filename" title="${safeFilename}">${safeFilename}</div>
        `;
        topRow.appendChild(main);
        card.appendChild(topRow);

        const warningBlock = this._buildCompletionWarningBlock(file, resultData);
        if (warningBlock) {
            card.appendChild(warningBlock);
        }

        const actionsGroup = FileActions.createActionGroup({
            actions: ['download', 'open', 'reveal', 'files-tab'],
            filename: outputFilename,
            variant: 'labeled'
        });
        actionsGroup.classList.add('completion-card__actions');
        card.appendChild(actionsGroup);

        card.querySelector('.completion-card__close').addEventListener('click', () => card.remove());
    },

    /**
     * Re-render every visible completion card whenever the user switches
     * locale, so the dynamically interpolated strings (title, stat badges,
     * warning block, action labels) stay in sync with the rest of the UI.
     * Bound once, lazily, the first time a card is rendered.
     */
    _ensureCompletionCardsLocaleListener() {
        if (this._completionLocaleListenerBound) return;
        this._completionLocaleListenerBound = true;
        window.addEventListener('localeChanged', () => {
            const container = DomHelpers.getElement('completionCardsContainer');
            if (!container) return;
            container.querySelectorAll('.completion-card').forEach((card) => {
                if (card._tblPayload) {
                    this._populateCompletionCard(card, card._tblPayload.file, card._tblPayload.resultData);
                }
            });
        });
    },

    /**
     * Build the thumbnail element for the completion card.
     * Uses the book cover for EPUBs (with SVG fallback), generic icon otherwise.
     * @param {Object} file - File object (fileType, thumbnail)
     * @returns {HTMLElement} Thumb wrapper element
     */
    _buildCompletionThumb(file) {
        const wrap = document.createElement('div');
        wrap.className = 'completion-card__thumb';

        if (file.fileType === 'epub' && file.thumbnail) {
            const img = document.createElement('img');
            img.src = `/api/thumbnails/${encodeURIComponent(file.thumbnail)}`;
            img.alt = 'Cover';
            img.onerror = () => {
                wrap.innerHTML = createGenericEPUBIcon();
            };
            wrap.appendChild(img);
        } else {
            wrap.innerHTML = getFileIcon(file.fileType);
        }

        return wrap;
    },

    /**
     * Build the stats block HTML for the completion card.
     * @param {Object} file - File object (for fileType)
     * @param {Object} resultData - Final payload (contains stats)
     * @returns {string} HTML for the stats block (empty string if no stats)
     */
    _buildCompletionStatsHtml(file, resultData) {
        const stats = resultData.stats || {};

        const failed = stats.failed_chunks || 0;
        const elapsed = stats.elapsed_time;
        const fallbacks = (file && file.fileType === 'srt')
            ? 0
            : (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
        const placeholderErrors = (file && file.fileType === 'srt')
            ? 0
            : (stats.placeholder_errors || 0);

        const cost = stats.openrouter_cost || 0;
        const promptTokens = stats.openrouter_prompt_tokens || 0;
        const completionTokens = stats.openrouter_completion_tokens || 0;
        const totalTokens = promptTokens + completionTokens;

        const items = [];

        if (typeof elapsed === 'number' && elapsed > 0) {
            items.push(formatElapsedTime(elapsed));
        }

        if (failed > 0) {
            items.push(`<span class="completion-card__stat--error">${t('translation:completion_failed_chunks', { count: failed })}</span>`);
        }

        if (fallbacks > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_fallback_chunks', { count: fallbacks })}</span>`);
        }

        if (placeholderErrors > 0) {
            items.push(`<span class="completion-card__stat--warn">${t('translation:completion_placeholder_errors', { count: placeholderErrors })}</span>`);
        }

        if (cost > 0 || totalTokens > 0) {
            items.push(`$${cost.toFixed(4)} · ${totalTokens.toLocaleString()} tokens`);
        }

        if (items.length === 0) return '';

        return `<span class="completion-card__stats"> - ${items.join(' · ')}</span>`;
    },

    /**
     * Build the warning block surfaced beneath the title when the run produced
     * fallbacks, placeholder errors, or failed chunks. Mirrors the live
     * recommendation panel from progress-manager so the post-translation
     * advice stays in sync with what was shown during the run.
     *
     * @param {Object} file - File object (used to gate by file type)
     * @param {Object} resultData - Final payload (contains stats)
     * @returns {HTMLElement|null} Warning block element, or null when there is
     *   nothing worth surfacing.
     */
    _buildCompletionWarningBlock(file, resultData) {
        const stats = resultData.stats || {};
        if (file && file.fileType === 'srt') {
            return null;
        }

        const fallbacks = (stats.token_alignment_used || 0) + (stats.fallback_used || 0);
        const placeholderErrors = stats.placeholder_errors || 0;
        const failed = stats.failed_chunks || 0;
        const tokenAlignment = stats.token_alignment_used || 0;
        const untranslated = stats.fallback_used || 0;

        if (fallbacks === 0 && placeholderErrors === 0 && failed === 0) {
            return null;
        }

        const block = document.createElement('div');
        block.className = 'completion-card__warning';

        const heading = document.createElement('div');
        heading.className = 'completion-card__warning-heading';
        const icon = document.createElement('span');
        icon.className = 'material-symbols-outlined';
        icon.textContent = 'warning';
        heading.appendChild(icon);
        const headingText = document.createElement('span');
        headingText.textContent = t('translation:completion_warning_heading');
        heading.appendChild(headingText);
        block.appendChild(heading);

        const breakdownItems = [];
        if (tokenAlignment > 0) {
            breakdownItems.push(t('translation:completion_warning_token_alignment', { count: tokenAlignment }));
        }
        if (untranslated > 0) {
            breakdownItems.push(t('translation:completion_warning_untranslated', { count: untranslated }));
        }
        if (placeholderErrors > 0) {
            breakdownItems.push(t('translation:completion_warning_placeholder_errors', { count: placeholderErrors }));
        }
        if (failed > 0) {
            breakdownItems.push(t('translation:completion_warning_failed', { count: failed }));
        }
        if (breakdownItems.length > 0) {
            const breakdown = document.createElement('div');
            breakdown.className = 'completion-card__warning-breakdown';
            breakdown.textContent = breakdownItems.join(' · ');
            block.appendChild(breakdown);
        }

        // Only renew the rate-based recommendations when there were actual
        // fallbacks or placeholder issues — a run with only `failed_chunks`
        // (e.g. provider errors) is not really a "tune the LLM" situation.
        if (fallbacks > 0 || placeholderErrors > 0) {
            const recommendations = document.createElement('div');
            recommendations.className = 'completion-card__warning-recommendations';
            buildRecommendationContent(
                recommendations,
                deriveRateContext(stats),
                'translation:completion_warning_intro',
            );
            block.appendChild(recommendations);
        }

        return block;
    },

    /**
     * Remove all completion cards. Currently unused — cards are dismissed
     * individually by the user via the card's close button.
     */
    clearCompletionCards() {
        const container = DomHelpers.getElement('completionCardsContainer');
        if (container) container.innerHTML = '';
    },

    /**
     * Process next file in queue (delegates to batch-controller when available)
     */
    processNextFileInQueue() {
        // Trigger event for batch controller to handle
        window.dispatchEvent(new CustomEvent('processNextFile'));
    },

    /**
     * Check and update active translations state
     */
    async updateActiveTranslationsState() {
        try {
            const response = await ApiClient.getActiveTranslations();
            const activeJobs = (response.translations || []).filter(
                job => ACTIVE_STATUSES.has(job.status)
            );

            const wasActive = StateManager.getState('translation.hasActive');
            const hasActive = activeJobs.length > 0;

            StateManager.setState('translation.hasActive', hasActive);
            StateManager.setState('translation.activeJobs', activeJobs);
            if (hasActive) {
                this.startLiveProgressPolling();
            } else {
                this.stopLiveProgressPolling();
            }

            // If state changed, update UI
            if (wasActive !== hasActive) {
                this.updateResumeButtonsState();
            }

            if (!hasActive) {
                await this._clearStaleCurrentJobIfServerDisagrees();
            }

            return { hasActive, activeJobs };
        } catch {
            return {
                hasActive: StateManager.getState('translation.hasActive'),
                activeJobs: StateManager.getState('translation.activeJobs')
            };
        }
    },

    async _clearStaleCurrentJobIfServerDisagrees() {
        if (this._resettingToIdle) return;

        const isBatchActive = StateManager.getState('translation.isBatchActive');
        const currentJob = StateManager.getState('translation.currentJob');
        if (!isBatchActive || !currentJob || !currentJob.translationId) return;

        try {
            const serverState = await ApiClient.getTranslationStatus(currentJob.translationId);
            if (TERMINAL_STATUSES.has(serverState.status)
                || !ACTIVE_STATUSES.has(serverState.status)) {
                MessageLogger.addLog(t('translation:session_sync_log', { status: serverState.status || 'missing' }));
                this.resetUIToIdle();
            }
        } catch (error) {
            if (error.status === 404 || (error.message && error.message.includes('404'))) {
                MessageLogger.addLog(t('translation:session_job_missing_log'));
                this.resetUIToIdle();
            }
        }
    },

    /**
     * Update the state of all resume buttons based on active translations
     */
    updateResumeButtonsState() {
        const resumeButtons = DomHelpers.getElements('button[onclick^="resumeJob"]');
        const hasActive = StateManager.getState('translation.hasActive');

        resumeButtons.forEach(button => {
            if (hasActive) {
                button.disabled = true;
                button.style.opacity = '0.5';
                button.style.cursor = 'not-allowed';
                button.title = t('translation:cannot_resume_in_progress_title');
            } else {
                button.disabled = false;
                button.style.opacity = '1';
                button.style.cursor = 'pointer';
                button.title = t('translation:resume_btn_title');
            }
        });

        // Update warning banner
        this.updateResumableJobsWarningBanner();
    },

    /**
     * Update or create the warning banner in resumable jobs section
     */
    updateResumableJobsWarningBanner() {
        const listContainer = DomHelpers.getElement('resumableJobsList');
        if (!listContainer) return;

        const existingBanner = listContainer.querySelector('.active-translation-warning');
        const hasActive = StateManager.getState('translation.hasActive');
        const activeJobs = StateManager.getState('translation.activeJobs');

        if (hasActive) {
            const activeNames = activeJobs.map(job => job.output_filename || t('translation:job_card_unknown')).join(', ');
            const bannerHtml = `
                <div class="active-translation-warning" style="background: #fef3c7; border: 1px solid #f59e0b; padding: 12px; margin-bottom: 15px; border-radius: 6px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 20px;">⚠️</span>
                        <div style="flex: 1;">
                            <strong style="color: #92400e;">${t('translation:active_translation_warning_title')}</strong>
                            <p style="margin: 5px 0 0 0; font-size: 13px; color: #78350f;">
                                ${t('translation:active_translation_warning_desc', { names: DomHelpers.escapeHtml(activeNames) })}
                            </p>
                        </div>
                    </div>
                </div>
            `;

            if (existingBanner) {
                existingBanner.outerHTML = bannerHtml;
            } else {
                // Insert at the beginning of the container
                listContainer.insertAdjacentHTML('afterbegin', bannerHtml);
            }
        } else if (existingBanner) {
            // Remove banner if no active translations
            existingBanner.remove();
        }
    },

    resetUIToIdle() {
        if (this._resettingToIdle) return;
        this._resettingToIdle = true;
        this.stopLiveProgressPolling();
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.currentJob', null);

        this.clearTranslationState();

        const filesToProcess = StateManager.getState('files.toProcess') || [];
        let requeuedFiles = false;
        for (const file of filesToProcess) {
            if (['Processing', 'Preparing...', 'Submitted'].includes(file.status)) {
                file.status = 'Queued';
                file.translationId = null;
                requeuedFiles = true;
            }
        }
        if (requeuedFiles) {
            StateManager.setState('files.toProcess', filesToProcess);
            FileUpload.notifyFileListChanged();
        }

        DomHelpers.hide('interruptBtn');
        DomHelpers.setDisabled('interruptBtn', false);
        DomHelpers.setText('interruptBtn', t('translation:interrupt_batch_with_icon'));

        const currentFilesToProcess = StateManager.getState('files.toProcess') || [];
        DomHelpers.setDisabled('translateBtn', currentFilesToProcess.length === 0 || !StatusManager.isConnected());
        DomHelpers.setText('translateBtn', t('translation:start_batch_with_icon'));
        DomHelpers.hide('progressSection');

        this.updateActiveTranslationsState();

        if (window.loadResumableJobs) {
            window.loadResumableJobs();
        }
        this._resettingToIdle = false;
    }
};
