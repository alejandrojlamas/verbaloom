/**
 * Batch Controller - Batch translation orchestration
 *
 * Manages batch translation queue processing, configuration validation,
 * and sequential file translation.
 */

import { StateManager } from '../core/state-manager.js';
import { ApiClient } from '../core/api-client.js';
import { MessageLogger } from '../ui/message-logger.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { Validators } from '../utils/validators.js';
import { ApiKeyUtils } from '../utils/api-key-utils.js';
import { StatusManager } from '../utils/status-manager.js';
import { ProgressManager } from './progress-manager.js?v=20260705-transform-flow';
import { renderTranslationTitle } from './progress-title.js?v=20260705-transform-flow';
import { FileUpload, generateOutputFilename, normalizeOutputFormat, resolveOutputExtension } from '../files/file-upload.js';
import { TranslationTracker } from './translation-tracker.js?v=20260717-live-recovery';
import { t } from '../i18n/i18n.js';

async function waitForTranslationTracker(timeoutMs = 8000) {
    if (TranslationTracker.isInitialized && TranslationTracker.isInitialized()) {
        return true;
    }

    const timeout = new Promise(resolve => setTimeout(resolve, timeoutMs, false));
    const initialization = Promise.resolve()
        .then(() => TranslationTracker.initialize && TranslationTracker.initialize())
        .then(() => true)
        .catch(() => false);

    const initialized = await Promise.race([initialization, timeout]);
    return initialized && TranslationTracker.isInitialized && TranslationTracker.isInitialized();
}

/**
 * Validation helper for early failures
 * @param {string} message - Error message
 */
function earlyValidationFail(message) {
    MessageLogger.showMessage(message, 'error');
    MessageLogger.addLog(t('translation:validation_failed_log', { message }));
    return false;
}

function isTransformFile(file) {
    return Boolean(file?.transformMode || file?.operation === 'transform');
}

function transformLabelForFile(file) {
    return file?.transformLabel || file?.transformMode || 'Transform';
}

const PROVIDER_ENDPOINT_DEFAULTS = {
    deepseek: 'https://api.deepseek.com/chat/completions',
    gemini: '',
    mistral: 'https://api.mistral.ai/v1/chat/completions',
    nim: 'https://integrate.api.nvidia.com/v1/chat/completions',
    openrouter: '',
    poe: 'https://api.poe.com/v1/chat/completions'
};

function effectiveEndpointForProvider(provider) {
    if (provider === 'ollama') {
        return (DomHelpers.getValue('apiEndpoint') || '').trim();
    }
    if (provider === 'openai') {
        return (DomHelpers.getValue('openaiEndpoint') || '').trim()
            || 'https://api.openai.com/v1/chat/completions';
    }
    return PROVIDER_ENDPOINT_DEFAULTS[provider] || '';
}

function selectedProfileStrength() {
    const value = (DomHelpers.getValue('profileStrengthSelect') || 'balanced').trim().toLowerCase();
    return ['light', 'balanced', 'strict'].includes(value) ? value : 'balanced';
}


/**
 * Get translation configuration from form
 * @param {Object} file - File to translate
 * @returns {Object} Translation configuration
 */
function getTranslationConfig(file) {
    // Use languages stored in the file object (captured when added to queue)
    // This ensures each file can have different source/target languages in batch
    const sourceLanguageVal = file.sourceLanguage;
    const targetLanguageVal = file.targetLanguage;

    const provider = DomHelpers.getValue('llmProvider');
    const currentModel = DomHelpers.getValue('model') || '';

    // Regenerate output filename at translation time so placeholders like
    // {model}, {date}, {datetime} reflect the current run. The value stored
    // on the file object was computed at upload time and may be stale,
    // especially when the same file is re-translated with a different model.
    const outputPattern = DomHelpers.getValue('outputFilenamePattern')
        || '{originalName} ({targetLang}).{ext}';
    const outputFormat = normalizeOutputFormat(file.outputFormat || DomHelpers.getValue('outputFormat'));
    const resolvedOutputFilename = generateOutputFilename(
        { name: file.name },
        outputPattern,
        {
            sourceLang: sourceLanguageVal,
            targetLang: file.outputLabel || file.transformLabel || targetLanguageVal,
            model: currentModel,
            ext: resolveOutputExtension(file, file.fileType, outputFormat)
        }
    );

    const operation = file.operation || 'translate';
    const refineAfter = operation === 'translate';

    const promptOptions = {
        preserve_technical_content: true,
        text_cleanup: DomHelpers.getElement('textCleanup')?.checked || false,
        refine: refineAfter,
        strict_stage_contract: refineAfter,
        review_entire_book: refineAfter,
        audit_entire_book: refineAfter,
        permit_silent_source_fallback: false,
        permit_unreviewed_segments: false,
        permit_unaudited_segments: false,
        plain_text_mode: DomHelpers.getElement('plainTextMode')?.checked || false,
        literary_continuity: DomHelpers.getElement('literaryContinuity')?.checked || false,
        text_type: DomHelpers.getValue('textTypeMode') || 'auto',
        continuity_max_prompt_chars: 1800,
        custom_instruction_file: DomHelpers.getValue('customInstructionSelect') || ''
    };

    const readableCharacters = Number(file.readableCharacters || 0);
    if (Number.isFinite(readableCharacters) && readableCharacters > 0) {
        promptOptions._input_readable_characters = readableCharacters;
    }

    const glossarySelection = (
        file.transformMode
            ? (DomHelpers.getValue('transformGlossarySelect') || DomHelpers.getValue('translateGlossarySelect') || DomHelpers.getValue('glossarySelect') || '')
            : (DomHelpers.getValue('glossarySelect') || DomHelpers.getValue('translateGlossarySelect') || '')
    ).trim();
    const glossaryProfilePrefix = 'profile:';
    const selectedProfileFromGlossary = glossarySelection.startsWith(glossaryProfilePrefix)
        ? glossarySelection.slice(glossaryProfilePrefix.length)
        : '';
    const selectedBookProfile = (
        file.profileId
        || selectedProfileFromGlossary
        || (file.transformMode ? DomHelpers.getValue('transformProfile') : '')
        || DomHelpers.getValue('bookProfileSelect')
        || ''
    ).trim();

    if (file.transformMode) {
        promptOptions.text_transform_mode = file.transformMode;
        promptOptions.text_transform_label = file.transformLabel || '';
        promptOptions.refinement_instructions = file.transformDescription || '';
        if (selectedBookProfile) {
            promptOptions.editorial_mode = 'book_profile';
            promptOptions.profile_id = selectedBookProfile;
            promptOptions.profile_strength = selectedProfileStrength();
            if (/^spanish$/i.test(targetLanguageVal || '')) {
                promptOptions.target_locale = 'es-MX';
            }
            promptOptions.modernization_strength = file.transformMode === 'modernize' ? 'high' : 'medium';
            promptOptions.preserve_author_voice = true;
            promptOptions.use_profile_glossary = true;
            promptOptions.allow_common_glossary = true;
            promptOptions.allow_cross_profile_glossary = false;
            promptOptions.glossary_suggestions_enabled = true;
            promptOptions.auto_approve_glossary_suggestions = false;
            promptOptions.min_glossary_suggestion_confidence = 0.92;
            promptOptions.avoid_hardcoded_editorial_rules = true;
        }
        if (file.transformMode === 'modernize') {
            promptOptions.text_transform_profile = 'faithful_current_spanish';
            promptOptions.preserve_block_structure = true;
            promptOptions.transform_guard = 'strict';
            promptOptions.transform_auditor_model = 'deepseek-v4-flash';
            promptOptions.transform_repair_attempts = 2;
            promptOptions.suppress_attribution_footer = true;
            promptOptions.editorial_quality_guard = true;
            promptOptions.fidelity_supervisor = true;
            promptOptions.fidelity_supervisor_mode = 'alerted';
            promptOptions.fidelity_supervisor_model = 'deepseek-v4-flash';
            promptOptions.transform_fallback = 'best_candidate';
            if (selectedBookProfile) {
                promptOptions.modernization_strength = 'high';
                promptOptions.preserve_archaisms = false;
                promptOptions.preserve_iconic_formulas = true;
                promptOptions.audit_dimensions = 'editorial_full';
                promptOptions.repair_until_pass = true;
                promptOptions.min_dimension_score = 8.5;
                promptOptions.max_repair_rounds = 2;
                if (promptOptions.profile_strength === 'strict') {
                    promptOptions.profile_audit_enabled = true;
                    promptOptions.profile_audit_model = 'deepseek-v4-pro';
                    promptOptions.profile_repair_model = 'deepseek-v4-pro';
                    promptOptions.transform_auditor_model = 'deepseek-v4-pro';
                    promptOptions.fidelity_supervisor_mode = 'always';
                    promptOptions.fidelity_supervisor_model = 'deepseek-v4-pro';
                }
            }
        } else {
            promptOptions.editorial_quality_guard = false;
            promptOptions.source_aware_editorial_guard = false;
            promptOptions.fidelity_supervisor_mode = 'local';
        }
    }

    if (!file.transformMode && selectedBookProfile) {
        promptOptions.editorial_mode = 'book_profile';
        promptOptions.profile_id = selectedBookProfile;
        promptOptions.profile_strength = selectedProfileStrength();
        if (/^spanish$/i.test(targetLanguageVal || '')) {
            promptOptions.target_locale = 'es-MX';
        }
        promptOptions.preserve_author_voice = true;
        promptOptions.use_profile_glossary = true;
        promptOptions.allow_common_glossary = true;
        promptOptions.allow_cross_profile_glossary = false;
        promptOptions.glossary_suggestions_enabled = true;
        promptOptions.auto_approve_glossary_suggestions = false;
        promptOptions.min_glossary_suggestion_confidence = 0.92;
        promptOptions.avoid_hardcoded_editorial_rules = true;
        promptOptions.audit_dimensions = 'translation_editorial_full';
        promptOptions.translation_profile_mode = true;
        promptOptions.profile_audit_enabled = false;
        promptOptions.repair_until_pass = true;
        promptOptions.max_repair_rounds = 1;
    }

    const glossaryId = glossarySelection && !glossarySelection.startsWith(glossaryProfilePrefix)
        ? glossarySelection
        : '';
    if (glossaryId) {
        const parsedGlossaryId = parseInt(glossaryId, 10);
        if (Number.isFinite(parsedGlossaryId)) {
            promptOptions.glossary_id = parsedGlossaryId;
        }
    }

    // Get TTS configuration
    const ttsEnabled = DomHelpers.getElement('ttsEnabled')?.checked || false;

    const config = {
        source_language: sourceLanguageVal,
        target_language: targetLanguageVal,
        model: currentModel,
        llm_api_endpoint: effectiveEndpointForProvider(provider),
        llm_provider: provider,
        gemini_api_key: provider === 'gemini' ? ApiKeyUtils.getValue('geminiApiKey') : '',
        openai_api_key: provider === 'openai' ? ApiKeyUtils.getValue('openaiApiKey') : '',
        openrouter_api_key: provider === 'openrouter' ? ApiKeyUtils.getValue('openrouterApiKey') : '',
        mistral_api_key: provider === 'mistral' ? ApiKeyUtils.getValue('mistralApiKey') : '',
        deepseek_api_key: provider === 'deepseek' ? ApiKeyUtils.getValue('deepseekApiKey') : '',
        poe_api_key: provider === 'poe' ? ApiKeyUtils.getValue('poeApiKey') : '',
        nim_api_key: provider === 'nim' ? ApiKeyUtils.getValue('nimApiKey') : '',
        input_filename: file.name,
        output_filename: resolvedOutputFilename,
        output_format: outputFormat,
        file_type: file.fileType,
        operation: file.transformMode ? 'transform' : operation,
        text_transform_mode: file.transformMode || '',
        text_transform_label: file.transformLabel || '',
        prompt_options: promptOptions,
        bilingual_output: DomHelpers.getElement('bilingualMode')?.checked || false,
        refine_only: operation === 'refine',
        refine_after: refineAfter,
        auto_pause_on_rate_limit: !(DomHelpers.getElement('disableAutoPause')?.checked || false),
        tts_enabled: ttsEnabled,
        tts_voice: ttsEnabled ? (DomHelpers.getValue('ttsVoice') || '') : '',
        tts_rate: ttsEnabled ? (DomHelpers.getValue('ttsRate') || '+0%') : '+0%',
        tts_format: ttsEnabled ? (DomHelpers.getValue('ttsFormat') || 'opus') : 'opus',
        tts_bitrate: ttsEnabled ? (DomHelpers.getValue('ttsBitrate') || '64k') : '64k'
    };

    if (file.fileType === 'epub' || file.fileType === 'srt') {
        config.file_path = file.filePath;
    } else {
        if (file.content) {
            config.text = file.content;
        } else {
            config.file_path = file.filePath;
        }
    }

    return config;
}

/**
 * Update file status in the display
 * @param {string} filename - File name
 * @param {string} status - New status
 * @param {string} translationId - Optional translation ID
 */
function updateFileStatusInList(filename, status, translationId = null) {
    const filesToProcess = StateManager.getState('files.toProcess') || [];
    const fileIndex = filesToProcess.findIndex(f => f.name === filename);

    if (fileIndex !== -1) {
        filesToProcess[fileIndex].status = status;
        if (translationId) {
            filesToProcess[fileIndex].translationId = translationId;
        }
        StateManager.setState('files.toProcess', filesToProcess);
        // Persist to localStorage
        FileUpload.notifyFileListChanged();
    }

    // Emit event for UI update
    const event = new CustomEvent('fileStatusChanged', { detail: { filename, status, translationId } });
    window.dispatchEvent(event);
}

function isTerminalFileError(file) {
    return String(file?.status || '').toLowerCase().includes('error');
}

export const BatchController = {
    /**
     * Start batch translation
     */
    async startBatchTranslation() {
        if (!TranslationTracker.isInitialized || !TranslationTracker.isInitialized()) {
            const ready = await waitForTranslationTracker();
            if (!ready) {
                MessageLogger.showMessage(t('translation:system_initializing'), 'warning');
                return;
            }
        }

        const isBatchActive = StateManager.getState('translation.isBatchActive') || false;
        const filesToProcess = StateManager.getState('files.toProcess') || [];

        if (isBatchActive || filesToProcess.length === 0) return;

        let retryableFilesUpdated = false;
        for (const file of filesToProcess) {
            if (!isTerminalFileError(file)) continue;
            file.status = 'Queued';
            file.translationId = null;
            retryableFilesUpdated = true;
        }
        if (retryableFilesUpdated) {
            StateManager.setState('files.toProcess', filesToProcess);
            FileUpload.notifyFileListChanged();
        }

        // Validate configuration
        let sourceLanguageVal = DomHelpers.getValue('sourceLang');
        if (sourceLanguageVal === 'Other') {
            sourceLanguageVal = DomHelpers.getValue('customSourceLang').trim();
            if (!sourceLanguageVal) {
                return earlyValidationFail(t('translation:validation_custom_source'));
            }
        }

        let targetLanguageVal = DomHelpers.getValue('targetLang');
        if (targetLanguageVal === 'Other') {
            targetLanguageVal = DomHelpers.getValue('customTargetLang').trim();
            if (!targetLanguageVal) {
                return earlyValidationFail(t('translation:validation_custom_target'));
            }
        }

        const selectedModel = DomHelpers.getValue('model');
        if (!selectedModel) {
            return earlyValidationFail(t('translation:validation_model'));
        }

        const provider = DomHelpers.getValue('llmProvider');
        if (provider === 'ollama') {
            const ollamaApiEndpoint = DomHelpers.getValue('apiEndpoint').trim();
            if (!ollamaApiEndpoint) {
                return earlyValidationFail(t('translation:validation_ollama_endpoint'));
            }
        }

        let filesUpdated = false;
        for (const file of filesToProcess) {
            if (file.status !== 'Queued') continue;

            if (!file.sourceLanguage || file.sourceLanguage === 'Other') {
                file.sourceLanguage = sourceLanguageVal;
                filesUpdated = true;
            }
            if (!file.targetLanguage || file.targetLanguage === 'Other') {
                file.targetLanguage = targetLanguageVal;
                filesUpdated = true;
            }
        }

        if (filesUpdated) {
            StateManager.setState('files.toProcess', filesToProcess);
        }

        StateManager.setState('translation.isBatchActive', true);

        const queuedFilesCount = filesToProcess.filter(f => f.status === 'Queued').length;

        // Update UI
        const translateBtn = DomHelpers.getElement('translateBtn');
        if (translateBtn) {
            translateBtn.disabled = true;
            translateBtn.innerHTML = t('translation:batch_in_progress');
        }

        const interruptBtn = DomHelpers.getElement('interruptBtn');
        if (interruptBtn) {
            DomHelpers.show('interruptBtn');
            interruptBtn.disabled = false;
        }

        MessageLogger.clearAlerts();
        const queuedTransformCount = filesToProcess.filter(f => f.status === 'Queued' && isTransformFile(f)).length;
        MessageLogger.addLog(t(
            queuedTransformCount === queuedFilesCount
                ? 'translation:transform_batch_started_log'
                : 'translation:batch_started_log',
            { count: queuedFilesCount }
        ));
        MessageLogger.showMessage(t('translation:batch_initiated', { count: queuedFilesCount }), 'info');

        // Start processing queue
        this.processNextFileInQueue();
    },

    /**
     * Process next file in queue
     */
    async processNextFileInQueue() {
        const currentJob = StateManager.getState('translation.currentJob');
        if (currentJob) return;

        const filesToProcess = StateManager.getState('files.toProcess') || [];
        const fileToTranslate = filesToProcess.find(f => f.status === 'Queued');

        if (!fileToTranslate) {
            StateManager.setState('translation.isBatchActive', false);
            StateManager.setState('translation.currentJob', null);

            const translateBtn = DomHelpers.getElement('translateBtn');
            if (translateBtn) {
                translateBtn.disabled = filesToProcess.length === 0 || !StatusManager.isConnected();
                translateBtn.innerHTML = t('translation:start_batch_with_icon');
            }

            DomHelpers.hide('interruptBtn');

            const failedFiles = filesToProcess.filter(isTerminalFileError);
            if (failedFiles.length > 0) {
                MessageLogger.showMessage(t('translation:batch_completed_with_errors', {
                    count: failedFiles.length,
                    defaultValue: `Batch finished with ${failedFiles.length} file error(s).`
                }), 'error');
                MessageLogger.addLog(t('translation:batch_completed_with_errors_log', {
                    count: failedFiles.length,
                    defaultValue: `Batch finished with ${failedFiles.length} file error(s); check the selected file cards.`
                }));
                DomHelpers.setText('currentFileProgressTitle', t('translation:batch_completed_with_errors_title', {
                    defaultValue: 'Batch Finished With Errors'
                }));
            } else {
                MessageLogger.showMessage(t('translation:batch_completed'), 'success');
                MessageLogger.addLog(t('translation:batch_completed_log'));
                DomHelpers.setText('currentFileProgressTitle', t('translation:batch_completed_title'));
            }
            return;
        }

        ProgressManager.reset();

        const lastTranslationPreview = DomHelpers.getElement('lastTranslationPreview');
        if (lastTranslationPreview) {
            lastTranslationPreview.innerHTML = `<div style="color: #6b7280; font-style: italic; padding: 10px;">${t('translation:no_translation_yet')}</div>`;
        }

        if (fileToTranslate.fileType === 'epub') {
            DomHelpers.hide('statsGrid');
        } else {
            DomHelpers.show('statsGrid');
        }

        this.updateTranslationTitle(fileToTranslate);
        ProgressManager.show();
        MessageLogger.addLog(t(
            isTransformFile(fileToTranslate)
                ? 'translation:starting_transform_log'
                : 'translation:starting_translation_log',
            {
                name: fileToTranslate.name,
                type: fileToTranslate.fileType.toUpperCase(),
                mode: transformLabelForFile(fileToTranslate),
            }
        ));
        updateFileStatusInList(fileToTranslate.name, 'Preparing...');

        const provider = DomHelpers.getValue('llmProvider');
        const endpoint = effectiveEndpointForProvider(provider);
        const apiKeyValidation = ApiKeyUtils.validateForProvider(provider, endpoint);

        if (!apiKeyValidation.valid) {
            MessageLogger.addLog(t('translation:api_key_error_log', { message: apiKeyValidation.message }));
            MessageLogger.showMessage(apiKeyValidation.message, 'error');
            updateFileStatusInList(fileToTranslate.name, 'Error: Missing API key');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
            return;
        }

        // Validate file path
        if (!fileToTranslate.filePath && !fileToTranslate.content) {
            MessageLogger.addLog(t('translation:critical_no_path_log', { name: fileToTranslate.name }));
            MessageLogger.showMessage(t('translation:critical_no_path_msg', { name: fileToTranslate.name }), 'error');
            updateFileStatusInList(fileToTranslate.name, 'Path Error');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
            return;
        }

        const config = getTranslationConfig(fileToTranslate);

        try {
            const data = await ApiClient.startTranslation(config);

            StateManager.setState('translation.currentJob', {
                fileRef: fileToTranslate,
                translationId: data.translation_id
            });

            fileToTranslate.translationId = data.translation_id;
            updateFileStatusInList(fileToTranslate.name, 'Submitted', data.translation_id);

            DomHelpers.show('progressSection');
            DomHelpers.show('interruptBtn');

            requestAnimationFrame(() => {
                const progressSection = DomHelpers.getElement('progressSection');
                if (progressSection) {
                    progressSection.style.display = 'block';
                }
            });

            this.updateTranslationTitle(fileToTranslate);
            MessageLogger.addLog(t(
                isTransformFile(fileToTranslate)
                    ? 'translation:transform_submitted_log'
                    : 'translation:submitted_log',
                {
                    name: fileToTranslate.name,
                    mode: transformLabelForFile(fileToTranslate),
                }
            ));
            this.removeFileFromProcessingList(fileToTranslate.name);

            const event = new CustomEvent('translationStarted', { detail: { file: fileToTranslate, translationId: data.translation_id } });
            window.dispatchEvent(event);

        } catch (error) {
            MessageLogger.addLog(t('translation:init_error_log', { name: fileToTranslate.name, error: error.message }));
            MessageLogger.showMessage(t('translation:init_error_msg', { name: fileToTranslate.name, error: error.message }), 'error');
            updateFileStatusInList(fileToTranslate.name, 'Initiation Error');
            StateManager.setState('translation.currentJob', null);
            this.processNextFileInQueue();
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
     * Stop batch translation
     */
    stopBatch() {
        StateManager.setState('translation.isBatchActive', false);
        StateManager.setState('translation.currentJob', null);

        // Clear saved translation state from localStorage
        if (TranslationTracker && TranslationTracker.clearTranslationState) {
            TranslationTracker.clearTranslationState();
        }

        const translateBtn = DomHelpers.getElement('translateBtn');
        const filesToProcess = StateManager.getState('files.toProcess') || [];
        if (translateBtn) {
            translateBtn.disabled = filesToProcess.length === 0 || !StatusManager.isConnected();
            translateBtn.innerHTML = t('translation:start_batch_with_icon');
        }

        DomHelpers.hide('interruptBtn');

        MessageLogger.addLog(t('translation:batch_stopped_log'));
        MessageLogger.showMessage(t('translation:batch_stopped'), 'info');
    },

    /**
     * Remove file from processing list
     * @param {string} filename - Filename to remove
     */
    removeFileFromProcessingList(filename) {
        const filesToProcess = StateManager.getState('files.toProcess');
        const fileIndex = filesToProcess.findIndex(f => f.name === filename);

        if (fileIndex !== -1) {
            filesToProcess.splice(fileIndex, 1);
            StateManager.setState('files.toProcess', filesToProcess);
            MessageLogger.addLog(t('translation:file_removed_log', { name: filename }));
            // Notify file list change to update UI and persist to localStorage
            FileUpload.notifyFileListChanged();
        }
    }
};
