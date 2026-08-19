import { FileUpload } from '../files/file-upload.js';
import { BatchController } from './batch-controller.js?v=20260705-transform-flow';
import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';
import { t } from '../i18n/i18n.js';

export const TextTransformManager = {
    pendingFiles: [],
    profilesLoaded: false,

    init() {
        this.bindEvents();
        this.loadProfiles();
        this.updateModeDescription();
        this.renderPendingFiles();
    },

    bindEvents() {
        const upload = DomHelpers.getElement('transformFileUpload');
        const input = DomHelpers.getElement('transformFileInput');
        const mode = DomHelpers.getElement('transformMode');
        const profile = DomHelpers.getElement('transformProfile');
        const start = DomHelpers.getElement('transformStartBtn');
        const clear = DomHelpers.getElement('transformClearBtn');

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

        if (mode) {
            mode.addEventListener('change', () => this.updateModeDescription());
        }
        if (profile) {
            profile.addEventListener('change', () => this.updateModeDescription());
        }
        window.addEventListener('localeChanged', () => {
            this.loadProfiles();
            this.updateModeDescription();
        });
        window.addEventListener('bookProfilesChanged', () => {
            this.loadProfiles();
            this.updateModeDescription();
        });

        if (start) {
            start.addEventListener('click', () => this.start());
        }

        if (clear) {
            clear.addEventListener('click', () => {
                this.pendingFiles = [];
                this.renderPendingFiles();
            });
        }

    },

    setPendingFiles(files) {
        this.pendingFiles = files.filter(Boolean);
        this.renderPendingFiles();
    },

    selectedMode() {
        const select = DomHelpers.getElement('transformMode');
        const option = select?.selectedOptions?.[0];
        return {
            value: select?.value || 'modernize',
            label: option?.dataset?.label || option?.textContent?.trim() || 'Transformar',
            description: option?.dataset?.description || '',
        };
    },

    selectedProfile() {
        const select = DomHelpers.getElement('transformProfile');
        return select?.value || '';
    },

    async loadProfiles() {
        const transformSelect = DomHelpers.getElement('transformProfile');
        const translationSelect = DomHelpers.getElement('bookProfileSelect');
        if (!transformSelect && !translationSelect) return;
        const currentTransform = transformSelect?.value || '';
        const currentTranslation = translationSelect?.value || '';
        try {
            const data = await ApiClient.getBookProfiles();
            const profiles = Array.isArray(data.profiles) ? data.profiles : [];
            this.populateProfileSelect(transformSelect, profiles, currentTransform, t('transform:profile_none'));
            this.populateProfileSelect(translationSelect, profiles, currentTranslation, t('settings:select_none'));
            this.profilesLoaded = true;
        } catch (error) {
            if (!this.profilesLoaded) {
                console.warn('[profiles] could not load book profiles', error);
            }
        }
    },

    populateProfileSelect(select, profiles, currentValue, noneLabel) {
        if (!select) return;
        select.innerHTML = '';
        const general = document.createElement('option');
        general.value = '';
        general.textContent = noneLabel || t('settings:select_none');
        select.appendChild(general);
        for (const profile of profiles) {
            this.upsertProfileOptionForSelect(select, profile, false);
        }
        if (currentValue && Array.from(select.options).some((option) => option.value === currentValue)) {
            select.value = currentValue;
        }
    },

    upsertProfileOptionForSelect(select, profile, selectIt = false) {
        if (!select || !profile?.profile_id) return;
        let option = Array.from(select.options).find((item) => item.value === profile.profile_id);
        if (!option) {
            option = document.createElement('option');
            option.value = profile.profile_id;
            select.appendChild(option);
        }
        option.textContent = this.profileOptionLabel(profile);
        option.dataset.generated = profile.generated_profile ? '1' : '0';
        if (selectIt) {
            select.value = profile.profile_id;
        }
    },

    upsertProfileEverywhere(profile, selectIt = false) {
        this.upsertProfileOptionForSelect(DomHelpers.getElement('transformProfile'), profile, selectIt);
        this.upsertProfileOptionForSelect(DomHelpers.getElement('bookProfileSelect'), profile, selectIt);
        if (selectIt) {
            this.updateModeDescription();
        }
    },

    profileOptionLabel(profile) {
        const name = profile?.name || profile?.profile_name || profile?.profile_id || '';
        const approved = Number(profile?.approved_count ?? profile?.approved_entries ?? 0);
        const pending = Number(profile?.pending_count ?? profile?.pending_suggestions ?? 0);
        const signals = this.editorialSignalCount(profile);
        if (approved || pending || signals) {
            return t('transform:profile_option_counts', {
                name,
                approved,
                pending,
                signals,
                defaultValue: `${name} (${approved}/${pending}/${signals})`,
            });
        }
        return name;
    },

    editorialSignalCount(profile) {
        const counts = profile?.editorial_artifact_counts || profile?.editorial_artifacts || {};
        if (!counts || typeof counts !== 'object') return 0;
        return Object.values(counts).reduce((sum, value) => {
            const number = Number(value || 0);
            return sum + (Number.isFinite(number) ? number : 0);
        }, 0);
    },

    upsertProfileOption(profile, selectIt = false) {
        this.upsertProfileEverywhere(profile, selectIt);
    },

    updateModeDescription() {
        const mode = this.selectedMode();
        const profile = this.selectedProfile();
        const description = DomHelpers.getElement('transformModeDescription');
        if (description) {
            const suffix = profile
                ? ` ${t('transform:profile_active_note')}`
                : '';
            description.textContent = `${mode.description}${suffix}`;
        }
    },

    renderPendingFiles() {
        const list = DomHelpers.getElement('transformPendingFiles');
        const start = DomHelpers.getElement('transformStartBtn');
        const clear = DomHelpers.getElement('transformClearBtn');
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

        if (start) start.disabled = this.pendingFiles.length === 0;
        if (clear) clear.disabled = this.pendingFiles.length === 0;
    },

    async start() {
        if (this.pendingFiles.length === 0) {
            MessageLogger.showMessage(t('transform:select_file_first'), 'info');
            return;
        }

        const mode = this.selectedMode();
        const language = DomHelpers.getValue('transformLanguage') || DomHelpers.getValue('targetLang') || 'Spanish';
        const outputFormat = DomHelpers.getValue('transformOutputFormat') || 'auto';
        const files = [...this.pendingFiles];

        MessageLogger.showMessage(t('transform:queueing', { count: files.length }), 'info');
        await FileUpload.handleFiles(files, 'refine', {
            language,
            outputFormat,
            transformMode: mode.value,
            transformLabel: mode.label,
            transformDescription: mode.description,
            profileId: this.selectedProfile(),
            outputLabel: mode.label,
        });

        this.pendingFiles = [];
        this.renderPendingFiles();

        if (typeof window.switchTopTab === 'function') {
            window.switchTopTab('translate');
        }
        await BatchController.startBatchTranslation();
    },
};
