import { t } from '../i18n/i18n.js';

/** Distinguish editions of a book profile using the localized preparation choices. */
export function bookProfileName(profile, { includeLanguage = true } = {}) {
    const name = profile?.name || profile?.profile_name || profile?.profile_id || '';
    if (!profile?.profile_goal) return name;
    const choices = document.getElementById('profilePrepPurpose')?.options || [];
    const choice = Array.from(choices).find(option => option.dataset.profileGoal === profile.profile_goal);
    if (!choice) return name;
    const language = includeLanguage && (profile.target_locale || profile.target_language);
    return [name, t(choice.dataset.i18n), language].filter(Boolean).join(' · ');
}
