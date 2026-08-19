/**
 * Binds the <select id="uiLocaleSelect"> in the Settings tab to setLocale().
 */

import { setLocale, getCurrentLocale, SUPPORTED_LOCALES } from './i18n.js';

export const UiLocaleControl = {
    initialize() {
        const select = document.getElementById('uiLocaleSelect');
        if (!select) return;

        const display = document.getElementById('uiLocaleDisplay');
        const syncDisplay = () => {
            if (!display) return;
            const opt = select.options[select.selectedIndex];
            display.textContent = opt?.dataset?.short || opt?.value?.toUpperCase() || '';
        };

        const current = getCurrentLocale();
        if (SUPPORTED_LOCALES.includes(current)) {
            select.value = current;
        }
        syncDisplay();

        select.addEventListener('change', async () => {
            const value = select.value;
            syncDisplay();
            try {
                await setLocale(value);
            } catch (err) {
                console.error('[i18n] setLocale failed', err);
            }
        });

        window.addEventListener('localeChanged', (event) => {
            const locale = event?.detail?.locale;
            if (locale && select.value !== locale) {
                select.value = locale;
            }
            syncDisplay();
        });
    }
};
