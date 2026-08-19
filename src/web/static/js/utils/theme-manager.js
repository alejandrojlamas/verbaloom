/**
 * Theme Manager
 *
 * Handles dark/light mode theme switching and persistence.
 */

export class ThemeManager {
    constructor() {
        this.htmlElement = document.documentElement;
        this.themeIcon = document.getElementById('themeIcon');
        this.themeColorMeta = document.getElementById('themeColorMeta');
        this.STORAGE_KEY = 'tbl-theme-preference';
        this.themeChrome = {
            dark: '#080c12',
            light: '#f5f7fb'
        };

        // Initialize theme from localStorage or the product default.
        this.initializeTheme();
    }

    /**
     * Initialize theme based on saved preference or system default
     */
    initializeTheme() {
        const savedTheme = localStorage.getItem(this.STORAGE_KEY);

        if (savedTheme) {
            this.setTheme(savedTheme);
        } else {
            // TBL is optimized for long mobile reading sessions; default to
            // the dark product theme while still preserving explicit user choice.
            this.setTheme('dark');
        }

        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            // Only auto-switch if user hasn't manually set a preference
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.setTheme(e.matches ? 'dark' : 'light');
            }
        });
    }

    /**
     * Get current theme
     * @returns {string} 'light' or 'dark'
     */
    getCurrentTheme() {
        return this.htmlElement.classList.contains('dark') ? 'dark' : 'light';
    }

    /**
     * Set theme
     * @param {string} theme - 'light' or 'dark'
     */
    setTheme(theme) {
        const normalizedTheme = theme === 'dark' ? 'dark' : 'light';

        if (normalizedTheme === 'dark') {
            this.htmlElement.classList.add('dark');
            if (this.themeIcon) {
                this.themeIcon.textContent = 'light_mode';
            }
        } else {
            this.htmlElement.classList.remove('dark');
            if (this.themeIcon) {
                this.themeIcon.textContent = 'dark_mode';
            }
        }

        this.syncBrowserChrome(normalizedTheme);

        // Save preference
        localStorage.setItem(this.STORAGE_KEY, normalizedTheme);
    }

    /**
     * Keep browser-owned UI areas (Android bottom gesture area, status bar)
     * visually aligned with the active app theme.
     *
     * @param {string} theme - 'light' or 'dark'
     */
    syncBrowserChrome(theme) {
        const color = this.themeChrome[theme] || this.themeChrome.light;
        this.htmlElement.style.backgroundColor = color;
        if (document.body) {
            document.body.style.backgroundColor = color;
        }
        if (this.themeColorMeta) {
            this.themeColorMeta.setAttribute('content', color);
        }
    }

    /**
     * Toggle between light and dark themes
     */
    toggleTheme() {
        const currentTheme = this.getCurrentTheme();
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    }
}

// Global instance
let themeManagerInstance = null;

/**
 * Initialize theme manager
 * @returns {ThemeManager}
 */
export function initializeThemeManager() {
    if (!themeManagerInstance) {
        themeManagerInstance = new ThemeManager();
    }
    return themeManagerInstance;
}

/**
 * Get theme manager instance
 * @returns {ThemeManager}
 */
export function getThemeManager() {
    return themeManagerInstance;
}

/**
 * Global function for onclick handler
 * Called from HTML template
 */
window.toggleTheme = function() {
    const manager = getThemeManager();
    if (manager) {
        manager.toggleTheme();
    }
};
