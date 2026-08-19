/**
 * Minimal word-level inline diff for Translate+Refine mode.
 *
 * Builds an HTML string that wraps inserted words in <ins> and removed words
 * in <del>. Uses a standard LCS (longest common subsequence) backtrack — fine
 * for short sample-sized texts (a few hundred words at most).
 *
 * Pure DOM helper. No i18n needed (no user-facing strings).
 */

import { DomHelpers } from '../ui/dom-helpers.js';

// Shared HTML escaper (was a local copy here and in sample-table.js).
const escapeHtml = DomHelpers.escapeHtml;

function tokenize(text) {
    // Keep whitespace as its own token so we can rebuild the layout exactly.
    return String(text || '').split(/(\s+)/);
}

function isWord(token) {
    return token && !/^\s*$/.test(token);
}

function lcsTable(a, b) {
    const m = a.length;
    const n = b.length;
    const dp = Array.from({ length: m + 1 }, () => new Uint32Array(n + 1));
    for (let i = m - 1; i >= 0; i -= 1) {
        for (let j = n - 1; j >= 0; j -= 1) {
            if (a[i] === b[j]) {
                dp[i][j] = dp[i + 1][j + 1] + 1;
            } else {
                dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
            }
        }
    }
    return dp;
}

/**
 * Build an HTML diff between `before` and `after`, word-level.
 * Unchanged tokens are escaped as-is; insertions get <ins>, deletions <del>.
 */
export function inlineDiff(before, after) {
    const a = tokenize(before);
    const b = tokenize(after);
    if (a.length === 0) return `<ins>${escapeHtml(after)}</ins>`;
    if (b.length === 0) return `<del>${escapeHtml(before)}</del>`;

    const dp = lcsTable(a, b);
    const out = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
        if (a[i] === b[j]) {
            out.push(escapeHtml(a[i]));
            i += 1;
            j += 1;
        } else if (dp[i + 1][j] >= dp[i][j + 1]) {
            if (isWord(a[i])) {
                out.push(`<del>${escapeHtml(a[i])}</del>`);
            } else {
                out.push(escapeHtml(a[i]));
            }
            i += 1;
        } else {
            if (isWord(b[j])) {
                out.push(`<ins>${escapeHtml(b[j])}</ins>`);
            } else {
                out.push(escapeHtml(b[j]));
            }
            j += 1;
        }
    }
    while (i < a.length) {
        out.push(isWord(a[i]) ? `<del>${escapeHtml(a[i])}</del>` : escapeHtml(a[i]));
        i += 1;
    }
    while (j < b.length) {
        out.push(isWord(b[j]) ? `<ins>${escapeHtml(b[j])}</ins>` : escapeHtml(b[j]));
        j += 1;
    }
    return out.join('');
}
