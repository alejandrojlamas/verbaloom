#!/usr/bin/env node
/**
 * extract-keys.mjs — i18n consistency check.
 *
 * Walks the HTML template + JS modules under src/web/static/, extracts every
 * key referenced via data-i18n / data-i18n-attr / data-i18n-html and via
 * t('ns:key') / i18next.t('ns:key') calls, and diffs them against the JSON
 * files in src/web/static/locales/<lng>/<ns>.json.
 *
 * Reports per locale:
 *   - missing keys     (referenced in code but absent from JSON)
 *   - orphan keys      (present in JSON but never referenced)
 *   - empty namespaces (JSON file is `{}`)
 *
 * Exit code: 1 if EN has missing keys (reference locale must always be
 * complete), 0 otherwise. Non-EN missing keys are warnings, not errors —
 * they fall back to EN at runtime.
 *
 * Run from the repo root:   node src/web/static/js/i18n/extract-keys.mjs
 * Or with a stricter mode:   node src/web/static/js/i18n/extract-keys.mjs --strict
 */

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..', '..', '..', '..');

const STATIC_DIR = path.join(REPO_ROOT, 'src', 'web', 'static');
const TEMPLATE_FILE = path.join(REPO_ROOT, 'src', 'web', 'templates', 'translation_interface.html');
const LOCALES_DIR = path.join(STATIC_DIR, 'locales');
const JS_DIR = path.join(STATIC_DIR, 'js');

const STRICT = process.argv.includes('--strict');

const NS_KEY_RE = /^([a-z][a-z0-9_]*?):([a-zA-Z0-9_.\-]+)$/;
// Capture data-i18n="...", data-i18n-html="...", and individual attr:key pairs
// inside data-i18n-attr="attr1:key1;attr2:key2".
const HTML_TEXT_RE = /data-i18n(?:-html)?="([^"]+)"/g;
const HTML_ATTR_RE = /data-i18n-attr="([^"]+)"/g;
// t('ns:key', ...) or i18next.t('ns:key', ...) — single or double quotes.
const JS_T_RE = /(?:\b|\.)t\(\s*['"]([a-z][a-z0-9_]*?:[a-zA-Z0-9_.\-]+)['"]/g;

async function walk(dir, exts) {
    const out = [];
    let entries;
    try {
        entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
        return out;
    }
    for (const entry of entries) {
        const full = path.join(dir, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === 'vendor' || entry.name === 'locales') continue;
            out.push(...(await walk(full, exts)));
        } else if (exts.some((e) => entry.name.endsWith(e))) {
            out.push(full);
        }
    }
    return out;
}

function collectKeys(content, regex, group = 1) {
    const found = new Set();
    let m;
    while ((m = regex.exec(content)) !== null) {
        found.add(m[group]);
    }
    return found;
}

function parseHtmlKeys(content) {
    const keys = new Set();

    let m;
    while ((m = HTML_TEXT_RE.exec(content)) !== null) {
        // data-i18n / data-i18n-html — single key
        if (NS_KEY_RE.test(m[1])) keys.add(m[1]);
    }
    while ((m = HTML_ATTR_RE.exec(content)) !== null) {
        // data-i18n-attr — list of attr:key pairs, separated by ';'
        for (const pair of m[1].split(';')) {
            const idx = pair.indexOf(':');
            if (idx < 0) continue;
            const key = pair.slice(idx + 1).trim();
            if (NS_KEY_RE.test(key)) keys.add(key);
        }
    }
    return keys;
}

function parseJsKeys(content) {
    return collectKeys(content, JS_T_RE);
}

async function loadLocale(locale) {
    const dir = path.join(LOCALES_DIR, locale);
    let entries;
    try {
        entries = await fs.readdir(dir);
    } catch {
        return { available: false, map: new Map(), empty: [] };
    }
    const map = new Map();
    const empty = [];
    for (const entry of entries) {
        if (!entry.endsWith('.json')) continue;
        const ns = entry.replace(/\.json$/, '');
        const raw = await fs.readFile(path.join(dir, entry), 'utf8');
        let json = {};
        try {
            json = JSON.parse(raw);
        } catch (e) {
            console.error(`[${locale}/${entry}] invalid JSON: ${e.message}`);
            process.exitCode = 1;
            continue;
        }
        const flat = flatten(json);
        if (Object.keys(flat).length === 0) empty.push(ns);
        for (const k of Object.keys(flat)) {
            map.set(`${ns}:${k}`, true);
        }
    }
    return { available: true, map, empty };
}

function flatten(obj, prefix = '') {
    const out = {};
    for (const [k, v] of Object.entries(obj || {})) {
        const key = prefix ? `${prefix}.${k}` : k;
        if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
            Object.assign(out, flatten(v, key));
        } else {
            out[key] = v;
        }
    }
    return out;
}

async function main() {
    // 1. Collect referenced keys from sources.
    const usedKeys = new Set();

    const htmlContent = await fs.readFile(TEMPLATE_FILE, 'utf8').catch(() => '');
    for (const k of parseHtmlKeys(htmlContent)) usedKeys.add(k);

    const jsFiles = await walk(JS_DIR, ['.js', '.mjs']);
    for (const f of jsFiles) {
        if (f.endsWith('extract-keys.mjs')) continue;
        const content = await fs.readFile(f, 'utf8');
        for (const k of parseJsKeys(content)) usedKeys.add(k);
    }

    // 2. Inspect every locale.
    const locales = (await fs.readdir(LOCALES_DIR, { withFileTypes: true }))
        .filter((d) => d.isDirectory())
        .map((d) => d.name)
        .sort();

    let hadEnErrors = false;
    const report = [];

    for (const locale of locales) {
        const { map, empty } = await loadLocale(locale);
        const defined = new Set(map.keys());

        const missing = [...usedKeys].filter((k) => !defined.has(k)).sort();
        const orphans = [...defined].filter((k) => !usedKeys.has(k)).sort();

        report.push({ locale, missing, orphans, empty });

        if (locale === 'en' && missing.length > 0) hadEnErrors = true;
    }

    // 3. Print summary.
    const sep = '─'.repeat(72);
    console.log(`\nUsed keys across HTML + JS: ${usedKeys.size}`);
    console.log(sep);
    for (const { locale, missing, orphans, empty } of report) {
        console.log(`\n[${locale}]`);
        console.log(`  missing : ${missing.length}`);
        if (missing.length > 0) {
            for (const k of missing.slice(0, 20)) console.log(`    - ${k}`);
            if (missing.length > 20) console.log(`    ... (${missing.length - 20} more)`);
        }
        console.log(`  orphans : ${orphans.length}`);
        if (orphans.length > 0 && STRICT) {
            for (const k of orphans.slice(0, 20)) console.log(`    - ${k}`);
            if (orphans.length > 20) console.log(`    ... (${orphans.length - 20} more)`);
        }
        if (empty.length > 0) {
            console.log(`  empty   : ${empty.join(', ')}`);
        }
    }

    if (hadEnErrors) {
        console.error(`\nERROR: 'en' locale has missing keys — it must be the complete reference.`);
        process.exit(1);
    }
    if (STRICT) {
        const anyMissing = report.some((r) => r.missing.length > 0);
        if (anyMissing) {
            console.error(`\nSTRICT: at least one locale has missing keys.`);
            process.exit(1);
        }
    }
    console.log('\nOK');
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
