import { ApiClient } from '../core/api-client.js';
import { DomHelpers } from '../ui/dom-helpers.js';
import { MessageLogger } from '../ui/message-logger.js';

function money(value) {
    const n = Number(value || 0);
    if (n === 0) return '$0.0000';
    if (n < 0.0001) return '$<0.0001';
    return `$${n.toFixed(4)}`;
}

function number(value) {
    return new Intl.NumberFormat().format(Math.round(Number(value || 0)));
}

function decimal(value, digits = 1) {
    return new Intl.NumberFormat(undefined, {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
    }).format(Number(value || 0));
}

function shortDate(ts) {
    if (!ts) return '';
    return new Date(ts * 1000).toLocaleString();
}

function clockTime() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function shortName(name) {
    const raw = String(name || 'Sin libro');
    return raw.length > 86 ? `${raw.slice(0, 83)}...` : raw;
}

function pct(value, total) {
    if (!total) return 0;
    return Math.max(3, Math.min(100, (Number(value || 0) / Number(total)) * 100));
}

function cacheLabel(row) {
    const hit = Number(row?.prompt_cache_hit_tokens ?? row?.recent_prompt_cache_hit_tokens ?? 0);
    const miss = Number(row?.prompt_cache_miss_tokens ?? row?.recent_prompt_cache_miss_tokens ?? 0);
    const total = hit + miss;
    if (!total) return '';
    return `Cache ${decimal((hit / total) * 100, 0)}%`;
}

export const UsageManager = {
    _summary: null,
    _pollTimer: null,
    _active: false,
    _loading: false,

    initialize() {
        const refreshBtn = DomHelpers.getElement('usageRefreshBtn');
        if (refreshBtn) refreshBtn.addEventListener('click', () => this.refresh({ silent: false }));

        const backfillBtn = DomHelpers.getElement('usageBackfillBtn');
        if (backfillBtn) backfillBtn.addEventListener('click', () => this.backfill());

        window.addEventListener('usageRefreshRequested', () => this.activate());
        window.addEventListener('topTabChanged', (event) => {
            if (event?.detail?.tab === 'usage') this.activate();
            else this.deactivate();
        });
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.deactivate();
            } else if (!DomHelpers.getElement('tab-usage')?.classList.contains('hidden')) {
                this.activate();
            }
        });
    },

    activate() {
        this._active = true;
        this.refresh({ silent: false });
        this.startPolling();
    },

    deactivate() {
        this._active = false;
        if (this._pollTimer) {
            clearInterval(this._pollTimer);
            this._pollTimer = null;
        }
    },

    startPolling() {
        if (this._pollTimer) return;
        this._pollTimer = setInterval(() => {
            if (!this._active || document.hidden) return;
            this.refresh({ silent: true });
        }, 3000);
    },

    async refresh({ silent = false } = {}) {
        if (this._loading) return;
        this._loading = true;
        const loading = DomHelpers.getElement('usageLoading');
        const content = DomHelpers.getElement('usageContent');
        if (!silent && loading) loading.classList.remove('hidden');
        if (!silent && content) content.classList.add('usage-dimmed');
        try {
            this._summary = await ApiClient.getUsageSummary(120);
            this.render(this._summary);
        } catch (error) {
            if (!silent) MessageLogger.showMessage(`No se pudo cargar el uso de tokens: ${error.message}`, 'error');
        } finally {
            if (!silent && loading) loading.classList.add('hidden');
            if (!silent && content) content.classList.remove('usage-dimmed');
            this._loading = false;
        }
    },

    async backfill() {
        const btn = DomHelpers.getElement('usageBackfillBtn');
        if (btn) btn.disabled = true;
        try {
            const result = await ApiClient.backfillUsageFromCheckpoints();
            MessageLogger.showMessage(`Uso histórico estimado: ${result.created || 0} job(s), ${result.skipped || 0} omitido(s).`, 'success');
            await this.refresh();
        } catch (error) {
            MessageLogger.showMessage(`No se pudo reconstruir uso histórico: ${error.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    },

    render(summary) {
        const totals = summary?.totals || {};
        DomHelpers.setText('usageTotalCost', money(totals.total_cost_usd));
        DomHelpers.setText('usageTotalTokens', number(totals.total_tokens));
        DomHelpers.setText('usagePromptTokens', number(totals.prompt_tokens));
        DomHelpers.setText('usageCompletionTokens', number(totals.completion_tokens));
        DomHelpers.setText('usageCallCount', number(totals.calls));
        DomHelpers.setText('usageEstimatedEvents', number(totals.estimated_events));
        DomHelpers.setText('usageLiveStatus', `Actualizado ${clockTime()} · cada 3s`);

        this.renderLiveJobs(summary?.live_jobs || []);
        this.renderBooks(summary?.by_book || []);
        this.renderBreakdowns(summary?.by_model || [], summary?.by_phase || summary?.by_process || []);
        this.renderEvents(summary?.recent_events || []);
        this.renderDailyChart(summary?.daily || []);
    },

    renderLiveJobs(rows) {
        const host = DomHelpers.getElement('usageLiveJobsList');
        const empty = DomHelpers.getElement('usageLiveEmpty');
        if (!host) return;
        host.innerHTML = '';
        if (!rows.length) {
            if (empty) empty.classList.remove('hidden');
            return;
        }
        if (empty) empty.classList.add('hidden');
        rows.forEach((row) => {
            const totalChunks = Number(row.total_chunks || 0);
            const completed = Number(row.completed_chunks || 0);
            const failed = Number(row.failed_chunks || 0);
            const progress = Math.max(0, Math.min(100, Number(row.progress_percent || 0)));
            const projectionChips = this.liveProjectionChips(row);
            const cache = cacheLabel(row);
            const phaseChips = (row.phase_breakdown || []).slice(0, 6).map((phase) => `
                <span class="usage-phase-chip">
                    ${DomHelpers.escapeHtml(phase.phase || phase.process_type || 'fase')}
                    <b>${money(phase.total_cost_usd)}</b>
                    ${Number(phase.cost_share_pct || 0) ? `<em>${decimal(phase.cost_share_pct, 0)}%</em>` : ''}
                    ${cacheLabel(phase) ? `<em>${DomHelpers.escapeHtml(cacheLabel(phase))}</em>` : ''}
                </span>
            `).join('');
            const item = document.createElement('div');
            item.className = 'usage-live-row';
            item.innerHTML = `
                <div class="usage-live-top">
                    <div>
                        <div class="usage-book-title" title="${DomHelpers.escapeHtml(row.book_name || '')}">${DomHelpers.escapeHtml(shortName(row.book_name))}</div>
                        <div class="usage-book-meta">
                            ${DomHelpers.escapeHtml(row.status || 'activo')} · ${DomHelpers.escapeHtml(row.process_type || 'proceso')} · ${DomHelpers.escapeHtml(row.provider || '')} ${DomHelpers.escapeHtml(row.model || '')}
                        </div>
                    </div>
                    <div class="usage-book-values">
                        <strong>${money(row.total_cost_usd)}</strong>
                        <span>${number(row.total_tokens)} tokens</span>
                    </div>
                </div>
                <div class="usage-live-stats">
                    <span>${number(row.calls)} llamadas</span>
                    <span>Entrada ${number(row.prompt_tokens)}</span>
                    <span>Salida ${number(row.completion_tokens)}</span>
                    ${cache ? `<span>${DomHelpers.escapeHtml(cache)}</span>` : ''}
                    ${totalChunks ? `<span>${number(completed)}/${number(totalChunks)} fragmentos</span>` : ''}
                    ${failed ? `<span>${number(failed)} fallidos</span>` : ''}
                    ${Number(row.estimated_events || 0) ? '<span>incluye estimados</span>' : ''}
                </div>
                ${projectionChips ? `<div class="usage-live-stats usage-live-projections">${projectionChips}</div>` : ''}
                <div class="usage-bar usage-live-progress"><span style="width:${Math.max(3, progress)}%"></span></div>
                <div class="usage-live-progress-label">${progress.toFixed(1)}%</div>
                ${phaseChips ? `<div class="usage-phase-chips">${phaseChips}</div>` : '<div class="usage-book-meta">Esperando la primera llamada registrada del modelo.</div>'}
            `;
            host.appendChild(item);
        });
    },

    liveProjectionChips(row) {
        const chips = [];
        const projectedCost = Number(row.projected_total_cost_usd || 0);
        const remainingCost = Number(row.projected_remaining_cost_usd || 0);
        const costPerChunk = Number(row.cost_per_processed_chunk_usd || 0);
        const tokensPerChunk = Number(row.tokens_per_processed_chunk || 0);
        const recentCostPerMinute = Number(row.recent_cost_per_minute_usd || 0);
        const recentTokensPerMinute = Number(row.recent_tokens_per_minute || 0);
        const costPer1k = Number(row.cost_per_1k_tokens_usd || 0);

        if (projectedCost > 0) chips.push(`Proyección ${money(projectedCost)}`);
        if (remainingCost > 0) chips.push(`Restante ${money(remainingCost)}`);
        if (costPerChunk > 0) chips.push(`${money(costPerChunk)} / fragmento`);
        if (tokensPerChunk > 0) chips.push(`${number(tokensPerChunk)} tokens / fragmento`);
        if (recentCostPerMinute > 0 || recentTokensPerMinute > 0) {
            chips.push(`Últimos 5 min: ${money(recentCostPerMinute)}/min · ${number(recentTokensPerMinute)} tok/min`);
        }
        if (costPer1k > 0) chips.push(`${money(costPer1k)} / 1k tokens`);
        if (!chips.length && Number(row.calls || 0) === 0) chips.push('Sin gasto registrado todavía');

        return chips.map((chip) => `<span>${DomHelpers.escapeHtml(chip)}</span>`).join('');
    },

    renderBooks(rows) {
        const host = DomHelpers.getElement('usageBooksList');
        const empty = DomHelpers.getElement('usageBooksEmpty');
        if (!host) return;
        host.innerHTML = '';
        if (!rows.length) {
            if (empty) empty.classList.remove('hidden');
            return;
        }
        if (empty) empty.classList.add('hidden');
        const maxCost = Math.max(...rows.map((r) => Number(r.total_cost_usd || 0)), 0);
        const maxTokens = Math.max(...rows.map((r) => Number(r.total_tokens || 0)), 1);
        rows.forEach((row) => {
            const item = document.createElement('div');
            item.className = 'usage-book-row';
            const barByCost = maxCost > 0 ? pct(row.total_cost_usd, maxCost) : pct(row.total_tokens, maxTokens);
            const cache = cacheLabel(row);
            item.innerHTML = `
                <div class="usage-book-main">
                    <div class="usage-book-title" title="${DomHelpers.escapeHtml(row.book_name || '')}">${DomHelpers.escapeHtml(shortName(row.book_name))}</div>
                    <div class="usage-book-meta">
                        ${DomHelpers.escapeHtml(row.process_type || 'unknown')} · ${number(row.calls)} llamadas · ${shortDate(row.last_seen)}
                        ${Number(row.estimated_events || 0) > 0 ? ' · estimado' : ''}
                        ${cache ? ` · ${DomHelpers.escapeHtml(cache)}` : ''}
                    </div>
                </div>
                <div class="usage-book-values">
                    <strong>${money(row.total_cost_usd)}</strong>
                    <span>${number(row.total_tokens)} tokens</span>
                </div>
                <div class="usage-bar"><span style="width:${barByCost}%"></span></div>
            `;
            host.appendChild(item);
        });
    },

    renderBreakdowns(models, processes) {
        this.renderMiniTable('usageModelsBody', models, (row) => [
            `${row.provider || ''} / ${row.model || ''}`,
            number(row.calls),
            `${number(row.total_tokens)}${cacheLabel(row) ? ` · ${cacheLabel(row)}` : ''}`,
            money(row.total_cost_usd),
        ]);
        this.renderMiniTable('usageProcessesBody', processes, (row) => [
            row.phase || row.process_type || 'unknown',
            number(row.calls),
            `${number(row.total_tokens)}${cacheLabel(row) ? ` · ${cacheLabel(row)}` : ''}`,
            money(row.total_cost_usd),
        ]);
    },

    renderMiniTable(id, rows, mapper) {
        const body = DomHelpers.getElement(id);
        if (!body) return;
        body.innerHTML = '';
        rows.forEach((row) => {
            const tr = document.createElement('tr');
            tr.innerHTML = mapper(row).map((cell, idx) => (
                `<td class="${idx > 0 ? 'col-right' : ''}">${DomHelpers.escapeHtml(String(cell))}</td>`
            )).join('');
            body.appendChild(tr);
        });
    },

    renderEvents(events) {
        const body = DomHelpers.getElement('usageEventsBody');
        if (!body) return;
        body.innerHTML = '';
        events.slice(0, 80).forEach((event) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${DomHelpers.escapeHtml(shortDate(event.created_at))}</td>
                <td>${DomHelpers.escapeHtml(event.phase || event.process_type || 'unknown')}</td>
                <td>${DomHelpers.escapeHtml(shortName(event.book_name || event.input_filename || event.output_filename || event.translation_id || ''))}</td>
                <td>${DomHelpers.escapeHtml(`${event.provider || ''} / ${event.model || ''}`)}</td>
                <td class="col-right">${number(event.prompt_tokens)}</td>
                <td class="col-right">${number(event.completion_tokens)}</td>
                <td class="col-right">${money(event.total_cost_usd)}</td>
                <td>${event.estimated_tokens ? 'estimado' : DomHelpers.escapeHtml(event.status || 'ok')}${cacheLabel(event) ? ` · ${DomHelpers.escapeHtml(cacheLabel(event))}` : ''}</td>
            `;
            body.appendChild(tr);
        });
    },

    renderDailyChart(rows) {
        const canvas = DomHelpers.getElement('usageDailyChart');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const width = canvas.width = canvas.clientWidth * window.devicePixelRatio;
        const height = canvas.height = 180 * window.devicePixelRatio;
        ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        const w = canvas.clientWidth;
        const h = 180;
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--text-muted-light') || '#8b949e';
        ctx.font = '12px Inter, sans-serif';
        if (!rows.length) {
            ctx.fillText('Sin datos todavía', 16, 94);
            return;
        }
        const pad = 24;
        const maxTokens = Math.max(...rows.map((r) => Number(r.total_tokens || 0)), 1);
        const barGap = 8;
        const barWidth = Math.max(12, (w - pad * 2 - barGap * (rows.length - 1)) / rows.length);
        rows.forEach((row, idx) => {
            const x = pad + idx * (barWidth + barGap);
            const barH = Math.max(2, (Number(row.total_tokens || 0) / maxTokens) * (h - 60));
            const y = h - 34 - barH;
            const grad = ctx.createLinearGradient(0, y, 0, h - 34);
            grad.addColorStop(0, '#60a5fa');
            grad.addColorStop(1, '#22c55e');
            ctx.fillStyle = grad;
            ctx.fillRect(x, y, barWidth, barH);
            ctx.fillStyle = '#9ca3af';
            ctx.fillText(String(row.day || '').slice(5), x, h - 12);
        });
    },
};
