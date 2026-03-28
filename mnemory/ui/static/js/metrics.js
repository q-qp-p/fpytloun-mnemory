/**
 * mnemory UI — Dashboard / Metrics tab component.
 *
 * Loads server stats from /api/stats and renders summary cards
 * plus Chart.js donut charts for memory breakdowns by type,
 * category, and role. Supports auto-refresh on a 30s interval.
 * When a specific user is selected, stat cards and charts show
 * that user's data instead of global totals.
 *
 * Usage:
 *   <div x-data="metricsTab()" x-init="init()"> ... </div>
 */

function metricsTab() {
  return {
    /** Parsed response from /api/stats (null until loaded) */
    data: null,

    /** True while a fetch is in progress */
    loading: true,

    /** Whether the 30-second auto-refresh is active */
    autoRefresh: false,

    /** Handle returned by setInterval (for cleanup) */
    refreshInterval: null,

    /** Map of canvas-id -> Chart instance (for destroy before re-render) */
    chartInstances: {},

    /** Whether event listeners have been registered (guard against duplicates) */
    _listenersRegistered: false,

    // ── Lifecycle ────────────────────────────────────────────────

    /**
     * Initialize: load data immediately and register event listeners
     * so the tab re-fetches when it becomes active or the user changes.
     * Guards against duplicate listeners on repeated init() calls.
     */
    init() {
      // Clear any stale auto-refresh interval from a previous init
      if (this.refreshInterval) {
        clearInterval(this.refreshInterval);
        this.refreshInterval = null;
        this.autoRefresh = false;
      }

      this.loadData();

      if (!this._listenersRegistered) {
        this._listenersRegistered = true;

        // Re-load when the user switches to the dashboard tab
        window.addEventListener('mnemory:tab-changed', (e) => {
          if (e.detail.tab === 'dashboard') {
            this.loadData();
          }
        });

        // Re-render charts when the active user changes (per-user filtering)
        window.addEventListener('mnemory:user-changed', () => {
          if (this.data) {
            this.$nextTick(() => this.renderCharts());
          } else {
            this.loadData();
          }
        });
      }
    },

    // ── Data fetching ───────────────────────────────────────────

    /**
     * Fetch /api/stats and render charts on success.
     */
    async loadData() {
      this.loading = true;
      try {
        this.data = await MnemoryAPI.stats();
        // Expose data in the shared Alpine store so other tabs (e.g. fsck)
        // can read autofsck status without making a separate API call.
        Alpine.store('metrics', this.data);
        // Wait one tick so Alpine has flushed DOM updates for canvases
        this.$nextTick(() => this.renderCharts());
      } catch (err) {
        Alpine.store('notify').error(`Failed to load stats: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    // ── Per-user data helpers ────────────────────────────────────

    /**
     * Return the currently selected user from the auth store, or null.
     * @returns {string|null}
     */
    _selectedUser() {
      return Alpine.store('auth')?.selectedUser || null;
    },

    /**
     * Return the per-user data slice if a user is selected and has data,
     * otherwise return null (caller falls back to global).
     * @returns {object|null}
     */
    _userSlice() {
      const uid = this._selectedUser();
      if (!uid || !this.data?.by_user?.[uid]) return null;
      return this.data.by_user[uid];
    },

    /**
     * Totals to display in stat cards — per-user if selected, else global.
     * @returns {object}
     */
    get displayTotals() {
      const u = this._userSlice();
      return u ? {
                 memories: u.total,
                 raw: u.raw,
                 consolidated: u.consolidated,
                 pinned: u.pinned,
                 decayed: u.decayed,
                 with_artifacts: u.with_artifacts,
               }
               : (this.data?.totals ?? {});
    },

    /**
     * Human-readable age of the most recent auto-fsck run across all users.
     * Returns null if auto-fsck is disabled or no runs have occurred.
     * @returns {string|null}
     */
    get autofsckLastRunAge() {
      const autofsck = this.data?.autofsck;
      if (!autofsck?.enabled) return null;
      const byUser = autofsck.by_user || {};
      let latestTs = null;
      for (const uid of Object.keys(byUser)) {
        const ts = byUser[uid]?.last_run;
        if (ts && (latestTs === null || ts > latestTs)) latestTs = ts;
      }
      if (!latestTs) return null;
      const ageSeconds = Math.floor(Date.now() / 1000) - latestTs;
      if (ageSeconds < 60) return `${ageSeconds}s ago`;
      if (ageSeconds < 3600) return `${Math.floor(ageSeconds / 60)}m ago`;
      if (ageSeconds < 86400) return `${Math.floor(ageSeconds / 3600)}h ago`;
      return `${Math.floor(ageSeconds / 86400)}d ago`;
    },

    // ── Chart rendering ─────────────────────────────────────────

    /**
     * Render (or re-render) the three donut charts using Chart.js.
     * Uses per-user breakdowns when a user is selected.
     *
     * Each chart is keyed by its canvas ID. Existing instances are
     * destroyed first to avoid "Canvas is already in use" errors.
     */
    renderCharts() {
      if (!this.data) return;

      const u = this._userSlice();

      // -- Memory type colors (from CSS custom properties) --
      const typeColors = getTypeColors();

      // -- Brand palette for category / role charts --
      const brandPalette = [
        '#3B82F6', // blue
        '#8B5CF6', // violet
        '#F59E0B', // amber
        '#10B981', // emerald
        '#EF4444', // red
        '#06B6D4', // cyan
        '#F97316', // orange
        '#EC4899', // pink
        '#14B8A6', // teal
        '#A855F7', // purple
        '#64748B', // slate
        '#84CC16', // lime
        '#6366F1', // indigo
      ];

      // By type — use per-user slice if available
      const byType = u?.by_type ?? this.data.by_type;
      this._renderDonut('chart-by-type', byType, typeColors);

      // By category — assign colors from the brand palette in order
      const byCategory = u?.by_category ?? this.data.by_category;
      const catColors = {};
      const catKeys = Object.keys(byCategory || {});
      catKeys.forEach((key, i) => {
        catColors[key] = brandPalette[i % brandPalette.length];
      });
      this._renderDonut('chart-by-category', byCategory, catColors, {
        htmlLegendId: 'chart-by-category-legend',
      });

      // By layer
      const byLayer = u?.by_layer ?? this.data.by_layer;
      const layerColors = {
        raw: '#F59E0B',
        consolidated: '#10B981',
      };
      this._renderDonut('chart-by-layer', byLayer, layerColors);

      // By role
      const byRole = u?.by_role ?? this.data.by_role;
      const roleColors = {
        user:      '#3B82F6', // blue
        assistant: '#8B5CF6', // violet
      };
      this._renderDonut('chart-by-role', byRole, roleColors);
    },

    /**
     * Render a single donut chart on the given canvas element.
     *
     * @param {string}            canvasId   - DOM id of the <canvas>
     * @param {Object<string,number>} dataset - label -> value mapping
     * @param {Object<string,string>} colorMap - label -> hex color mapping
     */
    _renderDonut(canvasId, dataset, colorMap, options = {}) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;
      this._clearHtmlLegend(options.htmlLegendId);

      // Destroy previous instance to allow clean re-render
      if (this.chartInstances[canvasId]) {
        this.chartInstances[canvasId].destroy();
        delete this.chartInstances[canvasId];
      }

      const labels = Object.keys(dataset || {});
      const values = Object.values(dataset || {});

      // Nothing to show — skip rendering
      if (labels.length === 0) return;

      const colors = labels.map((l) => colorMap[l] || '#64748B');

      if (options.htmlLegendId) {
        this._renderHtmlLegend(options.htmlLegendId, labels, values, colors);
      }

      this.chartInstances[canvasId] = new Chart(canvas, {
        type: 'doughnut',
        data: {
          labels,
          datasets: [{
            data: values,
            backgroundColor: colors,
            borderWidth: 0,
          }],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '60%',
          plugins: {
            legend: {
              display: !options.htmlLegendId,
              position: 'bottom',
              labels: {
                color: '#CBD5E1',       // slate-300 — readable on dark bg
                padding: 12,
                usePointStyle: true,
                pointStyleWidth: 8,
                font: { size: 12 },
              },
            },
            tooltip: {
              backgroundColor: 'rgba(15, 23, 42, 0.9)', // slate-900
              titleColor: '#F1F5F9',
              bodyColor: '#F1F5F9',
              borderWidth: 0,
              padding: 10,
            },
          },
          // Transparent canvas background (inherits page/card bg)
          backgroundColor: 'transparent',
        },
      });
    },

    _clearHtmlLegend(legendId) {
      if (!legendId) return;
      const legend = document.getElementById(legendId);
      if (legend) legend.innerHTML = '';
    },

    _renderHtmlLegend(legendId, labels, values, colors) {
      const legend = document.getElementById(legendId);
      if (!legend) return;

      labels.forEach((label, index) => {
        const row = document.createElement('div');
        row.className = 'flex items-start gap-2 text-xs text-secondary py-1';

        const dot = document.createElement('span');
        dot.className = 'w-2.5 h-2.5 rounded-full mt-1 shrink-0';
        dot.style.backgroundColor = colors[index] || '#64748B';

        const textWrap = document.createElement('div');
        textWrap.className = 'min-w-0 flex-1';

        const labelEl = document.createElement('div');
        labelEl.className = 'text-[#E6EDF3] break-words';
        labelEl.textContent = label;

        const valueEl = document.createElement('div');
        valueEl.className = 'text-muted';
        valueEl.textContent = String(values[index] ?? 0);

        textWrap.appendChild(labelEl);
        textWrap.appendChild(valueEl);
        row.appendChild(dot);
        row.appendChild(textWrap);
        legend.appendChild(row);
      });
    },

    // ── Auto-refresh ────────────────────────────────────────────

    /**
     * Toggle a 30-second polling interval for live dashboard updates.
     */
    toggleAutoRefresh() {
      this.autoRefresh = !this.autoRefresh;

      if (this.autoRefresh) {
        this.refreshInterval = setInterval(() => this.loadData(), 30_000);
      } else {
        clearInterval(this.refreshInterval);
        this.refreshInterval = null;
      }
    },

    // ── Cleanup ─────────────────────────────────────────────────

    /**
     * Tear down chart instances and intervals.
     * Called when the component is removed from the DOM.
     */
    destroy() {
      // Destroy all Chart.js instances
      for (const [id, chart] of Object.entries(this.chartInstances)) {
        chart.destroy();
        delete this.chartInstances[id];
      }

      // Clear auto-refresh interval
      if (this.refreshInterval) {
        clearInterval(this.refreshInterval);
        this.refreshInterval = null;
      }
    },
  };
}
