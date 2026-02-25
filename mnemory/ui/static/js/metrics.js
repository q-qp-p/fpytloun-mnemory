/**
 * mnemory UI — Dashboard / Metrics tab component.
 *
 * Loads server stats from /api/stats and renders summary cards
 * plus Chart.js donut charts for memory breakdowns by type,
 * category, and role. Supports auto-refresh on a 30s interval.
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

        // Re-load when the active user changes (multi-user switching)
        window.addEventListener('mnemory:user-changed', () => {
          this.loadData();
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
        // Wait one tick so Alpine has flushed DOM updates for canvases
        this.$nextTick(() => this.renderCharts());
      } catch (err) {
        Alpine.store('notify').error(`Failed to load stats: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    // ── Chart rendering ─────────────────────────────────────────

    /**
     * Render (or re-render) the three donut charts using Chart.js.
     *
     * Each chart is keyed by its canvas ID. Existing instances are
     * destroyed first to avoid "Canvas is already in use" errors.
     */
    renderCharts() {
      if (!this.data) return;

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

      // By type
      this._renderDonut('chart-by-type', this.data.by_type, typeColors);

      // By category — assign colors from the brand palette in order
      const catColors = {};
      const catKeys = Object.keys(this.data.by_category || {});
      catKeys.forEach((key, i) => {
        catColors[key] = brandPalette[i % brandPalette.length];
      });
      this._renderDonut('chart-by-category', this.data.by_category, catColors);

      // By role
      const roleColors = {
        user:      '#3B82F6', // blue
        assistant: '#8B5CF6', // violet
      };
      this._renderDonut('chart-by-role', this.data.by_role, roleColors);
    },

    /**
     * Render a single donut chart on the given canvas element.
     *
     * @param {string}            canvasId   - DOM id of the <canvas>
     * @param {Object<string,number>} dataset - label -> value mapping
     * @param {Object<string,string>} colorMap - label -> hex color mapping
     */
    _renderDonut(canvasId, dataset, colorMap) {
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

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
