/**
 * mnemory UI — Graph tab component (D3.js force-directed visualization).
 *
 * Renders memories as a force-directed graph where nodes are memories
 * and edges connect memories that share categories. Node size reflects
 * importance, color maps to memory type, and pinned memories get a
 * cyan ring.
 *
 * Usage:
 *   <div x-data="graphTab()" x-init="init()"> ... </div>
 */

// ── Constants ───────────────────────────────────────────────────

/** Memory type → node fill color (lazy-loaded from CSS custom properties) */
let TYPE_COLORS = null;
function _getTypeColors() {
  if (!TYPE_COLORS) TYPE_COLORS = getTypeColors();
  return TYPE_COLORS;
}

/** Importance level → node radius in pixels */
const IMPORTANCE_RADIUS = {
  low:      6,
  normal:   10,
  high:     14,
  critical: 18,
};

/** Default radius when importance is unknown */
const DEFAULT_RADIUS = 10;

/** Pinned-node ring color (brand accent / cyan) */
const PINNED_STROKE = '#22D3EE';

// ── Component ───────────────────────────────────────────────────

function graphTab() {
  return {
    /** Raw memory objects from the API */
    memories: [],

    /** True while fetching data */
    loading: false,

    /** Whether the graph has been rendered at least once */
    initialized: false,

    /** Currently selected (clicked) node — full memory object */
    selectedNode: null,

    /** Max memories to fetch for graph rendering */
    nodeLimit: 100,

    /** Toggle visibility per memory type */
    typeFilters: {
      fact:       true,
      preference: true,
      episodic:   true,
      procedural: true,
      context:    true,
    },

    /** D3 force simulation reference (for stop/restart) */
    simulation: null,

    /** D3 SVG selection reference */
    svg: null,

    /** D3 zoom behavior reference (for resetZoom) */
    _zoom: null,

    /** Tooltip element reference */
    _tooltip: null,

    // ── Lifecycle ──────────────────────────────────────────────

    /**
     * Initialize event listeners. The graph is only built when the
     * 'graph' tab becomes active for the first time (lazy load).
     */
    init() {
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'graph') {
          this.loadAndRender();
        }
      });

      window.addEventListener('mnemory:user-changed', () => {
        // Reset state so the next activation rebuilds from scratch
        this.initialized = false;
        this.selectedNode = null;
        if (Alpine.store('nav').activeTab === 'graph') {
          this.loadAndRender();
        }
      });
    },

    // ── Data fetching ──────────────────────────────────────────

    /**
     * Fetch memories from the API and build the graph.
     */
    async loadAndRender() {
      this.loading = true;
      try {
        const result = await MnemoryAPI.listMemories({ limit: this.nodeLimit });
        // listMemories returns { results: [...] }
        this.memories = result.results || [];
        this.$nextTick(() => this.buildGraph());
        this.initialized = true;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load graph data: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    // ── Graph construction ─────────────────────────────────────

    /**
     * Build the D3 force-directed graph from the current memory set.
     * Clears any previous SVG and simulation before rendering.
     */
    buildGraph() {
      const container = document.getElementById('graph-container');
      if (!container) return;

      // Tear down previous render
      this._teardownGraph();

      const width = container.clientWidth || 800;
      const height = container.clientHeight || 600;

      // ── Filter memories by active type toggles ───────────
      const filtered = this.memories.filter(
        (m) => this.typeFilters[(m.metadata || {}).memory_type] !== false
      );

      if (filtered.length === 0) {
        // Nothing to draw — show empty state SVG
        this.svg = d3.select(container)
          .append('svg')
          .attr('width', width)
          .attr('height', height);
        this.svg.append('text')
          .attr('x', width / 2)
          .attr('y', height / 2)
          .attr('text-anchor', 'middle')
          .attr('fill', '#64748B')
          .attr('font-size', '14px')
          .text('No memories match the current filters');
        return;
      }

      // ── Build node list ──────────────────────────────────
      const nodes = filtered.map((m) => {
        const meta = m.metadata || {};
        const text = m.memory || '';
        return {
          id:         m.id,
          content:    text.slice(0, 50) + (text.length > 50 ? '...' : ''),
          type:       meta.memory_type || 'context',
          importance: meta.importance || 'normal',
          pinned:     !!meta.pinned,
          categories: meta.categories || [],
          fullData:   m,
        };
      });

      // ── Build edge list (shared categories) ──────────────
      const links = [];
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const shared = nodes[i].categories.filter(
            (c) => nodes[j].categories.includes(c)
          );
          if (shared.length > 0) {
            links.push({
              source: nodes[i].id,
              target: nodes[j].id,
              weight: shared.length,
            });
          }
        }
      }

      // Max weight for normalizing opacity
      const maxWeight = links.reduce((max, l) => Math.max(max, l.weight), 1);

      // ── Create SVG ───────────────────────────────────────
      this.svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('background', 'transparent');

      // Group for zoom/pan transforms
      const g = this.svg.append('g');

      // ── Zoom behavior ────────────────────────────────────
      this._zoom = d3.zoom()
        .scaleExtent([0.1, 6])
        .on('zoom', (event) => {
          g.attr('transform', event.transform);
        });

      this.svg.call(this._zoom);

      // ── Create tooltip element ───────────────────────────
      this._ensureTooltip();

      // ── Force simulation ─────────────────────────────────
      this.simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links)
          .id((d) => d.id)
          .distance((d) => 120 / d.weight)   // closer if more shared categories
        )
        .force('charge', d3.forceManyBody().strength(-80))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collide', d3.forceCollide()
          .radius((d) => (IMPORTANCE_RADIUS[d.importance] || DEFAULT_RADIUS) + 4)
        );

      // ── Draw links ───────────────────────────────────────
      const linkSel = g.append('g')
        .attr('class', 'graph-links')
        .selectAll('line')
        .data(links)
        .join('line')
          .attr('stroke', '#1E293B')
          .attr('stroke-width', (d) => Math.max(1, d.weight))
          .attr('stroke-opacity', (d) => 0.3 + 0.5 * (d.weight / maxWeight));

      // ── Draw nodes ───────────────────────────────────────
      const self = this;

      const nodeSel = g.append('g')
        .attr('class', 'graph-nodes')
        .selectAll('circle')
        .data(nodes)
        .join('circle')
          .attr('r', (d) => IMPORTANCE_RADIUS[d.importance] || DEFAULT_RADIUS)
          .attr('fill', (d) => _getTypeColors()[d.type] || _getTypeColors().context)
          .attr('fill-opacity', 0.85)
          .attr('stroke', (d) => d.pinned ? PINNED_STROKE : 'none')
          .attr('stroke-width', (d) => d.pinned ? 2 : 0)
          .attr('cursor', 'pointer')
          // ── Hover → tooltip ──────────────────────────────
          .on('mouseenter', function (event, d) {
            // Highlight node
            d3.select(this)
              .attr('fill-opacity', 1)
              .attr('stroke', PINNED_STROKE)
              .attr('stroke-width', 2);

            self._showTooltip(event, d);
          })
          .on('mousemove', function (event) {
            self._moveTooltip(event);
          })
          .on('mouseleave', function (event, d) {
            // Restore original style
            d3.select(this)
              .attr('fill-opacity', 0.85)
              .attr('stroke', d.pinned ? PINNED_STROKE : 'none')
              .attr('stroke-width', d.pinned ? 2 : 0);

            self._hideTooltip();
          })
          // ── Click → select node ──────────────────────────
          .on('click', function (event, d) {
            event.stopPropagation();
            self.selectedNode = d.fullData;
          })
          // ── Drag behavior ────────────────────────────────
          .call(this._dragBehavior());

      // ── Click on background → deselect ───────────────────
      this.svg.on('click', () => {
        this.selectedNode = null;
      });

      // ── Simulation tick → update positions ───────────────
      this.simulation.on('tick', () => {
        linkSel
          .attr('x1', (d) => d.source.x)
          .attr('y1', (d) => d.source.y)
          .attr('x2', (d) => d.target.x)
          .attr('y2', (d) => d.target.y);

        nodeSel
          .attr('cx', (d) => d.x)
          .attr('cy', (d) => d.y);
      });
    },

    // ── Drag behavior ──────────────────────────────────────────

    /**
     * Returns a D3 drag behavior for nodes. Pins dragged nodes by
     * setting fx/fy, and reheats the simulation while dragging.
     */
    _dragBehavior() {
      const sim = () => this.simulation;

      return d3.drag()
        .on('start', function (event, d) {
          if (!event.active && sim()) sim().alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', function (event, d) {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', function (event, d) {
          if (!event.active && sim()) sim().alphaTarget(0);
          // Release node so simulation can place it freely again
          d.fx = null;
          d.fy = null;
        });
    },

    // ── Tooltip helpers ────────────────────────────────────────

    /**
     * Create (or re-use) the tooltip div attached to the graph container.
     */
    _ensureTooltip() {
      if (this._tooltip) return;

      const container = document.getElementById('graph-container');
      if (!container) return;

      // Make container positioned so the absolute tooltip stays inside
      container.style.position = 'relative';

      const el = document.createElement('div');
      el.className = 'graph-tooltip';
      Object.assign(el.style, {
        position:      'absolute',
        pointerEvents: 'none',
        display:       'none',
        zIndex:        '50',
        maxWidth:      '280px',
        padding:       '8px 12px',
        borderRadius:  '8px',
        fontSize:      '12px',
        lineHeight:    '1.4',
        background:    '#1A2438',       // bg-brand-elevated
        border:        '1px solid #1E293B', // border-brand-border
        color:         '#E6EDF3',       // text-primary
        boxShadow:     '0 4px 12px rgba(0,0,0,0.4)',
      });

      container.appendChild(el);
      this._tooltip = el;
    },

    /** Show the tooltip near the cursor with node info */
    _showTooltip(event, d) {
      if (!this._tooltip) return;

      const typeLabel = d.type.charAt(0).toUpperCase() + d.type.slice(1);
      const importanceLabel = d.importance.charAt(0).toUpperCase() + d.importance.slice(1);
      const pinnedBadge = d.pinned ? ' &middot; <span style="color:#22D3EE">Pinned</span>' : '';
      const cats = d.categories.length > 0
        ? `<div style="color:#94A3B8;margin-top:4px">${d.categories.join(', ')}</div>`
        : '';

      this._tooltip.innerHTML = `
        <div style="font-weight:600;margin-bottom:4px;">${this._escapeHtml(d.content)}</div>
        <div style="color:#94A3B8">
          <span style="color:${_getTypeColors()[d.type] || '#64748B'}">${typeLabel}</span>
          &middot; ${importanceLabel}${pinnedBadge}
        </div>
        ${cats}
      `;
      this._tooltip.style.display = 'block';
      this._moveTooltip(event);
    },

    /** Reposition tooltip relative to the graph container */
    _moveTooltip(event) {
      if (!this._tooltip) return;

      const container = document.getElementById('graph-container');
      if (!container) return;

      const rect = container.getBoundingClientRect();
      const x = event.clientX - rect.left + 14;
      const y = event.clientY - rect.top - 10;

      // Keep tooltip inside the container bounds
      const maxX = container.clientWidth - this._tooltip.offsetWidth - 8;
      const maxY = container.clientHeight - this._tooltip.offsetHeight - 8;

      this._tooltip.style.left = Math.min(x, maxX) + 'px';
      this._tooltip.style.top  = Math.min(y, maxY) + 'px';
    },

    /** Hide the tooltip */
    _hideTooltip() {
      if (this._tooltip) {
        this._tooltip.style.display = 'none';
      }
    },

    /** Simple HTML-escape for tooltip text */
    _escapeHtml(text) {
      const el = document.createElement('span');
      el.textContent = text;
      return el.innerHTML;
    },

    // ── Public actions ─────────────────────────────────────────

    /**
     * Reset zoom/pan to the default identity transform.
     */
    resetZoom() {
      if (this.svg && this._zoom) {
        this.svg.transition()
          .duration(500)
          .call(this._zoom.transform, d3.zoomIdentity);
      }
    },

    /**
     * Rebuild the graph after type filter toggles change.
     */
    updateFilters() {
      if (this.memories.length > 0) {
        this.buildGraph();
      }
    },

    /**
     * Switch to the search tab and trigger a search for the given
     * memory's content. Dispatches a custom event that the search
     * tab can listen for.
     */
    searchRelated(memory) {
      if (!memory) return;

      const query = (memory.memory || '').slice(0, 100);
      Alpine.store('nav').setTab('search');

      // Give the search tab a tick to mount, then dispatch
      this.$nextTick(() => {
        window.dispatchEvent(new CustomEvent('mnemory:search-from-graph', {
          detail: { query },
        }));
      });
    },

    // ── Cleanup ────────────────────────────────────────────────

    /**
     * Internal: tear down SVG, simulation, and tooltip without
     * resetting component state.
     */
    _teardownGraph() {
      if (this.simulation) {
        this.simulation.stop();
        this.simulation = null;
      }

      const container = document.getElementById('graph-container');
      if (container) {
        // Remove SVG
        const existingSvg = container.querySelector('svg');
        if (existingSvg) existingSvg.remove();

        // Remove tooltip
        if (this._tooltip) {
          this._tooltip.remove();
          this._tooltip = null;
        }
      }

      this.svg = null;
      this._zoom = null;
    },

    /**
     * Full cleanup — stop simulation, remove SVG, reset state.
     * Called when the component is removed from the DOM.
     */
    destroy() {
      this._teardownGraph();
      this.memories = [];
      this.initialized = false;
      this.selectedNode = null;
    },
  };
}
