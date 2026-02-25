/**
 * mnemory UI — Search tab Alpine.js component.
 *
 * Provides semantic search (single-query) and AI-powered find (multi-query)
 * with filtering, result expansion, inline delete, and clipboard copy.
 *
 * Usage:
 *   <div x-data="searchTab()" x-init="init()">
 */

function searchTab() {
  return {
    // ── State ────────────────────────────────────────────────────

    /** Current search query text */
    query: '',

    /** Search mode: 'search' (fast vector) or 'find' (AI multi-query) */
    mode: 'search',

    /** Array of result objects from the last search */
    results: [],

    /** Whether a search request is in-flight */
    loading: false,

    /** Whether at least one search has been performed (controls empty state) */
    searched: false,

    /** ID of the currently expanded result (null = all collapsed) */
    expandedId: null,

    /** Search filters applied alongside the query */
    filters: {
      memory_type: '',
      categories: [],
      role: '',
      limit: 10,
      include_decayed: false,
    },

    /** Whether the filter panel is visible */
    showFilters: false,

    /** Available categories loaded from the API (for filter dropdown) */
    availableCategories: [],

    /** Memory ID pending delete confirmation (null = no pending delete) */
    deleteConfirm: null,

    // ── Lifecycle ────────────────────────────────────────────────

    /**
     * Initialize the component: load categories and listen for user switches.
     */
    async init() {
      // Load categories for the filter dropdown
      try {
        const data = await MnemoryAPI.categories();
        // API returns { categories: [ { name, ... }, ... ] }
        this.availableCategories = (data.categories || []).map(c => c.name || c);
      } catch {
        // Non-critical — filters will just lack category options
        this.availableCategories = [];
      }

      // Clear results when the active user changes
      window.addEventListener('mnemory:user-changed', () => {
        this.results = [];
        this.searched = false;
        this.expandedId = null;
        this.deleteConfirm = null;
      });
    },

    // ── Search ───────────────────────────────────────────────────

    /**
     * Execute a search or find based on the current mode and filters.
     */
    async search() {
      const q = this.query.trim();
      if (!q) return;

      this.loading = true;
      this.searched = true;
      this.expandedId = null;
      this.deleteConfirm = null;

      try {
        // Build filter payload — only include non-empty values
        const filters = {};
        if (this.filters.memory_type) {
          filters.memory_type = this.filters.memory_type;
        }
        if (this.filters.categories.length > 0) {
          filters.categories = this.filters.categories;
        }
        if (this.filters.role) {
          filters.role = this.filters.role;
        }
        if (this.filters.limit && this.filters.limit !== 10) {
          filters.limit = this.filters.limit;
        }
        if (this.filters.include_decayed) {
          filters.include_decayed = true;
        }

        let response;
        if (this.mode === 'find') {
          // AI-powered multi-query search
          response = await MnemoryAPI.findMemories(q, filters);
        } else {
          // Fast single-query vector search
          response = await MnemoryAPI.searchMemories(q, filters);
        }

        this.results = response.results || [];
      } catch (e) {
        Alpine.store('notify').error(`Search failed: ${e.message}`);
        this.results = [];
      } finally {
        this.loading = false;
      }
    },

    // ── Result interactions ───────────────────────────────────────

    /**
     * Toggle expansion of a result card by its ID.
     */
    toggleExpand(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    /**
     * Delete a memory by ID (requires prior confirmation via deleteConfirm).
     */
    async deleteMemory(id) {
      try {
        await MnemoryAPI.deleteMemory(id);
        this.results = this.results.filter(r => r.id !== id);
        this.deleteConfirm = null;
        Alpine.store('notify').success('Memory deleted');
      } catch (e) {
        Alpine.store('notify').error(`Delete failed: ${e.message}`);
      }
    },

    /**
     * Copy a memory ID to the clipboard.
     */
    async copyId(id) {
      try {
        await navigator.clipboard.writeText(id);
        Alpine.store('notify').success('Memory ID copied to clipboard');
      } catch {
        Alpine.store('notify').error('Failed to copy to clipboard');
      }
    },

    // ── Display helpers ──────────────────────────────────────────

    /**
     * CSS badge class for a memory type.
     * @param {string} type - Memory type (fact, preference, episodic, etc.)
     * @returns {string} CSS class name
     */
    typeBadgeClass(type) {
      const classes = {
        fact: 'badge-fact',
        preference: 'badge-preference',
        episodic: 'badge-episodic',
        procedural: 'badge-procedural',
        context: 'badge-context',
      };
      return classes[type] || 'badge-context';
    },

    /**
     * CSS badge class for an importance level.
     * @param {string} importance - Importance level (low, normal, high, critical)
     * @returns {string} CSS class name
     */
    importanceBadgeClass(importance) {
      const classes = {
        low: 'badge-low',
        normal: 'badge-normal',
        high: 'badge-high',
        critical: 'badge-critical',
      };
      return classes[importance] || 'badge-normal';
    },

    /**
     * Color for a relevance score (0.0–1.0).
     * Green for high confidence, yellow for mid, red for low.
     * @param {number} score - Relevance score between 0 and 1
     * @returns {string} CSS color value
     */
    scoreColor(score) {
      if (score >= 0.7) return '#22c55e'; // green
      if (score >= 0.4) return '#eab308'; // yellow
      return '#ef4444';                    // red
    },

    /**
     * Truncate text to a maximum length, adding ellipsis if needed.
     * @param {string} text - Input text
     * @param {number} len - Maximum length (default 100)
     * @returns {string} Truncated text
     */
    truncate(text, len = 100) {
      if (!text) return '';
      if (text.length <= len) return text;
      return text.slice(0, len) + '...';
    },

    /**
     * Format an ISO 8601 date string to a human-readable format.
     * Returns empty string for null/undefined input.
     * @param {string|null} isoStr - ISO date string
     * @returns {string} Formatted date (e.g. "Jan 15, 2025, 10:00 AM")
     */
    formatDate(isoStr) {
      if (!isoStr) return '';
      try {
        const date = new Date(isoStr);
        return date.toLocaleString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        });
      } catch {
        return isoStr;
      }
    },
  };
}
