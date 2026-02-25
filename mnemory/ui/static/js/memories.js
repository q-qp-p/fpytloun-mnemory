/**
 * mnemory UI — Memories Browse Tab (Alpine.js component).
 *
 * Lists all memories with filtering, inline expand, edit modal,
 * and delete confirmation. Lazy-loads on first tab activation.
 */

function memoriesTab() {
  return {
    // ── State ──────────────────────────────────────────────────
    memories: [],
    loading: false,
    totalCount: 0,

    filters: {
      memory_type: '',
      categories: [],
      role: '',
      limit: 25,
      include_decayed: false,
    },

    expandedId: null,
    availableCategories: [],

    editModal: {
      open: false,
      memory: null,
      form: {},
    },

    deleteConfirm: null,
    initialized: false,

    // ── Lifecycle ──────────────────────────────────────────────

    /**
     * Initialize the component.
     * Listens for tab activation and user switching events.
     * Loads category list for the filter dropdown.
     */
    init() {
      // Load data when the memories tab becomes active for the first time
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'memories' && !this.initialized) {
          this.initialized = true;
          this.loadMemories(false);
        }
      });

      // Reload when the user switches to a different identity
      window.addEventListener('mnemory:user-changed', () => {
        if (this.initialized) {
          this.loadMemories(false);
        }
      });

      // Fetch category list for filter dropdowns
      this.loadCategories();
    },

    /**
     * Fetch available categories from the API.
     */
    async loadCategories() {
      try {
        const data = await MnemoryAPI.categories();
        // The API returns { categories: [ { name, ... }, ... ] }
        this.availableCategories = (data.categories || []).map((c) => c.name);
      } catch (err) {
        console.warn('Failed to load categories:', err);
      }
    },

    // ── Data Loading ──────────────────────────────────────────

    /**
     * Load memories from the API.
     * @param {boolean} append - If true, append to existing list (load more).
     *                           If false, replace the list entirely.
     */
    async loadMemories(append = false) {
      if (!append) {
        this.memories = [];
      }
      this.loading = true;

      try {
        // Build query params from current filters
        const params = {
          limit: this.filters.limit,
          include_decayed: this.filters.include_decayed,
        };
        if (this.filters.memory_type) {
          params.memory_type = this.filters.memory_type;
        }
        if (this.filters.categories.length > 0) {
          params.categories = this.filters.categories.join(',');
        }
        if (this.filters.role) {
          params.role = this.filters.role;
        }

        const data = await MnemoryAPI.listMemories(params);
        const results = data.results || [];

        if (append) {
          this.memories = this.memories.concat(results);
        } else {
          this.memories = results;
        }

        // The list endpoint returns all matching results up to limit,
        // so totalCount is approximated from what we received.
        this.totalCount = this.memories.length;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load memories: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    /**
     * Load more results by increasing the limit.
     * The list API has no offset, so we request a larger batch.
     */
    loadMore() {
      this.filters.limit += 25;
      this.loadMemories(false);
    },

    /**
     * Reset to first page and reload with current filters.
     */
    applyFilters() {
      this.filters.limit = 25;
      this.loadMemories(false);
    },

    // ── Expand / Collapse ─────────────────────────────────────

    /**
     * Toggle the expanded detail view for a memory.
     * @param {string} id - Memory ID to toggle.
     */
    toggleExpand(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    // ── Edit Modal ────────────────────────────────────────────

    /**
     * Open the edit modal for a memory, pre-populating the form.
     * @param {object} memory - The memory object to edit.
     */
    openEdit(memory) {
      const meta = memory.metadata || {};
      this.editModal = {
        open: true,
        memory: memory,
        form: {
          content: memory.memory || '',
          memory_type: meta.memory_type || '',
          categories: (meta.categories || []).join(', '),
          importance: meta.importance || 'normal',
          pinned: !!meta.pinned,
          ttl_days: meta.ttl_days ?? '',
        },
      };
    },

    /**
     * Save edits from the modal to the API and update the local list.
     */
    async saveEdit() {
      const form = this.editModal.form;
      const memoryId = this.editModal.memory.id;

      // Build the update payload — only include changed fields
      const payload = {};

      if (form.content !== this.editModal.memory.memory) {
        payload.content = form.content;
      }

      const meta = this.editModal.memory.metadata || {};

      if (form.memory_type && form.memory_type !== meta.memory_type) {
        payload.memory_type = form.memory_type;
      }

      // Parse comma-separated categories back to array
      const newCategories = form.categories
        ? form.categories.split(',').map((c) => c.trim()).filter(Boolean)
        : [];
      const oldCategories = meta.categories || [];
      if (JSON.stringify(newCategories) !== JSON.stringify(oldCategories)) {
        payload.categories = newCategories;
      }

      if (form.importance && form.importance !== meta.importance) {
        payload.importance = form.importance;
      }

      if (form.pinned !== !!meta.pinned) {
        payload.pinned = form.pinned;
      }

      // TTL: empty string means no change, a number sets it
      if (form.ttl_days !== '' && form.ttl_days !== null) {
        const ttl = parseInt(form.ttl_days, 10);
        if (!isNaN(ttl) && ttl !== meta.ttl_days) {
          payload.ttl_days = ttl;
        }
      }

      // Nothing to update
      if (Object.keys(payload).length === 0) {
        this.editModal.open = false;
        return;
      }

      try {
        await MnemoryAPI.updateMemory(memoryId, payload);

        // Update the local memory object to reflect changes immediately
        const idx = this.memories.findIndex((m) => m.id === memoryId);
        if (idx !== -1) {
          const mem = this.memories[idx];
          if (payload.content !== undefined) {
            mem.memory = payload.content;
          }
          if (!mem.metadata) mem.metadata = {};
          if (payload.memory_type) mem.metadata.memory_type = payload.memory_type;
          if (payload.categories) mem.metadata.categories = payload.categories;
          if (payload.importance) mem.metadata.importance = payload.importance;
          if (payload.pinned !== undefined) mem.metadata.pinned = payload.pinned;
          if (payload.ttl_days !== undefined) mem.metadata.ttl_days = payload.ttl_days;
        }

        this.editModal.open = false;
        Alpine.store('notify').success('Memory updated');
      } catch (err) {
        Alpine.store('notify').error(`Failed to update: ${err.message}`);
      }
    },

    // ── Delete ────────────────────────────────────────────────

    /**
     * Delete a memory after confirmation.
     * @param {string} id - Memory ID to delete.
     */
    async deleteMemory(id) {
      try {
        await MnemoryAPI.deleteMemory(id);
        this.memories = this.memories.filter((m) => m.id !== id);
        this.totalCount = this.memories.length;
        this.deleteConfirm = null;
        Alpine.store('notify').success('Memory deleted');
      } catch (err) {
        this.deleteConfirm = null;
        Alpine.store('notify').error(`Failed to delete: ${err.message}`);
      }
    },

    // ── Clipboard ─────────────────────────────────────────────

    /**
     * Copy a memory ID to the clipboard.
     * @param {string} id - The ID to copy.
     */
    copyId(id) {
      navigator.clipboard.writeText(id).then(
        () => Alpine.store('notify').success('ID copied to clipboard'),
        () => Alpine.store('notify').error('Failed to copy ID'),
      );
    },

    // ── Display Helpers ───────────────────────────────────────

    /**
     * Return a Tailwind badge class for the memory type.
     * @param {string} type - Memory type (fact, preference, etc.)
     * @returns {string} CSS class string.
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
     * Return a Tailwind badge class for the importance level.
     * @param {string} importance - Importance level (low, normal, high, critical).
     * @returns {string} CSS class string.
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
     * Truncate a string to a maximum length, appending ellipsis.
     * @param {string} str - The string to truncate.
     * @param {number} max - Maximum character length (default 120).
     * @returns {string} Truncated string.
     */
    truncate(str, max = 120) {
      if (!str) return '';
      return str.length > max ? str.substring(0, max) + '...' : str;
    },

    /**
     * Format an ISO date string for display.
     * @param {string} dateStr - ISO 8601 date string.
     * @returns {string} Formatted date or empty string.
     */
    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        const d = new Date(dateStr);
        return d.toLocaleString(undefined, {
          year: 'numeric',
          month: 'short',
          day: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        });
      } catch {
        return dateStr;
      }
    },
  };
}
