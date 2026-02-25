/**
 * mnemory UI — Memories Browse Tab (Alpine.js component).
 *
 * Lists all memories with filtering, sorting, inline expand,
 * add/edit/delete, and artifact management.
 * Edit modal and artifact manager are global stores (app.js).
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

    sortBy: 'newest',

    expandedId: null,
    availableCategories: [],

    // Add memory modal (local — only triggered from this tab)
    addModal: {
      open: false,
      saving: false,
      showAdvanced: false,
      form: {
        content: '',
        memory_type: '',
        categories: '',
        importance: '',
        pinned: false,
        role: 'user',
        agent_id: '',
        ttl_days: '',
        event_date: '',
        infer: true,
      },
    },

    deleteConfirm: null,
    initialized: false,

    // ── Lifecycle ──────────────────────────────────────────────

    init() {
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'memories' && !this.initialized) {
          this.initialized = true;
          this.loadMemories(false);
        }
      });

      window.addEventListener('mnemory:user-changed', () => {
        if (this.initialized) {
          this.loadMemories(false);
        }
      });

      this.loadCategories();
    },

    async loadCategories() {
      try {
        const data = await MnemoryAPI.categories();
        this.availableCategories = (data.categories || []).map((c) => c.name);
      } catch (err) {
        console.warn('Failed to load categories:', err);
      }
    },

    // ── Data Loading ──────────────────────────────────────────

    async loadMemories(append = false) {
      if (!append) this.memories = [];
      this.loading = true;
      try {
        const params = {
          limit: this.filters.limit,
          include_decayed: this.filters.include_decayed,
        };
        if (this.filters.memory_type) params.memory_type = this.filters.memory_type;
        if (this.filters.categories.length > 0) params.categories = this.filters.categories.join(',');
        if (this.filters.role) params.role = this.filters.role;

        const data = await MnemoryAPI.listMemories(params);
        const results = data.results || [];
        this.memories = append ? this.memories.concat(results) : results;
        this.totalCount = this.memories.length;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load memories: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    loadMore() {
      this.filters.limit += 25;
      this.loadMemories(false);
    },

    applyFilters() {
      this.filters.limit = 25;
      this.loadMemories(false);
    },

    // ── Sorting ───────────────────────────────────────────────

    _importanceWeight(importance) {
      return { critical: 4, high: 3, normal: 2, low: 1 }[importance] ?? 2;
    },

    get sortedMemories() {
      const arr = [...this.memories];
      switch (this.sortBy) {
        case 'newest':
          return arr.sort((a, b) => (b.metadata?.created_at_utc || '').localeCompare(a.metadata?.created_at_utc || ''));
        case 'oldest':
          return arr.sort((a, b) => (a.metadata?.created_at_utc || '').localeCompare(b.metadata?.created_at_utc || ''));
        case 'importance':
          return arr.sort((a, b) => this._importanceWeight(b.metadata?.importance) - this._importanceWeight(a.metadata?.importance));
        case 'type':
          return arr.sort((a, b) => (a.metadata?.memory_type || '').localeCompare(b.metadata?.memory_type || ''));
        case 'alpha':
          return arr.sort((a, b) => (a.memory || '').localeCompare(b.memory || ''));
        default:
          return arr;
      }
    },

    // ── Expand / Collapse ─────────────────────────────────────

    toggleExpand(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    // ── Add Memory ────────────────────────────────────────────

    openAdd() {
      this.addModal = {
        open: true,
        saving: false,
        showAdvanced: false,
        form: {
          content: '',
          memory_type: '',
          categories: '',
          importance: '',
          pinned: false,
          role: 'user',
          agent_id: '',
          ttl_days: '',
          event_date: '',
          infer: true,
        },
      };
    },

    async saveAdd() {
      const f = this.addModal.form;
      if (!f.content.trim()) {
        Alpine.store('notify').error('Content is required');
        return;
      }
      this.addModal.saving = true;
      try {
        const payload = { content: f.content, infer: f.infer, role: f.role };
        if (f.memory_type) payload.memory_type = f.memory_type;
        if (f.importance) payload.importance = f.importance;
        if (f.pinned) payload.pinned = true;
        if (f.agent_id.trim()) payload.agent_id = f.agent_id.trim();
        if (f.ttl_days !== '') {
          const ttl = parseInt(f.ttl_days, 10);
          if (!isNaN(ttl)) payload.ttl_days = ttl;
        }
        if (f.event_date.trim()) payload.event_date = f.event_date.trim();
        const cats = f.categories ? f.categories.split(',').map(c => c.trim()).filter(Boolean) : [];
        if (cats.length > 0) payload.categories = cats;

        const result = await MnemoryAPI.addMemory(payload);
        // result.results is an array of {id, memory, event}
        const added = (result.results || []);
        if (added.length > 0) {
          // Reload to get full metadata; or prepend a minimal item
          await this.loadMemories(false);
          Alpine.store('notify').success(`Memory added (${added.length} fact${added.length > 1 ? 's' : ''} stored)`);
        } else {
          Alpine.store('notify').info('No new facts extracted — memory may already exist');
        }
        this.addModal.open = false;
      } catch (err) {
        Alpine.store('notify').error(`Failed to add memory: ${err.message}`);
      } finally {
        this.addModal.saving = false;
      }
    },

    // ── Edit Memory (delegates to global store) ───────────────

    openEdit(mem) {
      Alpine.store('memoryEdit').show(mem, (payload) => {
        // Update the local memory object immediately
        const idx = this.memories.findIndex(m => m.id === mem.id);
        if (idx !== -1) {
          const m = this.memories[idx];
          if (payload.content !== undefined) m.memory = payload.content;
          if (!m.metadata) m.metadata = {};
          if (payload.memory_type) m.metadata.memory_type = payload.memory_type;
          if (payload.categories) m.metadata.categories = payload.categories;
          if (payload.importance) m.metadata.importance = payload.importance;
          if (payload.pinned !== undefined) m.metadata.pinned = payload.pinned;
          if (payload.ttl_days !== undefined) m.metadata.ttl_days = payload.ttl_days;
        }
      });
    },

    // ── Artifacts (delegates to global store) ─────────────────

    openArtifacts(mem) {
      Alpine.store('artifactMgr').show(mem);
    },

    // ── Delete ────────────────────────────────────────────────

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

    copyId(id) {
      navigator.clipboard.writeText(id).then(
        () => Alpine.store('notify').success('ID copied to clipboard'),
        () => Alpine.store('notify').error('Failed to copy ID'),
      );
    },

    // ── Display Helpers ───────────────────────────────────────

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

    importanceBadgeClass(importance) {
      const classes = {
        low: 'badge-low',
        normal: 'badge-normal',
        high: 'badge-high',
        critical: 'badge-critical',
      };
      return classes[importance] || 'badge-normal';
    },

    truncate(str, max = 120) {
      if (!str) return '';
      return str.length > max ? str.substring(0, max) + '...' : str;
    },

    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        const d = new Date(dateStr);
        return d.toLocaleString(undefined, {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch { return dateStr; }
    },
  };
}
