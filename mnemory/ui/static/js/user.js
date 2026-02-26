/**
 * mnemory UI — User tab Alpine.js component.
 *
 * Shows core memories organized into sections matching the
 * get_core_memories format: User Facts, User Preferences,
 * User Context, and Recent Activity.
 *
 * Fetches structured memory data via GET /api/memories and
 * organizes client-side (since get_core_memories returns
 * pre-formatted text, not structured data).
 *
 * Agent filter uses dual-scope logic: when an agent is selected,
 * shows memories visible to that agent (agent_id matches OR no
 * agent_id set). When "All" is selected, shows everything.
 */

function userTab() {
  return {
    // ── State ──────────────────────────────────────────────────
    memories: [],
    loading: false,
    initialized: false,
    recentDays: 7,
    expandedId: null,
    deleteConfirm: null,

    /** Client-side agent filter (dual-scope: selected + shared) */
    filterAgentId: '',

    /** All known agent IDs (loaded from stats API) */
    availableAgentIds: [],

    /** Memory types considered "recent activity" */
    _recentTypes: ['episodic', 'context', 'procedural'],

    // ── Lifecycle ──────────────────────────────────────────────

    init() {
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'user' && !this.initialized) {
          this.initialized = true;
          this.loadData();
        }
      });

      window.addEventListener('mnemory:user-changed', () => {
        if (this.initialized) {
          this.loadData();
        }
        this._loadAgentIds();
      });

      this._loadAgentIds();
    },

    // ── Data Loading ──────────────────────────────────────────

    async loadData() {
      this.loading = true;
      try {
        const data = await MnemoryAPI.listMemories({
          limit: 5000,
          include_decayed: false,
        });
        this.memories = data.results || [];
      } catch (err) {
        Alpine.store('notify').error(`Failed to load memories: ${err.message}`);
        this.memories = [];
      } finally {
        this.loading = false;
      }
    },

    /** Load all known agent IDs from the stats API */
    async _loadAgentIds() {
      try {
        const data = await MnemoryAPI.stats();
        this.availableAgentIds = data.agents || [];
      } catch {
        this.availableAgentIds = [];
      }
    },

    // ── Agent Filter ──────────────────────────────────────────

    /**
     * Dual-scope agent visibility check.
     * When an agent is selected, a memory is visible if:
     *   - it has no agent_id (shared), OR
     *   - its agent_id matches the selected agent
     * When no agent is selected ("All"), everything is visible.
     */
    _isVisibleToAgent(m) {
      if (!this.filterAgentId) return true;
      const aid = m.metadata?.agent_id;
      return !aid || aid === this.filterAgentId;
    },

    // ── Computed Sections ─────────────────────────────────────

    /** Pinned facts (role=user only, agent-filtered) */
    get pinnedFacts() {
      return this.memories.filter(m =>
        m.metadata?.pinned &&
        m.metadata?.memory_type === 'fact' &&
        m.metadata?.role !== 'assistant' &&
        this._isVisibleToAgent(m)
      );
    },

    /** Pinned preferences (role=user only, agent-filtered) */
    get pinnedPreferences() {
      return this.memories.filter(m =>
        m.metadata?.pinned &&
        m.metadata?.memory_type === 'preference' &&
        m.metadata?.role !== 'assistant' &&
        this._isVisibleToAgent(m)
      );
    },

    /** Pinned memories that are not fact or preference (role=user only, agent-filtered) */
    get pinnedOther() {
      return this.memories.filter(m =>
        m.metadata?.pinned &&
        m.metadata?.memory_type !== 'fact' &&
        m.metadata?.memory_type !== 'preference' &&
        m.metadata?.role !== 'assistant' &&
        this._isVisibleToAgent(m)
      );
    },

    /** Recent non-pinned memories of episodic/context/procedural type, within recentDays */
    get recentMemories() {
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - this.recentDays);
      return this.memories.filter(m => {
        if (m.metadata?.pinned) return false;
        if (m.metadata?.role === 'assistant') return false;
        if (!this._isVisibleToAgent(m)) return false;
        if (!this._recentTypes.includes(m.metadata?.memory_type)) return false;
        const created = m.metadata?.created_at_utc;
        if (!created) return false;
        return new Date(created) >= cutoff;
      }).sort((a, b) => {
        const da = a.metadata?.created_at_utc || '';
        const db = b.metadata?.created_at_utc || '';
        return db.localeCompare(da);
      });
    },

    /** Whether any section has content */
    get hasContent() {
      return this.pinnedFacts.length > 0 ||
        this.pinnedPreferences.length > 0 ||
        this.pinnedOther.length > 0 ||
        this.recentMemories.length > 0;
    },

    // ── Interactions ──────────────────────────────────────────

    toggleExpand(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    openEdit(mem) {
      Alpine.store('memoryEdit').show(mem, (payload) => {
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

    openArtifacts(mem) {
      Alpine.store('artifactMgr').show(mem);
    },

    async deleteMemory(id) {
      try {
        await MnemoryAPI.deleteMemory(id);
        this.memories = this.memories.filter(m => m.id !== id);
        this.deleteConfirm = null;
        Alpine.store('notify').success('Memory deleted');
      } catch (err) {
        this.deleteConfirm = null;
        Alpine.store('notify').error(`Failed to delete: ${err.message}`);
      }
    },

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
