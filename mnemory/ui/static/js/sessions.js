/**
 * mnemory UI — Sessions panel component.
 *
 * Lists persistent session summaries with filtering and expandable details.
 * Linked memories are loaded on expand and shown as full memory cards
 * with edit/delete/artifact actions (same as Memories tab).
 */

function sessionsPanel() {
  return {
    sessions: [],
    loading: false,
    error: '',
    stateFilter: '',

    // Per-session state for memory loading and expansion
    sessionMemories: {},   // session_id → memory[]
    sessionMemLoading: {}, // session_id → bool
    expandedMemId: null,   // currently expanded memory ID within sessions
    deleteConfirm: null,

    // Per-session consolidated memories
    sessionConsolidatedMems: {},
    sessionConsolidatedLoading: {},
    consolidating: {},

    async load() {
      this.loading = true;
      this.error = '';
      try {
        const params = {};
        if (this.stateFilter) {
          params.consolidation_state = this.stateFilter;
        }
        const data = await MnemoryAPI.get('/sessions', params);
        this.sessions = (data.sessions || []).map(s => ({ ...s, _expanded: false }));
      } catch (e) {
        this.error = e.message || 'Failed to load sessions';
      } finally {
        this.loading = false;
      }
    },

    async toggleSession(session) {
      session._expanded = !session._expanded;
      if (session._expanded) {
        // Load raw memories on first expand
        if (session.memory_ids?.length > 0 && !this.sessionMemories[session.session_id]) {
          await this.loadSessionMemories(session);
        }
        // Load consolidated memories on first expand (if consolidated)
        if (session.consolidated_memory_ids?.length > 0 && !this.sessionConsolidatedMems[session.session_id]) {
          await this.loadConsolidatedMemories(session);
        }
      }
    },

    async loadSessionMemories(session) {
      const sid = session.session_id;
      this.sessionMemLoading[sid] = true;
      try {
        // Fetch all memories and filter by session's memory_ids
        const data = await MnemoryAPI.listMemories({ limit: 5000 });
        const allMems = data.results || [];
        const idSet = new Set(session.memory_ids || []);
        this.sessionMemories[sid] = allMems.filter(m => idSet.has(m.id));
      } catch (e) {
        Alpine.store('notify').error(`Failed to load session memories: ${e.message}`);
        this.sessionMemories[sid] = [];
      } finally {
        this.sessionMemLoading[sid] = false;
      }
    },

    getSessionMemories(sessionId) {
      return this.sessionMemories[sessionId] || [];
    },

    isMemLoading(sessionId) {
      return !!this.sessionMemLoading[sessionId];
    },

    toggleMemExpand(id) {
      this.expandedMemId = this.expandedMemId === id ? null : id;
    },

    async loadConsolidatedMemories(session) {
      const sid = session.session_id;
      const cids = session.consolidated_memory_ids || [];
      if (cids.length === 0) return;
      this.sessionConsolidatedLoading[sid] = true;
      try {
        const data = await MnemoryAPI.listMemories({ limit: 5000 });
        const allMems = data.results || [];
        const idSet = new Set(cids);
        this.sessionConsolidatedMems[sid] = allMems.filter(m => idSet.has(m.id));
      } catch (e) {
        Alpine.store('notify').error(`Failed to load consolidated memories: ${e.message}`);
        this.sessionConsolidatedMems[sid] = [];
      } finally {
        this.sessionConsolidatedLoading[sid] = false;
      }
    },

    getConsolidatedMemories(sessionId) {
      return this.sessionConsolidatedMems[sessionId] || [];
    },

    isConsolidatedLoading(sessionId) {
      return !!this.sessionConsolidatedLoading[sessionId];
    },

    async consolidateSession(session) {
      const sid = session.session_id;
      this.consolidating[sid] = true;
      try {
        const result = await MnemoryAPI.post(`/sessions/${sid}/consolidate`);
        Alpine.store('notify').success(
          `Consolidated: ${result.memories_produced} memories produced, ${result.memories_superseded} superseded`
        );
        // Reload session data
        session.consolidation_state = result.state;
        session.consolidated_at = new Date().toISOString();
        // Reload raw memories for this session (to show superseded badges)
        await this.loadSessionMemories(session);
        if (result.consolidated_memory_ids?.length > 0) {
          // Update session with the produced consolidated memory IDs
          session.consolidated_memory_ids = result.consolidated_memory_ids;
          await this.loadConsolidatedMemories(session);
        }
      } catch (e) {
        const msg = e.message || 'Consolidation failed';
        Alpine.store('notify').error(msg);
      } finally {
        this.consolidating[sid] = false;
      }
    },

    // ── Memory actions (delegate to global stores) ────────────

    openEdit(mem, sessionId) {
      Alpine.store('memoryEdit').show(mem, (payload) => {
        // Update local memory object
        const mems = this.sessionMemories[sessionId] || [];
        const m = mems.find(x => x.id === mem.id);
        if (m) {
          if (payload.content !== undefined) m.memory = payload.content;
          if (!m.metadata) m.metadata = {};
          if (payload.memory_type) m.metadata.memory_type = payload.memory_type;
          if (payload.categories) m.metadata.categories = payload.categories;
          if (payload.importance) m.metadata.importance = payload.importance;
          if (payload.pinned !== undefined) m.metadata.pinned = payload.pinned;
          if (payload.ttl_days !== undefined) m.metadata.ttl_days = payload.ttl_days;
          if (payload.agent_id !== undefined) m.metadata.agent_id = payload.agent_id || null;
          if ('labels' in payload) m.metadata.labels = payload.labels;
        }
      });
    },

    openArtifacts(mem) {
      Alpine.store('artifactMgr').show(mem);
    },

    async deleteMemory(id, sessionId) {
      try {
        await MnemoryAPI.deleteMemory(id);
        const mems = this.sessionMemories[sessionId] || [];
        this.sessionMemories[sessionId] = mems.filter(m => m.id !== id);
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

    // ── Display helpers (same as Memories tab) ────────────────

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

    stateBadgeClass(state) {
      return {
        'bg-yellow-500/20 text-yellow-300': state === 'idle',
        'bg-blue-500/20 text-blue-300': state === 'consolidating',
        'bg-green-500/20 text-green-300': state === 'consolidated',
        'bg-red-500/20 text-red-300': state === 'failed',
      };
    },
  };
}
