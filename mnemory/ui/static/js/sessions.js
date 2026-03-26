/**
 * mnemory UI — Sessions panel component.
 *
 * Lists persistent session summaries with filtering and expandable details.
 * Linked memories are loaded on expand via POST /memories/by-ids (batch)
 * and displayed with client-side "Show More / Show Less" pagination.
 */

function sessionsPanel() {
  const MEM_PAGE_SIZE = 10;

  return {
    sessions: [],
    loading: false,
    error: '',
    stateFilter: '',
    agentFilter: '',

    // Per-session state for memory loading and expansion
    sessionMemories: {},   // session_id → memory[]
    sessionMemLoading: {}, // session_id → bool
    expandedMemId: null,   // currently expanded memory ID within sessions
    deleteConfirm: null,

    // Per-session consolidated memories
    sessionConsolidatedMems: {},
    sessionConsolidatedLoading: {},
    consolidating: {},

    // Per-session pagination state (page number, 1-indexed)
    sessionMemPages: {},
    sessionConsolidatedPages: {},

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

    get filteredSessions() {
      let arr = this.sessions;
      if (this.agentFilter) {
        arr = arr.filter(s => (s.agent_id || '') === this.agentFilter);
      }
      return arr;
    },

    get uniqueAgents() {
      const agents = new Set();
      for (const s of this.sessions) {
        if (s.agent_id) agents.add(s.agent_id);
      }
      return [...agents].sort();
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

    /**
     * Fetch memories by IDs, chunking into batches of 500 to respect
     * the server's max_length limit on the by-ids endpoint.
     */
    async _fetchMemoriesByIds(ids) {
      const CHUNK = 500;
      if (ids.length <= CHUNK) {
        const data = await MnemoryAPI.getMemoriesByIds(ids);
        return data.results || [];
      }
      const all = [];
      for (let i = 0; i < ids.length; i += CHUNK) {
        const data = await MnemoryAPI.getMemoriesByIds(ids.slice(i, i + CHUNK));
        all.push(...(data.results || []));
      }
      return all;
    },

    async loadSessionMemories(session) {
      const sid = session.session_id;
      const ids = session.memory_ids || [];
      if (ids.length === 0) return;
      this.sessionMemLoading[sid] = true;
      this.sessionMemPages[sid] = 1;
      try {
        this.sessionMemories[sid] = await this._fetchMemoriesByIds(ids);
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

    /** Return paginated slice of raw memories for display. */
    getPagedSessionMemories(sessionId) {
      const all = this.sessionMemories[sessionId] || [];
      const page = this.sessionMemPages[sessionId] || 1;
      return all.slice(0, page * MEM_PAGE_SIZE);
    },

    canShowMoreMems(sessionId) {
      const all = this.sessionMemories[sessionId] || [];
      const page = this.sessionMemPages[sessionId] || 1;
      return page * MEM_PAGE_SIZE < all.length;
    },

    remainingMemCount(sessionId) {
      const all = this.sessionMemories[sessionId] || [];
      const page = this.sessionMemPages[sessionId] || 1;
      return Math.max(0, all.length - page * MEM_PAGE_SIZE);
    },

    showMoreMems(sessionId) {
      this.sessionMemPages[sessionId] = (this.sessionMemPages[sessionId] || 1) + 1;
    },

    showLessMems(sessionId) {
      this.sessionMemPages[sessionId] = 1;
    },

    canShowLessMems(sessionId) {
      return (this.sessionMemPages[sessionId] || 1) > 1;
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
      this.sessionConsolidatedPages[sid] = 1;
      try {
        this.sessionConsolidatedMems[sid] = await this._fetchMemoriesByIds(cids);
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

    /** Return paginated slice of consolidated memories for display. */
    getPagedConsolidatedMemories(sessionId) {
      const all = this.sessionConsolidatedMems[sessionId] || [];
      const page = this.sessionConsolidatedPages[sessionId] || 1;
      return all.slice(0, page * MEM_PAGE_SIZE);
    },

    canShowMoreConsolidated(sessionId) {
      const all = this.sessionConsolidatedMems[sessionId] || [];
      const page = this.sessionConsolidatedPages[sessionId] || 1;
      return page * MEM_PAGE_SIZE < all.length;
    },

    remainingConsolidatedCount(sessionId) {
      const all = this.sessionConsolidatedMems[sessionId] || [];
      const page = this.sessionConsolidatedPages[sessionId] || 1;
      return Math.max(0, all.length - page * MEM_PAGE_SIZE);
    },

    showMoreConsolidated(sessionId) {
      this.sessionConsolidatedPages[sessionId] = (this.sessionConsolidatedPages[sessionId] || 1) + 1;
    },

    showLessConsolidated(sessionId) {
      this.sessionConsolidatedPages[sessionId] = 1;
    },

    canShowLessConsolidated(sessionId) {
      return (this.sessionConsolidatedPages[sessionId] || 1) > 1;
    },

    isConsolidatedLoading(sessionId) {
      return !!this.sessionConsolidatedLoading[sessionId];
    },

    isConsolidating(sessionId) {
      return !!this.consolidating[sessionId];
    },

    async consolidateSession(session) {
      const sid = session.session_id;
      this.consolidating[sid] = true;
      try {
        await MnemoryAPI.post(`/sessions/${sid}/consolidate`);
        session.consolidation_state = 'consolidating';
        Alpine.store('notify').success('Consolidation started...');
        // Poll for completion
        this._pollConsolidation(session);
      } catch (e) {
        const msg = e.message || 'Consolidation failed';
        Alpine.store('notify').error(msg);
        this.consolidating[sid] = false;
      }
    },

    _pollConsolidation(session) {
      const sid = session.session_id;
      const poll = setInterval(async () => {
        try {
          const data = await MnemoryAPI.get(`/sessions/${sid}`);
          if (data.consolidation_state !== 'consolidating') {
            clearInterval(poll);
            this.consolidating[sid] = false;
            session.consolidation_state = data.consolidation_state;
            session.consolidated_at = data.consolidated_at;
            session.consolidated_memory_ids = data.consolidated_memory_ids;
            if (data.consolidation_state === 'consolidated') {
              Alpine.store('notify').success('Consolidation complete');
              // Invalidate caches and reload
              delete this.sessionMemories[sid];
              delete this.sessionConsolidatedMems[sid];
              delete this.sessionMemPages[sid];
              delete this.sessionConsolidatedPages[sid];
              if (session._expanded) {
                await this.loadSessionMemories(session);
                if (data.consolidated_memory_ids?.length > 0) {
                  await this.loadConsolidatedMemories(session);
                }
              }
            } else {
              Alpine.store('notify').error('Consolidation failed');
            }
          }
        } catch (e) {
          clearInterval(poll);
          this.consolidating[sid] = false;
        }
      }, 3000); // Poll every 3 seconds
    },

    // ── Session delete ──────────────────────────────────────────

    confirmDeleteSession(session) {
      Alpine.store('sessionDelete').show(session, (sid) => {
        // Remove from local list after successful deletion
        this.sessions = this.sessions.filter(s => s.session_id !== sid);
        delete this.sessionMemories[sid];
        delete this.sessionConsolidatedMems[sid];
        delete this.sessionMemPages[sid];
        delete this.sessionConsolidatedPages[sid];
      });
    },

    // ── Memory actions (delegate to global stores) ────────────

    openEdit(mem, sessionId) {
      Alpine.store('memoryEdit').show(mem, (payload) => {
        // Update local memory object in both raw and consolidated caches
        const _apply = (m) => {
          if (payload.content !== undefined) m.memory = payload.content;
          if (!m.metadata) m.metadata = {};
          if (payload.memory_type) m.metadata.memory_type = payload.memory_type;
          if (payload.categories) m.metadata.categories = payload.categories;
          if (payload.importance) m.metadata.importance = payload.importance;
          if (payload.pinned !== undefined) m.metadata.pinned = payload.pinned;
          if (payload.ttl_days !== undefined) m.metadata.ttl_days = payload.ttl_days;
          if (payload.agent_id !== undefined) m.metadata.agent_id = payload.agent_id || null;
          if ('labels' in payload) m.metadata.labels = payload.labels;
        };
        const raw = (this.sessionMemories[sessionId] || []).find(x => x.id === mem.id);
        if (raw) _apply(raw);
        const cons = (this.sessionConsolidatedMems[sessionId] || []).find(x => x.id === mem.id);
        if (cons) _apply(cons);
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
