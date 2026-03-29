/**
 * mnemory UI — User tab Alpine.js component.
 *
 * Shows the exact default core context returned by GET /api/memories/core.
 * Text mode previews the injected context as markdown-like text. Advanced
 * mode loads the underlying memory objects and renders them as interactive
 * cards for cleanup and inspection.
 */

// Known core-memory headings currently emitted by MemoryService.get_core_memories():
// - ## Agent Identity
// - ## Agent Knowledge
// - ## Agent Instructions
// - ## User Facts
// - ## User Preferences
// - ## Other User Memories
// - ## Recent Context
// - ### User Activity
// - ### Agent Activity

function userTab() {
  return {
    loading: false,
    advancedLoading: false,
    initialized: false,
    recentDays: 7,
    filterAgentId: '',
    availableAgentIds: [],
    rawCoreText: '',
    stats: null,
    parsedCore: { preamble: [], sections: [] },
    fallbackRawText: '',
    advancedMode: false,
    advancedSections: [],
    expandedId: null,
    deleteConfirm: null,
    advancedMissingCount: 0,
    _refreshTimer: null,

    init() {
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'user' && !this.initialized) {
          this.initialized = true;
          this._loadAgentIds().finally(() => this.loadData());
        }
      });

      window.addEventListener('mnemory:user-changed', async () => {
        if (this.initialized) {
          this.filterAgentId = '';
          this.advancedMode = false;
          await this._loadAgentIds();
          this.loadData();
        }
      });
    },

    async loadData() {
      this.loading = true;
      this.fallbackRawText = '';
      this.parsedCore = { preamble: [], sections: [] };
      this.stats = null;

      try {
        const data = await MnemoryAPI.getCoreMemories(
          { recent_days: this.recentDays, include_stats: true },
          this.filterAgentId,
        );
        this.rawCoreText = String(data?.text || '');
        this.stats = data?.stats || null;
        this._parseCoreText();
        if (this.advancedMode) {
          await this.loadAdvancedData();
        }
      } catch (err) {
        Alpine.store('notify').error(`Failed to load core memories: ${err.message}`);
        this.rawCoreText = '';
        this.stats = null;
        this.parsedCore = { preamble: [], sections: [] };
        this.fallbackRawText = '';
        this.advancedSections = [];
        this.advancedMissingCount = 0;
      } finally {
        this.loading = false;
      }
    },

    async loadAdvancedData() {
      if (!this.stats?.memory_ids?.length) {
        this.advancedSections = [];
        this.advancedMissingCount = 0;
        return;
      }

      this.advancedLoading = true;
      try {
        const memories = await this._fetchMemoriesByIds(this.stats.memory_ids);
        const lookup = new Map(memories.map((memory) => [memory.id, memory]));
        let missingCount = 0;
        const sections = [];

        for (const [sectionKey, label] of Object.entries(this.stats.section_labels || {})) {
          const ids = this.stats.sections?.[sectionKey] || [];
          const sectionMemories = ids.map((id) => lookup.get(id)).filter(Boolean);
          missingCount += ids.length - sectionMemories.length;
          if (sectionMemories.length > 0) {
            sections.push({
              key: sectionKey,
              label,
              memories: sectionMemories,
            });
          }
        }

        this.advancedSections = sections;
        this.advancedMissingCount = missingCount;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load core memory details: ${err.message}`);
        this.advancedSections = [];
        this.advancedMissingCount = 0;
      } finally {
        this.advancedLoading = false;
      }
    },

    async _fetchMemoriesByIds(ids) {
      const chunkSize = 500;
      const all = [];
      for (let i = 0; i < ids.length; i += chunkSize) {
        const data = await MnemoryAPI.getMemoriesByIds(ids.slice(i, i + chunkSize));
        all.push(...(data.results || []));
      }
      return all;
    },

    async _loadAgentIds() {
      try {
        const data = await MnemoryAPI.listMemories({
          limit: 5000,
          include_decayed: false,
          sort: 'newest',
        });
        const ids = new Set();
        for (const memory of data.results || []) {
          const agentId = memory.metadata?.agent_id;
          if (agentId) ids.add(agentId);
        }
        this.availableAgentIds = [...ids];
      } catch (err) {
        console.warn('Failed to load user-scoped agent IDs:', err);
        this.availableAgentIds = [];
      }
    },

    _parseCoreText() {
      const text = this.rawCoreText.trim();
      if (!text || text === 'No core memories found.') {
        this.parsedCore = { preamble: [], sections: [] };
        this.fallbackRawText = '';
        return;
      }

      const result = { preamble: [], sections: [] };
      let currentSection = null;
      let currentSubsection = null;

      for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line) continue;

        if (line.startsWith('## ')) {
          currentSection = {
            title: line.slice(3).trim(),
            lines: [],
            subsections: [],
          };
          result.sections.push(currentSection);
          currentSubsection = null;
          continue;
        }

        if (line.startsWith('### ')) {
          if (!currentSection) continue;
          currentSubsection = {
            title: line.slice(4).trim(),
            lines: [],
          };
          currentSection.subsections.push(currentSubsection);
          continue;
        }

        const entry = this._parseCoreLine(line);
        if (!entry.text) continue;

        if (currentSubsection) {
          currentSubsection.lines.push(entry);
        } else if (currentSection) {
          currentSection.lines.push(entry);
        } else {
          result.preamble.push(entry);
        }
      }

      this.parsedCore = result;

      if (result.preamble.length === 0 && result.sections.length === 0) {
        this.fallbackRawText = text;
        console.warn('Core memories parser fell back to raw text:', text.slice(0, 200));
      }
    },

    _parseCoreLine(line) {
      const bullet = line.startsWith('- ');
      const text = bullet ? line.slice(2).trim() : line.trim();
      return {
        bullet,
        text: bullet ? text.replace(/⟨\/?memory_item⟩/gu, '').trim() : text,
      };
    },

    get hasContent() {
      return this.parsedCore.preamble.length > 0 || this.parsedCore.sections.length > 0;
    },

    get hasStats() {
      return !!this.stats;
    },

    get hasAdvancedContent() {
      return this.advancedSections.some((section) => section.memories.length > 0);
    },

    get statsEmpty() {
      return this.stats?.memory_count === 0;
    },

    get emptyState() {
      return this.rawCoreText.trim() || 'No core memories found.';
    },

    get scopeDescription() {
      if (this.filterAgentId) {
        return `Previewing the exact default core context returned for agent ${this.filterAgentId}.`;
      }
      return 'Previewing the shared user core context returned when no specific agent is selected.';
    },

    get agentSourceNote() {
      if (this.availableAgentIds.length === 0) return '';
      return 'Agent options are derived from your newest visible memories.';
    },

    get typeStats() {
      return Object.entries(this.stats?.by_type || {});
    },

    get roleStats() {
      return Object.entries(this.stats?.by_role || {});
    },

    get sectionStats() {
      return Object.entries(this.stats?.by_section || {}).map(([key, count]) => ({
        key,
        label: this.stats?.section_labels?.[key] || key,
        count,
      }));
    },

    get advancedMissingNote() {
      if (this.advancedMissingCount <= 0) return '';
      return `${this.advancedMissingCount} mem${this.advancedMissingCount === 1 ? 'ory is' : 'ories are'} no longer available.`;
    },

    onAgentChange() {
      this.advancedMode = false;
      this.advancedSections = [];
      this.loadData();
    },

    async toggleAdvancedMode() {
      this.advancedMode = !this.advancedMode;
      this.deleteConfirm = null;
      if (this.advancedMode) {
        await this.loadAdvancedData();
      }
    },

    toggleExpand(id) {
      this.expandedId = this.expandedId === id ? null : id;
    },

    openEdit(mem) {
      Alpine.store('memoryEdit').show(mem, (payload) => {
        this._applyOptimisticEdit(mem.id, payload);
        this.scheduleRefresh();
      });
    },

    openArtifacts(mem) {
      Alpine.store('artifactMgr').show(mem);
    },

    async deleteMemory(id) {
      try {
        await MnemoryAPI.deleteMemory(id);
        this._applyOptimisticDelete(id);
        this.deleteConfirm = null;
        Alpine.store('notify').success('Memory deleted');
        this.scheduleRefresh();
      } catch (err) {
        this.deleteConfirm = null;
        Alpine.store('notify').error(`Failed to delete: ${err.message}`);
      }
    },

    scheduleRefresh() {
      if (this._refreshTimer) {
        clearTimeout(this._refreshTimer);
      }
      this._refreshTimer = setTimeout(() => {
        this._refreshTimer = null;
        this.loadData();
      }, 500);
    },

    _applyOptimisticEdit(id, payload) {
      for (const section of this.advancedSections) {
        const memory = section.memories.find((item) => item.id === id);
        if (!memory) continue;
        if (payload.content !== undefined) memory.memory = payload.content;
        if (!memory.metadata) memory.metadata = {};
        if (payload.memory_type) memory.metadata.memory_type = payload.memory_type;
        if (payload.categories) memory.metadata.categories = payload.categories;
        if (payload.importance) memory.metadata.importance = payload.importance;
        if (payload.pinned !== undefined) memory.metadata.pinned = payload.pinned;
        if (payload.ttl_days !== undefined) memory.metadata.ttl_days = payload.ttl_days;
        if (payload.event_date !== undefined) memory.metadata.event_date = payload.event_date;
        if (payload.agent_id !== undefined) memory.metadata.agent_id = payload.agent_id || null;
        if ('labels' in payload) memory.metadata.labels = payload.labels;
        break;
      }
    },

    _applyOptimisticDelete(id) {
      let removedMemory = null;
      let removedSectionKey = null;

      this.advancedSections = this.advancedSections
        .map((section) => {
          const before = section.memories.length;
          const memories = section.memories.filter((memory) => {
            if (memory.id === id) {
              removedMemory = memory;
              removedSectionKey = section.key;
              return false;
            }
            return true;
          });
          return before === memories.length ? section : { ...section, memories };
        })
        .filter((section) => section.memories.length > 0);

      if (!removedMemory || !this.stats) {
        return;
      }

      this.stats.memory_ids = (this.stats.memory_ids || []).filter((memoryId) => memoryId !== id);
      this.stats.memory_count = Math.max(0, (this.stats.memory_count || 0) - 1);
      if (removedSectionKey && this.stats.sections?.[removedSectionKey]) {
        this.stats.sections[removedSectionKey] = this.stats.sections[removedSectionKey].filter((memoryId) => memoryId !== id);
        const nextCount = Math.max(0, (this.stats.by_section?.[removedSectionKey] || 0) - 1);
        if (nextCount > 0) {
          this.stats.by_section[removedSectionKey] = nextCount;
        } else {
          delete this.stats.by_section[removedSectionKey];
          delete this.stats.sections[removedSectionKey];
          delete this.stats.section_labels[removedSectionKey];
        }
      }

      const memoryType = removedMemory.metadata?.memory_type || 'context';
      const role = removedMemory.metadata?.role || 'user';
      this._decrementStatBucket(this.stats.by_type, memoryType);
      this._decrementStatBucket(this.stats.by_role, role);
    },

    _decrementStatBucket(bucket, key) {
      if (!bucket?.[key]) return;
      if (bucket[key] <= 1) {
        delete bucket[key];
      } else {
        bucket[key] -= 1;
      }
    },

    copyId(id) {
      navigator.clipboard.writeText(id).then(
        () => Alpine.store('notify').success('ID copied to clipboard'),
        () => Alpine.store('notify').error('Failed to copy ID'),
      );
    },

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

    truncate(str, max = 200) {
      if (!str) return '';
      return str.length > max ? str.substring(0, max) + '...' : str;
    },

    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        return new Date(dateStr).toLocaleString(undefined, {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch {
        return dateStr;
      }
    },

    sectionIcon(title) {
      const icons = {
        'Agent Identity': '🧠',
        'Agent Knowledge': '📚',
        'Agent Instructions': '🧭',
        'User Facts': '👤',
        'User Preferences': '⚙️',
        'Other User Memories': '📝',
        'Recent Context': '🕒',
      };
      return icons[title] || '•';
    },

    destroy() {
      if (this._refreshTimer) {
        clearTimeout(this._refreshTimer);
        this._refreshTimer = null;
      }
    },
  };
}
