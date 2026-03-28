/**
 * mnemory UI — User tab Alpine.js component.
 *
 * Shows a read-only preview of the exact default core context returned by
 * GET /api/memories/core. This matches what an agent receives from
 * get_core_memories, including agent-specific sections when an agent is
 * selected.
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
    initialized: false,
    recentDays: 7,
    filterAgentId: '',
    availableAgentIds: [],
    rawCoreText: '',
    parsedCore: { preamble: [], sections: [] },
    fallbackRawText: '',

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
          await this._loadAgentIds();
          this.loadData();
        }
      });
    },

    async loadData() {
      this.loading = true;
      this.fallbackRawText = '';
      this.parsedCore = { preamble: [], sections: [] };

      try {
        const data = await MnemoryAPI.getCoreMemories(
          { recent_days: this.recentDays },
          this.filterAgentId,
        );
        this.rawCoreText = String(data?.text || '');
        this._parseCoreText();
      } catch (err) {
        Alpine.store('notify').error(`Failed to load core memories: ${err.message}`);
        this.rawCoreText = '';
        this.parsedCore = { preamble: [], sections: [] };
        this.fallbackRawText = '';
      } finally {
        this.loading = false;
      }
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

    onAgentChange() {
      this.loadData();
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
  };
}
