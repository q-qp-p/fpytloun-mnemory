/**
 * mnemory UI — Memory Check (fsck) Tab (Alpine.js component).
 *
 * Runs a memory consistency check, displays issues grouped by type
 * with full reasoning, Browse-tab-style memory cards, and clear
 * action cards with metadata change display.
 *
 * Usage:
 *   <div x-data="fsckTab()" x-init="init()"> ... </div>
 */

/* eslint-disable no-unused-vars */

// ── Constants ──────────────────────────────────────────────────────

/** Issue type display order (most critical first). */
const FSCK_GROUP_ORDER = ['security', 'contradiction', 'duplicate', 'quality', 'split', 'reclassify'];

/** Severity sort weight (lower = higher priority). */
const FSCK_SEVERITY_WEIGHT = { high: 0, medium: 1, low: 2 };

/** Group metadata keyed by issue type. */
const FSCK_GROUP_META = {
  security:      { label: 'Security Issues',  icon: '\u26D4', borderCls: 'border-red-500/40',    headerCls: 'text-red-400' },
  contradiction: { label: 'Contradictions',    icon: '\u26A0', borderCls: 'border-red-500/30',    headerCls: 'text-red-300' },
  duplicate:     { label: 'Duplicates',        icon: '\u229C', borderCls: 'border-amber-500/30',  headerCls: 'text-amber-300' },
  quality:       { label: 'Quality Issues',    icon: '\u270E', borderCls: 'border-blue-500/30',   headerCls: 'text-blue-300' },
  split:         { label: 'Should Be Split',   icon: '\u2702', borderCls: 'border-purple-500/30', headerCls: 'text-purple-300' },
  reclassify:    { label: 'Misclassified',     icon: '\u2699', borderCls: 'border-teal-500/30',   headerCls: 'text-teal-300' },
};

// ── Component ──────────────────────────────────────────────────────

function fsckTab() {
  return {
    // ── State ──────────────────────────────────────────────────
    initialized: false,
    loading: false,
    applying: false,

    // Check state
    checkId: null,
    status: null,       // null, 'running', 'completed', 'failed'
    progress: null,
    summary: null,
    issues: [],
    error: null,
    createdAt: null,
    expiresAt: null,

    // Polling
    pollTimer: null,

    // Filters for starting a check
    filters: {
      agent_id: '',
      memory_type: '',
      categories: [],
    },

    // Selection state
    selectedIssues: {},  // issue_id -> boolean

    // UI expansion state
    collapsedGroups: {},      // type -> boolean (true = collapsed)
    expandedMemories: {},     // memory id -> boolean (true = show detail grid)

    // Available filter options
    availableCategories: [],
    availableAgentIds: [],

    // ── Lifecycle ──────────────────────────────────────────────

    init() {
      window.addEventListener('mnemory:tab-changed', (e) => {
        if (e.detail.tab === 'check' && !this.initialized) {
          this.initialized = true;
          this.loadCategories();
          this._loadAgentIds();
        }
      });

      window.addEventListener('mnemory:user-changed', () => {
        if (this.initialized) {
          this.reset();
          this._loadAgentIds();
        }
      });
    },

    async loadCategories() {
      try {
        const data = await MnemoryAPI.categories();
        this.availableCategories = (data.categories || []).map((c) => c.name);
      } catch (err) {
        console.warn('Failed to load categories:', err);
      }
    },

    async _loadAgentIds() {
      try {
        const data = await MnemoryAPI.stats();
        const agents = data.agents || [];
        this.availableAgentIds = agents.map((a) => a.agent_id).filter(Boolean);
      } catch { /* ignore */ }
    },

    // ── Actions ───────────────────────────────────────────────

    async startCheck() {
      this.reset();
      this.loading = true;
      this.status = 'running';

      try {
        const params = {};
        if (this.filters.agent_id) params.agent_id = this.filters.agent_id;
        if (this.filters.memory_type) params.memory_type = this.filters.memory_type;
        if (this.filters.categories.length > 0) params.categories = this.filters.categories;

        const data = await MnemoryAPI.startFsck(params);
        this.checkId = data.check_id;
        this.startPolling();
      } catch (err) {
        this.status = 'failed';
        this.error = err.message;
        this.loading = false;
        Alpine.store('notify').error(`Failed to start check: ${err.message}`);
      }
    },

    startPolling() {
      this.stopPolling();
      this.pollTimer = setInterval(() => this.pollStatus(), 2000);
      this.pollStatus();
    },

    stopPolling() {
      if (this.pollTimer) {
        clearInterval(this.pollTimer);
        this.pollTimer = null;
      }
    },

    async pollStatus() {
      if (!this.checkId) return;

      try {
        const data = await MnemoryAPI.getFsckStatus(this.checkId);
        this.status = data.status;
        this.progress = data.progress;
        this.summary = data.summary;
        this.error = data.error;
        this.createdAt = data.created_at;
        this.expiresAt = data.expires_at;

        if (data.status === 'completed') {
          this.issues = data.issues || [];
          this.stopPolling();
          this.loading = false;
          this.selectAll();
        } else if (data.status === 'failed') {
          this.stopPolling();
          this.loading = false;
        }
      } catch (err) {
        this.stopPolling();
        this.loading = false;
        this.status = 'failed';
        this.error = err.message;
      }
    },

    async applySelected() {
      const ids = Object.entries(this.selectedIssues)
        .filter(([_, v]) => v)
        .map(([k]) => k);

      if (ids.length === 0) {
        Alpine.store('notify').warning('No issues selected');
        return;
      }

      this.applying = true;
      try {
        const data = await MnemoryAPI.applyFsck(this.checkId, ids);
        const msg = `Applied ${data.applied} fixes` +
          (data.failed > 0 ? `, ${data.failed} failed` : '');
        Alpine.store('notify').success(msg);

        const appliedIds = new Set(
          (data.details || [])
            .filter((d) => d.status === 'applied')
            .map((d) => d.issue_id)
        );
        this.issues = this.issues.filter((i) => !appliedIds.has(i.issue_id));
        for (const id of appliedIds) {
          delete this.selectedIssues[id];
        }
        this._recalcSummary();
      } catch (err) {
        Alpine.store('notify').error(`Apply failed: ${err.message}`);
      } finally {
        this.applying = false;
      }
    },

    async applyAll() {
      this.selectAll();
      await this.applySelected();
    },

    reset() {
      this.stopPolling();
      this.checkId = null;
      this.status = null;
      this.progress = null;
      this.summary = null;
      this.issues = [];
      this.error = null;
      this.createdAt = null;
      this.expiresAt = null;
      this.selectedIssues = {};
      this.collapsedGroups = {};
      this.expandedMemories = {};
      this.loading = false;
      this.applying = false;
    },

    // ── Selection ─────────────────────────────────────────────

    selectAll() {
      for (const issue of this.issues) {
        this.selectedIssues[issue.issue_id] = true;
      }
    },

    deselectAll() {
      this.selectedIssues = {};
    },

    get selectedCount() {
      return Object.values(this.selectedIssues).filter(Boolean).length;
    },

    toggleIssue(issueId) {
      this.selectedIssues[issueId] = !this.selectedIssues[issueId];
    },

    /** Select/deselect all issues within a specific group type. */
    toggleGroupSelection(type, select) {
      for (const issue of this.issues) {
        if (issue.type === type) {
          this.selectedIssues[issue.issue_id] = select;
        }
      }
    },

    /** Count selected issues within a group type. */
    groupSelectedCount(type) {
      return this.issues
        .filter((i) => i.type === type && this.selectedIssues[i.issue_id])
        .length;
    },

    // ── Computed ──────────────────────────────────────────────

    get progressPercent() {
      return this.progress ? this.progress.percent : 0;
    },

    get phaseLabel() {
      if (!this.progress) return '';
      const labels = {
        starting: 'Starting...',
        security_scan: 'Phase 1/3 \u2014 Security scan',
        duplicate_search: 'Phase 2/3 \u2014 Duplicate detection (searching)',
        duplicate_eval: 'Phase 2/3 \u2014 Duplicate detection (evaluating clusters)',
        quality_check: 'Phase 3/3 \u2014 Quality check',
        done: 'Complete',
      };
      return labels[this.progress.phase] || this.progress.phase;
    },

    get expiresIn() {
      if (!this.expiresAt) return '';
      const exp = new Date(this.expiresAt);
      const now = new Date();
      const diff = Math.max(0, Math.floor((exp - now) / 1000 / 60));
      if (diff <= 0) return 'expired';
      return `${diff}m remaining`;
    },

    get hasIssues() {
      return this.issues.length > 0;
    },

    /**
     * Issues grouped by type, ordered by FSCK_GROUP_ORDER.
     * Each group: { type, label, icon, borderCls, headerCls, issues: [...] }
     * Issues within each group are sorted by severity (high > medium > low).
     * Only groups with at least one issue are included.
     */
    get groupedIssues() {
      const byType = {};
      for (const issue of this.issues) {
        if (!byType[issue.type]) byType[issue.type] = [];
        byType[issue.type].push(issue);
      }

      const groups = [];
      for (const type of FSCK_GROUP_ORDER) {
        const issues = byType[type];
        if (!issues || issues.length === 0) continue;

        // Sort by severity within group
        issues.sort((a, b) =>
          (FSCK_SEVERITY_WEIGHT[a.severity] ?? 1) - (FSCK_SEVERITY_WEIGHT[b.severity] ?? 1)
        );

        const meta = FSCK_GROUP_META[type] || {};
        groups.push({
          type,
          label: meta.label || type,
          icon: meta.icon || '\u2022',
          borderCls: meta.borderCls || 'border-brand-border',
          headerCls: meta.headerCls || 'text-secondary',
          issues,
        });
      }
      return groups;
    },

    // ── Group Collapse ────────────────────────────────────────

    toggleGroup(type) {
      this.collapsedGroups[type] = !this.collapsedGroups[type];
    },

    isGroupCollapsed(type) {
      return !!this.collapsedGroups[type];
    },

    // ── Memory Detail Expansion ───────────────────────────────

    toggleMemoryDetail(memId) {
      this.expandedMemories[memId] = !this.expandedMemories[memId];
    },

    isMemoryExpanded(memId) {
      return !!this.expandedMemories[memId];
    },

    // ── Issue Type Badge (for issue headers) ──────────────────

    issueTypeBadgeClass(type) {
      const classes = {
        duplicate: 'bg-amber-500/20 text-amber-300',
        quality: 'bg-blue-500/20 text-blue-300',
        split: 'bg-purple-500/20 text-purple-300',
        contradiction: 'bg-red-500/20 text-red-300',
        reclassify: 'bg-teal-500/20 text-teal-300',
        security: 'bg-red-600/20 text-red-400',
      };
      return classes[type] || 'bg-slate-500/20 text-slate-300';
    },

    severityBadgeClass(severity) {
      const classes = {
        low: 'bg-slate-500/20 text-slate-300',
        medium: 'bg-amber-500/20 text-amber-300',
        high: 'bg-red-500/20 text-red-300',
      };
      return classes[severity] || 'bg-slate-500/20 text-slate-300';
    },

    // ── Memory Card Badges (matching Browse tab) ──────────────

    memoryTypeBadgeClass(type) {
      const classes = {
        fact: 'badge-fact',
        preference: 'badge-preference',
        episodic: 'badge-episodic',
        procedural: 'badge-procedural',
        context: 'badge-context',
      };
      return classes[type] || 'badge-context';
    },

    memoryImportanceBadgeClass(importance) {
      const classes = {
        low: 'badge-low',
        normal: 'badge-normal',
        high: 'badge-high',
        critical: 'badge-critical',
      };
      return classes[importance] || 'badge-normal';
    },

    // ── Action Helpers ────────────────────────────────────────

    /** Check if an action is a no-op update (no content change, no metadata change). */
    isNoopAction(action) {
      if (action.action !== 'update') return false;
      if (action.new_content) return false;
      if (!action.new_metadata) return true;
      // Check if all metadata fields are null
      return Object.values(action.new_metadata).every((v) => v === null || v === undefined);
    },

    /** Return actions for an issue with no-op updates filtered out. */
    filteredActions(issue) {
      return (issue.actions || []).filter((a) => !this.isNoopAction(a));
    },

    /** Action card border class (colored left border). */
    actionBorderClass(action) {
      const classes = {
        delete: 'border-l-4 border-l-red-500/60',
        update: 'border-l-4 border-l-blue-500/60',
        add: 'border-l-4 border-l-green-500/60',
      };
      return classes[action] || '';
    },

    actionLabelClass(action) {
      const classes = {
        delete: 'text-red-400',
        update: 'text-blue-400',
        add: 'text-green-400',
      };
      return classes[action] || 'text-secondary';
    },

    actionLabelText(action) {
      const labels = {
        update: 'UPDATE',
        delete: 'DELETE',
        add: 'ADD',
      };
      return labels[action] || action.toUpperCase();
    },

    actionIcon(action) {
      const icons = {
        delete: '\u2716',  // heavy multiplication X
        update: '\u270E',  // pencil
        add: '\u002B',     // plus
      };
      return icons[action] || '\u2022';
    },

    /**
     * Extract non-null metadata changes from an action's new_metadata.
     * Returns array of { field, label, value, type } objects.
     * type is 'badge-type', 'badge-importance', 'categories', 'boolean'.
     */
    metadataChanges(newMetadata) {
      if (!newMetadata) return [];
      const changes = [];

      if (newMetadata.memory_type !== null && newMetadata.memory_type !== undefined) {
        changes.push({
          field: 'memory_type',
          label: 'Type',
          value: newMetadata.memory_type,
          type: 'badge-type',
        });
      }
      if (newMetadata.importance !== null && newMetadata.importance !== undefined) {
        changes.push({
          field: 'importance',
          label: 'Importance',
          value: newMetadata.importance,
          type: 'badge-importance',
        });
      }
      if (newMetadata.categories !== null && newMetadata.categories !== undefined) {
        changes.push({
          field: 'categories',
          label: 'Categories',
          value: newMetadata.categories,
          type: 'categories',
        });
      }
      if (newMetadata.pinned !== null && newMetadata.pinned !== undefined) {
        changes.push({
          field: 'pinned',
          label: 'Pinned',
          value: newMetadata.pinned,
          type: 'boolean',
        });
      }

      return changes;
    },

    // ── Display Helpers ───────────────────────────────────────

    typeIcon(type) {
      return (FSCK_GROUP_META[type] || {}).icon || '\u2022';
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

    summaryCards() {
      if (!this.summary) return [];
      const cards = [];
      if (this.summary.security > 0) cards.push({ type: 'security', count: this.summary.security, label: 'Security', cls: 'border-red-500/30 bg-red-500/10' });
      if (this.summary.duplicate > 0) cards.push({ type: 'duplicate', count: this.summary.duplicate, label: 'Duplicates', cls: 'border-amber-500/30 bg-amber-500/10' });
      if (this.summary.quality > 0) cards.push({ type: 'quality', count: this.summary.quality, label: 'Quality', cls: 'border-blue-500/30 bg-blue-500/10' });
      if (this.summary.split > 0) cards.push({ type: 'split', count: this.summary.split, label: 'Split', cls: 'border-purple-500/30 bg-purple-500/10' });
      if (this.summary.contradiction > 0) cards.push({ type: 'contradiction', count: this.summary.contradiction, label: 'Contradictions', cls: 'border-red-500/30 bg-red-500/10' });
      if (this.summary.reclassify > 0) cards.push({ type: 'reclassify', count: this.summary.reclassify, label: 'Reclassify', cls: 'border-teal-500/30 bg-teal-500/10' });
      return cards;
    },

    _recalcSummary() {
      if (!this.summary) return;
      const counts = { duplicate: 0, quality: 0, split: 0, contradiction: 0, reclassify: 0, security: 0 };
      for (const issue of this.issues) {
        if (counts[issue.type] !== undefined) counts[issue.type]++;
      }
      this.summary = { ...counts, total: this.issues.length };
    },
  };
}
