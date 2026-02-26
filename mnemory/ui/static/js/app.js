/**
 * mnemory UI — Alpine.js stores and app initialization.
 *
 * Stores:
 * - auth: login state, API key, identity, user switching
 * - nav: active tab navigation
 * - notify: toast notification system
 */

document.addEventListener('alpine:init', () => {

  // ── Auth Store ───────────────────────────────────────────────
  Alpine.store('auth', {
    authenticated: false,
    loading: true,
    error: '',
    identity: null,     // { user_id, agent_id, timezone, can_switch_user }
    users: [],          // available users for switching
    selectedUser: '',   // currently selected user (for switching)

    async init() {
      // Check if we have a stored key
      const key = MnemoryAPI.getKey();
      if (key) {
        await this.verify();
      } else {
        this.loading = false;
      }
    },

    async login(apiKey) {
      this.error = '';
      this.loading = true;
      MnemoryAPI.setKey(apiKey);
      try {
        const identity = await MnemoryAPI.whoami();
        this.identity = identity;
        this.authenticated = true;
        // Load users list if switching is allowed (wildcard key)
        if (identity.can_switch_user) {
          await this.loadUsers();
          // If no user_id bound by key, auto-select first known user
          if (!identity.user_id && this.users.length > 0) {
            this.switchUser(this.users[0]);
          } else {
            this.selectedUser = identity.user_id || '';
            MnemoryAPI.setSelectedUser(''); // no override needed
          }
        } else {
          this.selectedUser = identity.user_id || '';
          MnemoryAPI.setSelectedUser(''); // bound key, no override needed
        }
      } catch (e) {
        MnemoryAPI.clearKey();
        this.authenticated = false;
        this.error = e.message === 'Unauthorized'
          ? 'Invalid API key'
          : `Connection failed: ${e.message}`;
      } finally {
        this.loading = false;
      }
    },

    async verify() {
      this.loading = true;
      try {
        const identity = await MnemoryAPI.whoami();
        this.identity = identity;
        this.authenticated = true;
        if (identity.can_switch_user) {
          await this.loadUsers();
          const stored = MnemoryAPI.getSelectedUser();
          if (stored) {
            this.selectedUser = stored;
          } else if (identity.user_id) {
            this.selectedUser = identity.user_id;
          } else if (this.users.length > 0) {
            this.switchUser(this.users[0]);
          }
        } else {
          this.selectedUser = identity.user_id || '';
        }
      } catch {
        MnemoryAPI.clearKey();
        MnemoryAPI.setSelectedUser('');
        this.authenticated = false;
      } finally {
        this.loading = false;
      }
    },

    async loadUsers() {
      try {
        const stats = await MnemoryAPI.stats();
        this.users = stats.users || [];
      } catch {
        // Non-critical — user list just won't be available
        this.users = [];
      }
    },

    switchUser(userId) {
      this.selectedUser = userId;
      if (this.identity && userId === this.identity.user_id) {
        MnemoryAPI.setSelectedUser(''); // use default
      } else {
        MnemoryAPI.setSelectedUser(userId);
      }
      // Dispatch event so tabs can refresh
      window.dispatchEvent(new CustomEvent('mnemory:user-changed', {
        detail: { userId }
      }));
    },

    logout() {
      MnemoryAPI.clearKey();
      MnemoryAPI.setSelectedUser('');
      this.authenticated = false;
      this.identity = null;
      this.users = [];
      this.selectedUser = '';
      this.error = '';
    },
  });

  // ── Memory Edit Store ────────────────────────────────────────
  // Global edit modal — shared between Memories and Search tabs.
  Alpine.store('memoryEdit', {
    open: false,
    memory: null,   // original memory object
    form: {},       // editable copy
    _onSaved: null, // callback(updatedFields) after successful save

    /**
     * Open the edit modal for a memory.
     * @param {object} memory - The memory object to edit.
     * @param {function} onSaved - Called with updated field map after save.
     */
    show(memory, onSaved) {
      const meta = memory.metadata || {};
      this.memory = memory;
      this._onSaved = onSaved || null;
      this.form = {
        content: memory.memory || '',
        memory_type: meta.memory_type || '',
        categories: (meta.categories || []).join(', '),
        importance: meta.importance || 'normal',
        pinned: !!meta.pinned,
        ttl_days: meta.ttl_days ?? '',
        event_date: meta.event_date ? meta.event_date.slice(0, 10) : '',
        agent_id: meta.agent_id || '',
      };
      this.open = true;
    },

    close() {
      this.open = false;
      this.memory = null;
      this._onSaved = null;
    },

    async save() {
      const form = this.form;
      const memoryId = this.memory.id;
      const meta = this.memory.metadata || {};
      const payload = {};

      if (form.content !== this.memory.memory) payload.content = form.content;
      if (form.memory_type && form.memory_type !== meta.memory_type) payload.memory_type = form.memory_type;

      const newCats = form.categories
        ? form.categories.split(',').map(c => c.trim()).filter(Boolean)
        : [];
      if (JSON.stringify(newCats) !== JSON.stringify(meta.categories || [])) payload.categories = newCats;

      if (form.importance && form.importance !== meta.importance) payload.importance = form.importance;
      if (form.pinned !== !!meta.pinned) payload.pinned = form.pinned;

      if (form.ttl_days !== '' && form.ttl_days !== null) {
        const ttl = parseInt(form.ttl_days, 10);
        if (!isNaN(ttl) && ttl !== meta.ttl_days) payload.ttl_days = ttl;
      }

      const origDate = meta.event_date ? meta.event_date.slice(0, 10) : '';
      if (form.event_date !== origDate) {
        payload.event_date = form.event_date || null;
      }

      // agent_id: empty string = clear, non-empty = set, unchanged = omit
      const origAgent = meta.agent_id || '';
      if (form.agent_id !== origAgent) {
        payload.agent_id = form.agent_id || '';  // '' signals "clear" to the API
      }

      if (Object.keys(payload).length === 0) { this.close(); return; }

      try {
        await MnemoryAPI.updateMemory(memoryId, payload);
        if (this._onSaved) this._onSaved(payload);
        Alpine.store('notify').success('Memory updated');
        this.close();
      } catch (err) {
        Alpine.store('notify').error(`Failed to update: ${err.message}`);
      }
    },
  });

  // ── Artifact Manager Store ───────────────────────────────────
  // Global artifact modal — shared between Memories and Search tabs.
  Alpine.store('artifactMgr', {
    open: false,
    memoryId: null,
    memoryText: '',
    artifacts: [],
    loading: false,
    deleteConfirm: null,

    // View sub-panel
    view: {
      open: false,
      artifact: null,
      content: '',
      hasMore: false,
      offset: 0,
      loading: false,
    },

    // Add artifact sub-form
    addForm: {
      open: false,
      saving: false,
      filename: 'note.md',
      content_type: 'text/markdown',
      content: '',
    },

    /**
     * Open the artifact manager for a memory.
     * @param {object} memory - Memory object with id and memory text.
     */
    show(memory) {
      this.memoryId = memory.id;
      this.memoryText = memory.memory || '';
      this.artifacts = [];
      this.deleteConfirm = null;
      this.view = { open: false, artifact: null, content: '', hasMore: false, offset: 0, loading: false };
      this.addForm = { open: false, saving: false, filename: 'note.md', content_type: 'text/markdown', content: '' };
      this.open = true;
      this.loadArtifacts();
    },

    close() {
      this.open = false;
      this.memoryId = null;
    },

    async loadArtifacts() {
      this.loading = true;
      try {
        const result = await MnemoryAPI.listArtifacts(this.memoryId);
        // API returns array directly or { artifacts: [...] }
        this.artifacts = Array.isArray(result) ? result : (result.artifacts || []);
      } catch (err) {
        Alpine.store('notify').error(`Failed to load artifacts: ${err.message}`);
      } finally {
        this.loading = false;
      }
    },

    async viewArtifact(artifact) {
      this.view = { open: true, artifact, content: '', hasMore: false, offset: 0, loading: true };
      try {
        const result = await MnemoryAPI.getArtifact(this.memoryId, artifact.id, 0, 5000);
        this.view.content = result.content || '';
        this.view.hasMore = result.has_more || false;
        this.view.offset = (result.content || '').length;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load artifact: ${err.message}`);
        this.view.open = false;
      } finally {
        this.view.loading = false;
      }
    },

    async loadMoreContent() {
      if (!this.view.hasMore || this.view.loading) return;
      this.view.loading = true;
      try {
        const result = await MnemoryAPI.getArtifact(this.memoryId, this.view.artifact.id, this.view.offset, 5000);
        this.view.content += result.content || '';
        this.view.hasMore = result.has_more || false;
        this.view.offset += (result.content || '').length;
      } catch (err) {
        Alpine.store('notify').error(`Failed to load more: ${err.message}`);
      } finally {
        this.view.loading = false;
      }
    },

    async saveArtifact() {
      const f = this.addForm;
      if (!f.content.trim()) {
        Alpine.store('notify').error('Content is required');
        return;
      }
      f.saving = true;
      try {
        const result = await MnemoryAPI.saveArtifact(this.memoryId, {
          content: f.content,
          filename: f.filename || 'note.md',
          content_type: f.content_type || 'text/markdown',
        });
        // result is the artifact metadata
        this.artifacts.push(result);
        f.open = false;
        f.content = '';
        f.filename = 'note.md';
        f.content_type = 'text/markdown';
        Alpine.store('notify').success('Artifact saved');
      } catch (err) {
        Alpine.store('notify').error(`Failed to save artifact: ${err.message}`);
      } finally {
        f.saving = false;
      }
    },

    async deleteArtifact(artifactId) {
      try {
        await MnemoryAPI.deleteArtifact(this.memoryId, artifactId);
        this.artifacts = this.artifacts.filter(a => a.id !== artifactId);
        this.deleteConfirm = null;
        // Close view panel if we just deleted the viewed artifact
        if (this.view.artifact?.id === artifactId) {
          this.view.open = false;
        }
        Alpine.store('notify').success('Artifact deleted');
      } catch (err) {
        this.deleteConfirm = null;
        Alpine.store('notify').error(`Failed to delete artifact: ${err.message}`);
      }
    },

    /** Format bytes to human-readable string */
    formatSize(bytes) {
      if (!bytes) return '0 B';
      if (bytes < 1024) return `${bytes} B`;
      if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
      return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    },

    /** Format ISO date string for display */
    formatDate(dateStr) {
      if (!dateStr) return '';
      try {
        return new Date(dateStr).toLocaleString(undefined, {
          year: 'numeric', month: 'short', day: 'numeric',
          hour: '2-digit', minute: '2-digit',
        });
      } catch { return dateStr; }
    },
  });

  // ── Navigation Store ─────────────────────────────────────────
  Alpine.store('nav', {
    activeTab: 'dashboard',

    setTab(tab) {
      this.activeTab = tab;
      // Dispatch event so tabs can initialize on first view
      window.dispatchEvent(new CustomEvent('mnemory:tab-changed', {
        detail: { tab }
      }));
    },
  });

  // ── Notification Store ───────────────────────────────────────
  Alpine.store('notify', {
    toasts: [],
    _nextId: 0,

    /**
     * Show a toast notification.
     * @param {string} message
     * @param {'success'|'error'|'info'|'warning'} type
     * @param {number} duration - ms before auto-dismiss (0 = manual)
     */
    show(message, type = 'info', duration = 5000) {
      const id = this._nextId++;
      this.toasts.push({ id, message, type, visible: true });
      if (duration > 0) {
        setTimeout(() => this.dismiss(id), duration);
      }
    },

    dismiss(id) {
      const idx = this.toasts.findIndex(t => t.id === id);
      if (idx !== -1) {
        this.toasts[idx].visible = false;
        setTimeout(() => {
          this.toasts = this.toasts.filter(t => t.id !== id);
        }, 300);
      }
    },

    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error', 8000); },
    info(msg) { this.show(msg, 'info'); },
    warning(msg) { this.show(msg, 'warning'); },
  });
});
