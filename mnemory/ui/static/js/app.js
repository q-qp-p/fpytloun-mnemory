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
