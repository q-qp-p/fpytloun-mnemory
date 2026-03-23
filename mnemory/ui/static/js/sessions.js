/**
 * mnemory UI — Sessions panel component.
 *
 * Lists persistent session summaries with filtering and expandable details.
 */

function sessionsPanel() {
  return {
    sessions: [],
    loading: false,
    error: '',
    stateFilter: '',

    async load() {
      this.loading = true;
      this.error = '';
      try {
        const params = {};
        if (this.stateFilter) {
          params.consolidation_state = this.stateFilter;
        }
        const data = await MnemoryAPI.get('/sessions', params);
        // Add _expanded flag for UI state
        this.sessions = (data.sessions || []).map(s => ({ ...s, _expanded: false }));
      } catch (e) {
        this.error = e.message || 'Failed to load sessions';
      } finally {
        this.loading = false;
      }
    },
  };
}
