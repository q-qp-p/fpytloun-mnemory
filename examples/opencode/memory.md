Memory recall and storage are handled AUTOMATICALLY by the mnemory plugin.
These instructions OVERRIDE any conflicting guidance from mnemory tool descriptions.

AUTOMATIC (do not duplicate):
- Do NOT call initialize_memory or get_core_memories — already done
- Do NOT call add_memory proactively — memories are stored automatically
- Do NOT call search_memories to "check for context" — relevant memories
  are already injected at session start

ALLOWED (explicit user requests only):
- search_memories / find_memories — when the user asks to look up something
  specific not already in context
- add_memory — when the user explicitly asks to remember something
- update_memory / delete_memory — when the user asks to change or forget
- list_memories / list_categories — when the user asks to browse
- Artifact operations — when the user needs detailed content

Use the memories in your context naturally to give better, more personalized
answers. Do not just acknowledge them — weave them in.
