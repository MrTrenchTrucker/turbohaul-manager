"""MLX warm-inherit fix: the idle-hot warm-inherit must NOT clobber the
incoming request's client_meta on the MLX backend.

Background: the inherit exists to restore llama.cpp KV-bin OWNERSHIP — the
session_id/role in client_meta select which on-disk /slots bin gets reloaded.
MLX has no KV bin at all (the MLX spawn path skips _restore_slot_kv), so the
inherit restores nothing; it only carries the idle holder's STALE `messages`
forward into the completion proxy, which forwards them to mlx_lm.server. The
model then answers the PREVIOUS turn's prompt.

Every request WITHOUT a session_id (bare curl, benchmark harness) took the
`not inc_sid` branch and had its prompt silently replaced — that is the
"completion cache/replay ignores the prompt" defect.
"""

from turbohaul.manager import should_inherit_idle_client_meta


IDLE = {"session_id": "sess-A", "role": "main", "messages": [{"role": "user", "content": "What is 12 times 13?"}]}


class TestMlxNeverInherits:
    def test_mlx_no_session_id_keeps_incoming(self):
        """THE BUG: a bare request (no session_id) must NOT inherit on MLX."""
        incoming = {"messages": [{"role": "user", "content": "Name three primary colors"}]}
        assert should_inherit_idle_client_meta(IDLE, incoming, is_mlx=True) is False

    def test_mlx_same_session_still_keeps_incoming(self):
        """Even a genuine same-session follow-up has no KV bin to restore on MLX."""
        incoming = {"session_id": "sess-A", "messages": [{"role": "user", "content": "follow up"}]}
        assert should_inherit_idle_client_meta(IDLE, incoming, is_mlx=True) is False

    def test_mlx_different_session_keeps_incoming(self):
        incoming = {"session_id": "sess-B", "messages": [{"role": "user", "content": "other"}]}
        assert should_inherit_idle_client_meta(IDLE, incoming, is_mlx=True) is False


class TestLlamaCppBehaviorUnchanged:
    """The llama.cpp path must stay byte-identical to the pre-fix behavior."""

    def test_llamacpp_no_session_id_inherits(self):
        incoming = {"messages": [{"role": "user", "content": "anything"}]}
        assert should_inherit_idle_client_meta(IDLE, incoming, is_mlx=False) is True

    def test_llamacpp_same_session_inherits(self):
        incoming = {"session_id": "sess-A", "messages": []}
        assert should_inherit_idle_client_meta(IDLE, incoming, is_mlx=False) is True

    def test_llamacpp_different_session_does_not_inherit(self):
        """Cross-session identity clobber guard (the earlier 2026-07-09 root fix)."""
        incoming = {"session_id": "sess-B", "messages": []}
        assert should_inherit_idle_client_meta(IDLE, incoming, is_mlx=False) is False


class TestDegenerateInputs:
    def test_no_idle_holder_never_inherits(self):
        assert should_inherit_idle_client_meta(None, {"session_id": "x"}, is_mlx=False) is False
        assert should_inherit_idle_client_meta(None, {"session_id": "x"}, is_mlx=True) is False

    def test_empty_idle_meta_never_inherits(self):
        assert should_inherit_idle_client_meta({}, {"session_id": "x"}, is_mlx=False) is False

    def test_non_dict_idle_meta_never_inherits(self):
        assert should_inherit_idle_client_meta("not-a-dict", {}, is_mlx=False) is False  # type: ignore[arg-type]

    def test_non_dict_incoming_treated_as_no_session(self):
        """A None incoming meta has no session of its own -> llama.cpp inherits."""
        assert should_inherit_idle_client_meta(IDLE, None, is_mlx=False) is True
        assert should_inherit_idle_client_meta(IDLE, None, is_mlx=True) is False
