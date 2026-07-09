from app import config
from app.memories import fact_point_id
from app.memory_store import message_point_id


def test_message_id_deterministic():
    a = message_point_id("local", "s1", "user", "hello")
    b = message_point_id("local", "s1", "user", "hello")
    assert a == b


def test_message_id_varies_with_each_component():
    base = message_point_id("local", "s1", "user", "hello")
    assert message_point_id("other", "s1", "user", "hello") != base
    assert message_point_id("local", "s2", "user", "hello") != base
    assert message_point_id("local", "s1", "assistant", "hello") != base
    assert message_point_id("local", "s1", "user", "hello!") != base


def test_fact_id_normalizes_whitespace_and_case():
    a = fact_point_id("local", "User likes Qdrant")
    assert fact_point_id("local", "  user   LIKES qdrant ") == a
    assert fact_point_id("local", "user\nlikes\tqdrant") == a


def test_fact_id_differs_for_different_text_or_user():
    a = fact_point_id("local", "User likes Qdrant")
    assert fact_point_id("local", "User likes Redis") != a
    assert fact_point_id("other", "User likes Qdrant") != a


def test_fact_id_scopes_by_project():
    # The same sentence is a distinct fact in two projects.
    assert fact_point_id("local", "x", "proj-a") != fact_point_id("local", "x", "proj-b")


def test_fact_id_empty_project_equals_default():
    # "" and the explicit default slug must mint the SAME id, or every
    # caller that omits the project would duplicate default-project facts.
    assert fact_point_id("local", "x") == fact_point_id("local", "x", config.DEFAULT_PROJECT)
