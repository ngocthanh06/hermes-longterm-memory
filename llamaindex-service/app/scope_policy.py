"""Shared project-recall scope policy.

This stays independent of Qdrant so fact and history retrieval cannot drift.
Boundary adapters validate too, but internal callers must not turn a typo into
cross-project recall.
"""

from typing import Literal, cast

ProjectScope = Literal["strict", "boost", "global"]
VALID_PROJECT_SCOPES = frozenset({"strict", "boost", "global"})


def validate(value: str) -> ProjectScope:
    """Return a valid scope or fail closed before retrieval occurs."""
    if value not in VALID_PROJECT_SCOPES:
        raise ValueError(
            f"invalid project_scope {value!r}; expected strict, boost, or global"
        )
    return cast(ProjectScope, value)


def filter_projects(
    project: str | None, scope: str, default_project: str
) -> list[str] | None:
    """Projects allowed by retrieval, or None when no filter is required."""
    checked = validate(scope)
    if project and checked == "strict":
        return [project, default_project]
    return None


def boost_same_project(
    project: str | None, hit_project: str, scope: str, default_project: str
) -> bool:
    """Whether the same-project score multiplier should be applied.

    A session whose own project IS the default/legacy bucket has no
    meaningful project to boost against — default is the catch-all, not a
    specific project — so it never qualifies, even though it still passes
    `filter_projects` for strict scoping."""
    checked = validate(scope)
    return bool(
        project
        and checked != "global"
        and hit_project == project
        and project != default_project
    )
