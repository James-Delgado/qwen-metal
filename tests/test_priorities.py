"""Backlog drift test for docs/PRIORITIES.yaml.

Run: python3 -m pytest tests/test_priorities.py -q
Deps: pip3 install pyyaml pytest
(The Swift engine's own tests run via `swift test` in the engine package; this
file guards only the backlog's invariants.)
"""

from pathlib import Path

import yaml

PRIORITIES = Path(__file__).resolve().parent.parent / "docs" / "PRIORITIES.yaml"


def _load():
    with open(PRIORITIES, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_statuses_are_in_enum():
    doc = _load()
    allowed = set(doc["schema"]["task_status"])
    for task in doc["tasks"]:
        assert task["status"] in allowed, f"{task['id']}: bad status {task['status']!r}"


def test_dependencies_resolve_to_real_ids():
    doc = _load()
    ids = {task["id"] for task in doc["tasks"]}
    assert len(ids) == len(doc["tasks"]), "duplicate task ids"
    for task in doc["tasks"]:
        for ref in task.get("depends_on", []) + task.get("blocks", []):
            assert ref in ids, f"{task['id']}: unresolved reference {ref!r}"


def test_at_most_one_in_progress():
    doc = _load()
    in_progress = [t["id"] for t in doc["tasks"] if t["status"] == "in_progress"]
    assert len(in_progress) <= 1, f"multiple in_progress: {in_progress}"


def test_done_tasks_have_completed_at():
    doc = _load()
    for task in doc["tasks"]:
        if task["status"] == "done":
            assert task.get("completed_at"), f"{task['id']}: done without completed_at"


def test_blocked_tasks_have_unmet_dependencies_or_note():
    """A task marked blocked should have at least one not-done dependency,
    or an explicit notes field explaining the external blocker."""
    doc = _load()
    status_by_id = {t["id"]: t["status"] for t in doc["tasks"]}
    for task in doc["tasks"]:
        if task["status"] != "blocked":
            continue
        deps = task.get("depends_on", [])
        unmet = [d for d in deps if status_by_id.get(d) != "done"]
        assert unmet or task.get("notes"), (
            f"{task['id']}: blocked but all deps done and no explanatory note"
        )
