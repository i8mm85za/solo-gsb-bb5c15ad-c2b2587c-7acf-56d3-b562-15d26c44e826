"""End-to-end tests for checkpoint-safe storage retention.

POST /sessions/{session_id}/retention deletes (or, with dry_run, only
inventories) published/quarantined readings older than a cutoff while keeping
every published reading the session's earliest checkpoint can still return.
"""
from __future__ import annotations

import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = str(tmp_path / "test.db")
    monkeypatch.setenv("SENSOR_DB_PATH", db_path)
    from app import database, main

    main.app.state.db_path = db_path
    database.init_db(db_path)
    with TestClient(main.app) as c:
        c.db_path = db_path  # type: ignore[attr-defined]
        yield c


def create_session(client, sources=("a", "b"), lateness=0.0, session_id="s"):
    r = client.post(
        "/sessions",
        json={
            "session_id": session_id,
            "source_ids": list(sources),
            "allowed_lateness_seconds": lateness,
        },
    )
    assert r.status_code == 201, r.text
    return session_id


def report(client, sid, readings):
    return client.post(f"/sessions/{sid}/readings", json={"readings": readings})


def seal(client, sid, source, final_observed_at):
    return client.post(
        f"/sessions/{sid}/sources/{source}/seal",
        json={"final_observed_at": final_observed_at},
    )


def make_checkpoint(client, sid, name):
    return client.post(f"/sessions/{sid}/checkpoints", json={"name": name})


def retention(client, sid, cutoff, dry_run=None):
    body = {"cutoff": cutoff}
    if dry_run is not None:
        body["dry_run"] = dry_run
    return client.post(f"/sessions/{sid}/retention", json=body)


def keys(rows):
    return [(x["source_id"], x["seq"]) for x in rows]


def full_stream(client, sid, limit=1000):
    out, cursor = [], None
    while True:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        page = client.get(f"/sessions/{sid}/stream", params=params).json()
        out.extend(page["readings"])
        cursor = page["next_cursor"]
        if not cursor:
            return out


def full_checkpoint_stream(client, sid, name, limit=1000):
    out, cursor = [], None
    while True:
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        page = client.get(
            f"/sessions/{sid}/checkpoints/{name}/stream", params=params
        ).json()
        out.extend(page["readings"])
        cursor = page["next_cursor"]
        if not cursor:
            return out


T0 = "2026-01-01T00:00:00Z"


def setup_ladder(client, sid="s"):
    """Publish a deterministic six-reading ladder with lateness 0.

    After the ladder the published stream (in order) is
    a@:00/1, b@:10/1, a@:20/2, b@:30/2, a@:40/3, b@:50/3; a@:01:00/4 remains
    buffered (strict-less-than publishing leaves it on the watermark).
    """
    report(client, sid, [
        {"source_id": "a", "seq": 1, "observed_at": T0, "value": 1},
        {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z", "value": 2},
    ])  # wm :10 -> a@:00 published
    report(client, sid, [
        {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:20Z", "value": 3}])
    # wm :20 -> b@:10 published
    report(client, sid, [
        {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:30Z", "value": 4}])
    # wm :30 -> a@:20 published
    report(client, sid, [
        {"source_id": "a", "seq": 3, "observed_at": "2026-01-01T00:00:40Z", "value": 5}])
    # wm :40 -> b@:30 published
    report(client, sid, [
        {"source_id": "b", "seq": 3, "observed_at": "2026-01-01T00:00:50Z", "value": 6}])
    # wm :50 -> a@:40 published
    report(client, sid, [
        {"source_id": "a", "seq": 4, "observed_at": "2026-01-01T00:01:00Z", "value": 7}])
    # wm :01:00 -> b@:50 published; a@:01:00 stays buffered
    report(client, sid, [
        {"source_id": "b", "seq": 4, "observed_at": "2026-01-01T00:01:10Z", "value": 8}])
    # wm :01:10 -> a@:01:00 still buffered (boundary)


class TestDryRun:
    def test_dry_run_reports_counts_without_deleting(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        r = retention(client, sid, "2026-01-01T00:00:35Z", dry_run=True)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is True
        assert body["deleted"] == 0
        # a@:00, b@:10, a@:20, b@:30 are strictly before :35
        assert body["candidates"] == 4
        assert body["retained"] == 0
        assert body["deletable"] == 4
        assert body["protected_by_checkpoint"] is None
        assert body["by_status"] == {
            "published": {"candidates": 4, "retained": 0, "deletable": 4},
            "quarantined": {"candidates": 0, "retained": 0, "deletable": 0},
        }
        # nothing actually changed
        assert len(full_stream(client, sid)) == 6

    def test_dry_run_is_default(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        r = retention(client, sid, "2026-01-01T00:00:35Z")  # dry_run omitted
        assert r.status_code == 200
        assert r.json()["dry_run"] is True
        assert r.json()["deleted"] == 0
        assert len(full_stream(client, sid)) == 6

    def test_cutoff_strictly_before_boundary(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        # cutoff exactly at b@:30: b@:30 sits on the boundary and is not a
        # candidate; only a@:00, b@:10, a@:20 are.
        body = retention(client, sid, "2026-01-01T00:00:30Z", True).json()
        assert body["candidates"] == 3
        assert body["deletable"] == 3

    def test_cutoff_normalised_to_utc(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        # 02:00:35+02:00 == 00:00:35Z
        body = retention(
            client, sid, "2026-01-01T02:00:35+02:00", True
        ).json()
        assert body["cutoff"] == "2026-01-01T00:00:35Z"
        assert body["candidates"] == 4

    def test_cutoff_before_all_data(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        body = retention(client, sid, "1999-01-01T00:00:00Z", True).json()
        assert body["candidates"] == body["retained"] == body["deletable"] == 0
        assert body["by_status"]["published"]["candidates"] == 0


class TestDeleteWithoutCheckpoints:
    def test_execute_deletes_candidates_and_keeps_rest(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        r = retention(client, sid, "2026-01-01T00:00:35Z", False)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["dry_run"] is False
        assert body["deleted"] == 4
        assert body["deletable"] == 4
        survivors = keys(full_stream(client, sid))
        # a@:40 and b@:50 are newer than the cutoff; a@:01:00 was never
        # published (buffered), so the published tail is two readings
        assert survivors == [("a", 3), ("b", 3)]

    def test_buffered_readings_are_never_candidates(self, client):
        # Readings sitting on the watermark boundary (or before any publish)
        # remain buffered; retention must never touch them even with a cutoff
        # far in the future.
        sid = create_session(client, sources=("a", "b"), lateness=0.0)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": T0},
            {"source_id": "b", "seq": 1, "observed_at": T0},
        ])  # wm :00: both readings sit on the boundary, nothing published
        state = client.get(f"/sessions/{sid}").json()
        assert state["published_watermark"] == "2026-01-01T00:00:00Z"
        body = retention(client, sid, "2026-12-31T00:00:00Z", False).json()
        assert body["candidates"] == 0
        assert body["deleted"] == 0
        assert len(full_stream(client, sid)) == 0
        # the buffered readings are still there and publish once clocks pass
        # the boundary
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:01Z"},
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:01Z"},
        ])
        assert keys(full_stream(client, sid)) == [("a", 1), ("b", 1)]

    def test_quarantined_always_deleted_without_checkpoints(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        late = report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T00:00:05Z", "value": 9}])
        assert late.json()["quarantined"] == 1
        body = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        assert body["by_status"]["quarantined"] == {
            "candidates": 1, "retained": 0, "deletable": 1,
        }
        assert body["deleted"] == 5
        assert client.get(f"/sessions/{sid}/quarantine").json()["readings"] == []
        # survivors in the same deterministic order
        assert keys(full_stream(client, sid)) == [("a", 3), ("b", 3)]

    def test_quarantined_after_cutoff_survives(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T00:00:45Z", "value": 9}])
        body = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        # the quarantined row (observed :45) is newer than cutoff: kept
        assert body["by_status"]["quarantined"]["candidates"] == 0
        q = client.get(f"/sessions/{sid}/quarantine").json()["readings"]
        assert keys(q) == [("a", 9)]


class TestCheckpointProtection:
    def test_keeps_readings_the_earliest_checkpoint_can_still_read(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": T0},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:20Z"}])
        report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:30Z"}])
        # two published (a@:00, b@:10); checkpoint boundary = b@:10
        cp = make_checkpoint(client, sid, "cp-early")
        assert cp.json()["stream_offset"] == 2
        setup_ladder(client, sid)  # remaining ladder reports (duplicates no-op)

        body = retention(client, sid, "2026-01-01T00:00:35Z", True).json()
        protection = body["protected_by_checkpoint"]
        assert protection == {
            "checkpoint_name": "cp-early",
            "stream_offset": 2,
            "last_observed_at": "2026-01-01T00:00:10Z",
            "last_source_id": "b",
            "last_seq": 1,
        }
        # candidates a@:00, b@:10, a@:20, b@:30; resume reads keyset strictly
        # after b@:10 -> a@:20 and b@:30 retained; a@:00 and boundary b@:10
        # are deletable.
        pub = body["by_status"]["published"]
        assert pub == {"candidates": 4, "retained": 2, "deletable": 2}
        assert body["candidates"] == 4
        assert body["retained"] == 2
        assert body["deletable"] == 2

        executed = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        assert executed["deleted"] == 2
        assert keys(full_stream(client, sid)) == [
            ("a", 2), ("b", 2), ("a", 3), ("b", 3)]
        # checkpoint resume semantics are intact: exactly the retained tail
        assert keys(full_checkpoint_stream(client, sid, "cp-early")) == [
            ("a", 2), ("b", 2), ("a", 3), ("b", 3),
        ]

    def test_empty_stream_checkpoint_protects_everything(self, client):
        sid = create_session(client)
        cp = make_checkpoint(client, sid, "start")
        assert cp.json()["stream_offset"] == 0
        setup_ladder(client, sid)
        body = retention(client, sid, "2026-12-31T00:00:00Z", True).json()
        assert body["protected_by_checkpoint"]["checkpoint_name"] == "start"
        assert body["protected_by_checkpoint"]["last_observed_at"] is None
        pub = body["by_status"]["published"]
        assert pub == {"candidates": 6, "retained": 6, "deletable": 0}
        assert body["deletable"] == 0
        # executing deletes no published readings (quarantined ones still go)
        executed = retention(client, sid, "2026-12-31T00:00:00Z", False).json()
        assert executed["deleted"] == 0
        assert len(full_checkpoint_stream(client, sid, "start")) == 6

    def test_quarantined_deleted_even_with_protecting_checkpoint(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": T0},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:20Z"}])
        report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:30Z"}])
        make_checkpoint(client, sid, "cp-early")  # boundary b@:10
        setup_ladder(client, sid)
        report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T00:00:05Z", "value": 9}])
        body = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        # published: 2 retained / 2 deleted; quarantine has no stream semantics
        assert body["by_status"]["published"] == {
            "candidates": 4, "retained": 2, "deletable": 2}
        assert body["by_status"]["quarantined"]["deletable"] == 1
        assert body["deleted"] == 3
        assert keys(full_stream(client, sid)) == [
            ("a", 2), ("b", 2), ("a", 3), ("b", 3)]

    def test_earliest_checkpoint_is_selected_by_position(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": T0},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:20Z"}])
        report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:30Z"}])
        make_checkpoint(client, sid, "cp-early")   # offset 2, boundary b@:10
        setup_ladder(client, sid)
        make_checkpoint(client, sid, "cp-late")    # offset 6, boundary b@:50
        # the furthest-back checkpoint governs, not the newest
        body = retention(client, sid, "2026-01-01T00:00:35Z", True).json()
        assert body["protected_by_checkpoint"]["checkpoint_name"] == "cp-early"
        assert body["retained"] == 2
        assert body["deletable"] == 2

    def test_checkpoint_rows_are_never_modified(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": T0},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:20Z"}])
        before = make_checkpoint(client, sid, "cp").json()
        setup_ladder(client, sid)
        retention(client, sid, "2026-12-31T00:00:00Z", False)
        after = client.get(f"/sessions/{sid}/checkpoints").json()["checkpoints"]
        assert len(after) == 1
        # the immutable snapshot (offset, keyset, watermark, created_at) survives
        assert after[0] == before

    def test_resume_pagination_still_works_after_deletion(self, client):
        sid = create_session(client)
        report(client, sid, [
            {"source_id": "a", "seq": 1, "observed_at": T0},
            {"source_id": "b", "seq": 1, "observed_at": "2026-01-01T00:00:10Z"},
        ])
        report(client, sid, [
            {"source_id": "a", "seq": 2, "observed_at": "2026-01-01T00:00:20Z"}])
        report(client, sid, [
            {"source_id": "b", "seq": 2, "observed_at": "2026-01-01T00:00:30Z"}])
        make_checkpoint(client, sid, "cp")  # offset 2, boundary b@:10
        setup_ladder(client, sid)
        retention(client, sid, "2026-12-31T00:00:00Z", False)
        # walk the resume window one row at a time over the gap left behind
        paged = full_checkpoint_stream(client, sid, "cp", limit=1)
        assert keys(paged) == [("a", 2), ("b", 2), ("a", 3), ("b", 3)]


def advance_stream(client, sid):
    """Publish four more readings after the ladder.

    Each report advances one source clock far enough to publish the other
    source's buffered reading, so a@01:00/4, b@01:10/4, a@01:20/5 and
    b@01:30/5 join the published tail in that order; a@01:40/6 and b@01:50/6
    remain buffered.  Works right after ``setup_ladder`` whether or not a
    retention round has already deleted published rows (deletions touch
    neither buffered readings nor the per-source clocks).
    """
    report(client, sid, [
        {"source_id": "a", "seq": 5, "observed_at": "2026-01-01T00:01:20Z"}])
    report(client, sid, [
        {"source_id": "b", "seq": 5, "observed_at": "2026-01-01T00:01:30Z"}])
    report(client, sid, [
        {"source_id": "a", "seq": 6, "observed_at": "2026-01-01T00:01:40Z"}])
    report(client, sid, [
        {"source_id": "b", "seq": 6, "observed_at": "2026-01-01T00:01:50Z"}])


class TestMultiRoundCheckpointProtection:
    """Regression: checkpoint protection must survive earlier cleanup rounds.

    A retention execution deletes published rows, so a checkpoint created
    *afterwards* records a smaller ``stream_offset`` than an older checkpoint
    whose captured keyset is actually earlier in the stream.  The earliest
    protector must therefore be chosen by the immutable captured keyset
    ``(last_observed_at, last_source_id, last_seq)`` -- never by the offset.
    """

    def test_offset_shrink_does_not_make_later_checkpoint_earliest(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)  # 6 published: a@:00/1 .. b@:50/3
        cp_a = make_checkpoint(client, sid, "cp-a").json()
        assert cp_a["stream_offset"] == 6
        assert (cp_a["last_observed_at"], cp_a["last_source_id"], cp_a["last_seq"]) == (
            "2026-01-01T00:00:50Z", "b", 3)

        # Round 1: delete everything cp-a already covered (boundary b@:50/3).
        first = retention(client, sid, "2026-01-01T00:00:45Z", False).json()
        assert first["protected_by_checkpoint"]["checkpoint_name"] == "cp-a"
        assert first["deleted"] == 5
        assert keys(full_stream(client, sid)) == [("b", 3)]

        # The stream advances again: a@01:00/4 .. b@01:30/5 are published.
        advance_stream(client, sid)
        assert keys(full_stream(client, sid)) == [
            ("b", 3), ("a", 4), ("b", 4), ("a", 5), ("b", 5)]

        # Checkpoint B is created *after* the round-1 deletion: its offset (5)
        # is smaller than cp-a's (6) even though its keyset is later.
        cp_b = make_checkpoint(client, sid, "cp-b").json()
        assert cp_b["stream_offset"] == 5
        assert cp_b["stream_offset"] < cp_a["stream_offset"]
        assert (cp_b["last_observed_at"], cp_b["last_source_id"], cp_b["last_seq"]) == (
            "2026-01-01T00:01:30Z", "b", 5)

        # Round 2 preview: the immutable keyset makes cp-a the earliest
        # protector, so a@01:00/4 (still inside cp-a's resume window) is kept.
        preview = retention(client, sid, "2026-01-01T00:01:10Z", True).json()
        assert preview["protected_by_checkpoint"] == {
            "checkpoint_name": "cp-a",
            "stream_offset": 6,
            "last_observed_at": "2026-01-01T00:00:50Z",
            "last_source_id": "b",
            "last_seq": 3,
        }
        assert preview["candidates"] == 2          # b@:50/3, a@01:00/4
        assert preview["retained"] == 1            # a@01:00/4 still read by cp-a
        assert preview["deletable"] == 1           # boundary b@:50/3 itself
        assert preview["by_status"]["published"] == {
            "candidates": 2, "retained": 1, "deletable": 1}

        # Round 2 execution reports exactly what the preview promised.
        executed = retention(client, sid, "2026-01-01T00:01:10Z", False).json()
        for field in ("candidates", "retained", "deletable",
                      "by_status", "protected_by_checkpoint"):
            assert executed[field] == preview[field]
        assert executed["deleted"] == preview["deletable"] == 1
        assert keys(full_stream(client, sid)) == [
            ("a", 4), ("b", 4), ("a", 5), ("b", 5)]

        # cp-a's resume window survived both rounds, row by row.
        assert keys(full_checkpoint_stream(client, sid, "cp-a", limit=1)) == [
            ("a", 4), ("b", 4), ("a", 5), ("b", 5)]
        # cp-b sits at the stream tail: nothing to resume yet...
        assert full_checkpoint_stream(client, sid, "cp-b") == []
        # ...until the next publishes, which it then serves normally.
        report(client, sid, [
            {"source_id": "b", "seq": 7, "observed_at": "2026-01-01T00:02:00Z"}])
        report(client, sid, [
            {"source_id": "a", "seq": 7, "observed_at": "2026-01-01T00:02:10Z"}])
        assert keys(full_checkpoint_stream(client, sid, "cp-b")) == [
            ("a", 6), ("b", 6)]

    def test_same_keyset_tie_is_broken_by_name(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        make_checkpoint(client, sid, "cp-a")  # offset 6, boundary b@:50/3
        deleted = retention(client, sid, "2026-01-01T00:00:45Z", False).json()
        assert deleted["deleted"] == 5
        # cp-b captures the *same* keyset as cp-a but a smaller offset (1 < 6)
        # because round 1 removed the rows cp-a's offset had counted.
        cp_b = make_checkpoint(client, sid, "cp-b").json()
        assert cp_b["stream_offset"] == 1
        assert cp_b["last_observed_at"] == "2026-01-01T00:00:50Z"

        body = retention(client, sid, "2026-01-01T00:00:55Z", True).json()
        # Equal immutable boundaries protect the identical set; the tie is
        # resolved deterministically by name.
        assert body["protected_by_checkpoint"]["checkpoint_name"] == "cp-a"
        assert body["candidates"] == 1
        assert body["retained"] == 0
        assert body["deletable"] == 1
        executed = retention(client, sid, "2026-01-01T00:00:55Z", False).json()
        assert executed["deleted"] == 1
        assert full_stream(client, sid) == []

    def test_empty_stream_checkpoint_protects_across_rounds(self, client):
        sid = create_session(client)
        make_checkpoint(client, sid, "cp-start")  # empty stream: null keyset
        setup_ladder(client, sid)
        make_checkpoint(client, sid, "cp-mid")    # offset 6, boundary b@:50/3

        # Round 1: the empty-stream checkpoint protects every published row.
        first = retention(client, sid, "2026-01-01T00:00:45Z", False).json()
        assert first["protected_by_checkpoint"]["checkpoint_name"] == "cp-start"
        assert first["protected_by_checkpoint"]["last_observed_at"] is None
        assert first["by_status"]["published"] == {
            "candidates": 5, "retained": 5, "deletable": 0}
        assert first["deleted"] == 0
        assert len(full_stream(client, sid)) == 6

        # The stream advances and a third checkpoint is created.
        advance_stream(client, sid)
        assert len(full_stream(client, sid)) == 10
        make_checkpoint(client, sid, "cp-late")   # offset 10, boundary b@01:30/5

        # Round 2: still the empty-stream checkpoint, still nothing deletable.
        second = retention(client, sid, "2026-01-01T00:01:10Z", False).json()
        assert second["protected_by_checkpoint"]["checkpoint_name"] == "cp-start"
        assert second["by_status"]["published"] == {
            "candidates": 7, "retained": 7, "deletable": 0}
        assert second["deleted"] == 0
        assert len(full_stream(client, sid)) == 10

        # Once the empty-stream checkpoint is removed, the next-earliest
        # immutable boundary (cp-mid) governs the following round.
        r = client.delete(f"/sessions/{sid}/checkpoints/cp-start")
        assert r.status_code == 204
        third = retention(client, sid, "2026-01-01T00:01:10Z", False).json()
        assert third["protected_by_checkpoint"]["checkpoint_name"] == "cp-mid"
        assert third["deleted"] == 6
        assert keys(full_stream(client, sid)) == [
            ("a", 4), ("b", 4), ("a", 5), ("b", 5)]
        assert keys(full_checkpoint_stream(client, sid, "cp-mid")) == [
            ("a", 4), ("b", 4), ("a", 5), ("b", 5)]

    def test_empty_stream_checkpoint_after_full_cleanup_protects_all(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        # No checkpoint exists: every published candidate is deletable.
        wiped = retention(client, sid, "2026-01-01T00:01:00Z", False).json()
        assert wiped["protected_by_checkpoint"] is None
        assert wiped["deleted"] == 6
        assert full_stream(client, sid) == []

        # A checkpoint created now captures the (again) empty stream.
        cp = make_checkpoint(client, sid, "cp-restart").json()
        assert cp["stream_offset"] == 0
        assert cp["last_observed_at"] is None

        advance_stream(client, sid)
        assert keys(full_stream(client, sid)) == [
            ("a", 4), ("b", 4), ("a", 5), ("b", 5)]
        make_checkpoint(client, sid, "cp-later")  # offset 4, boundary b@01:30/5

        # The empty-stream checkpoint always protects every later record.
        body = retention(client, sid, "2026-12-31T00:00:00Z", False).json()
        assert body["protected_by_checkpoint"]["checkpoint_name"] == "cp-restart"
        assert body["by_status"]["published"] == {
            "candidates": 4, "retained": 4, "deletable": 0}
        assert body["deleted"] == 0
        assert keys(full_checkpoint_stream(client, sid, "cp-restart")) == [
            ("a", 4), ("b", 4), ("a", 5), ("b", 5)]


class TestIdempotency:
    def test_repeated_execution_deletes_nothing_the_second_time(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        first = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        assert first["deleted"] == 4
        second = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        assert second["deleted"] == 0
        assert second["candidates"] == 0
        assert second["retained"] == 0
        assert second["deletable"] == 0
        assert keys(full_stream(client, sid)) == [("a", 3), ("b", 3)]

    def test_dry_run_after_execution_reports_zero(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        retention(client, sid, "2026-01-01T00:00:35Z", False)
        body = retention(client, sid, "2026-01-01T00:00:35Z", True).json()
        assert body["candidates"] == body["deletable"] == 0
        assert body["deleted"] == 0

    def test_repeated_request_is_safe_under_concurrency(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        outcomes = []
        barrier = threading.Barrier(2)

        def run():
            barrier.wait()
            r = retention(client, sid, "2026-01-01T00:00:35Z", False)
            outcomes.append(r.status_code)

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start(); t2.start(); t1.join(); t2.join()
        # serialised by BEGIN IMMEDIATE: one deletes 4, the other 0; no 500
        assert outcomes == [200, 200]
        assert keys(full_stream(client, sid)) == [("a", 3), ("b", 3)]


class TestWatermarksAlertsUnchanged:
    def test_watermark_unchanged(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        before = client.get(f"/sessions/{sid}").json()
        retention(client, sid, "2026-12-31T00:00:00Z", False)
        after = client.get(f"/sessions/{sid}").json()
        assert after["published_watermark"] == before["published_watermark"]
        assert after["status"] == before["status"]
        assert after["sources"] == before["sources"]

    def test_alert_state_and_history_unchanged(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        rule = client.post(f"/sessions/{sid}/alert-rules", json={
            "rule_id": "a-low", "source_id": "a", "direction": "below",
            "enter_threshold": 100, "recover_threshold": 200,
            "enter_consecutive": 1, "recover_consecutive": 1,
        })
        assert rule.status_code == 201
        current_before = client.get(f"/sessions/{sid}/alerts/current").json()
        history_before = client.get(f"/sessions/{sid}/alerts/history").json()
        retention(client, sid, "2026-12-31T00:00:00Z", False)
        current_after = client.get(f"/sessions/{sid}/alerts/current").json()
        history_after = client.get(f"/sessions/{sid}/alerts/history").json()
        assert current_after == current_before
        assert history_after == history_before

    def test_works_on_closed_session(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        seal(client, sid, "b", "2026-01-01T00:01:10Z")
        seal(client, sid, "a", "2026-01-01T00:01:00Z")
        assert client.get(f"/sessions/{sid}").json()["closed"] is True
        body = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        assert body["deleted"] == 4
        assert client.get(f"/sessions/{sid}").json()["closed"] is True
        assert keys(full_stream(client, sid)) == [
            ("a", 3), ("b", 3), ("a", 4), ("b", 4)]


class TestAtomicFailure:
    def test_failed_delete_rolls_the_whole_transaction_back(self, client):
        sid = create_session(client)
        setup_ladder(client, sid)
        report(client, sid, [
            {"source_id": "a", "seq": 9, "observed_at": "2026-01-01T00:00:05Z", "value": 9}])
        quarantined_before = client.get(
            f"/sessions/{sid}/quarantine").json()["readings"]
        assert len(quarantined_before) == 1

        # A trigger living in the database makes every readings DELETE abort;
        # it fires inside the endpoint's own serialised transaction, so both
        # the quarantined and published deletes must roll back together.
        conn = sqlite3.connect(client.db_path)
        try:
            conn.execute(
                "CREATE TRIGGER fail_retention DELETE ON readings"
                " BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END")
            conn.commit()
        finally:
            conn.close()

        r = retention(client, sid, "2026-01-01T00:00:35Z", False)
        assert r.status_code == 500, r.text
        # nothing was removed: failure changes no data
        assert len(full_stream(client, sid)) == 6
        assert len(client.get(f"/sessions/{sid}/quarantine").json()["readings"]) == 1

        conn = sqlite3.connect(client.db_path)
        try:
            conn.execute("DROP TRIGGER fail_retention")
            conn.commit()
        finally:
            conn.close()
        # after the fault clears, the identical request succeeds
        body = retention(client, sid, "2026-01-01T00:00:35Z", False).json()
        assert body["deleted"] == 5


class TestErrors:
    def test_unknown_session_404(self, client):
        r = retention(client, "nope", T0, True)
        assert r.status_code == 404

    def test_invalid_cutoff_400(self, client):
        sid = create_session(client)
        r = retention(client, sid, "not-a-timestamp", True)
        assert r.status_code == 400
        assert "ISO 8601" in r.json()["detail"]

    def test_blank_cutoff_422(self, client):
        sid = create_session(client)
        r = client.post(f"/sessions/{sid}/retention", json={"cutoff": "   "})
        assert r.status_code == 422

    def test_missing_cutoff_422(self, client):
        sid = create_session(client)
        r = client.post(f"/sessions/{sid}/retention", json={"dry_run": False})
        assert r.status_code == 422

    def test_invalid_json_422(self, client):
        sid = create_session(client)
        # bool fields coerce "yes"/"no" strings; a non-scalar is rejected
        r = client.post(f"/sessions/{sid}/retention", json={"cutoff": T0, "dry_run": ["no"]})
        assert r.status_code == 422


class TestNoSchemaChange:
    def test_retention_introduces_no_new_migration(self, client):
        # Retention is a delete over existing tables; the schema version and
        # the readings/checkpoints tables stay exactly as before.
        from app import database

        conn = sqlite3.connect(client.db_path)
        try:
            versions = [r[0] for r in conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version")]
            assert versions == [1, 2, 3]
            assert database.LATEST_VERSION == 3
        finally:
            conn.close()
