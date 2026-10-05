"""Committed-operation log: GET /api/operations.

Every committed registry replacement and every sample purge that actually
cleared packages is recorded with a shared, ever-increasing positive seq, in
the order the operations take effect. The record commits in the same
transaction as the business change, so a failed write or commit leaves
neither behind; stale requests are never recorded. The log is paged with
``after_seq``/``limit``/``until_seq`` and survives restarts.
"""

import sqlite3

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage import get_store, reset_store

client = TestClient(app)
# A storage fault must surface as a 500/INTERNAL response, not a test crash.
no_raise = TestClient(app, raise_server_exceptions=False)

# Clean graph: nothing sensitive in-graph, so the registry alone decides.
GRAPH = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {"id": "only_cn", "type": "filter", "input": "users",
         "references": ["country"]},
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}
ROWS = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS = {"only_cn": {"country": "CN"}}
ENTRIES = [{"source": "users", "column": "email"}]


def register(payload=GRAPH):
    resp = client.post("/api/graphs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def seal_ok(gid, rows=ROWS, conditions=CONDITIONS):
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": rows, "conditions": conditions},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def put_registry(entries, generation, http=None):
    return (http or client).put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )


def purge(*columns, http=None):
    return (http or client).post(
        "/api/packages/purge",
        json={"columns": [{"source": s, "column": c} for s, c in columns]},
    )


def operations(**params):
    resp = client.get("/api/operations", params=params)
    assert resp.status_code == 200, resp.text
    return resp.json()


# ------------------------------ what is recorded -----------------------------


def test_empty_log_pages_to_nothing():
    assert operations() == {
        "until_seq": 0,
        "records": [],
        "next_after_seq": 0,
    }


def test_replacement_is_recorded_with_generations_and_sorted_diff():
    # Entries submitted out of order: the recorded diff is sorted by
    # (source, column), not by submission order.
    resp = put_registry(
        [
            {"source": "b", "column": "y"},
            {"source": "a", "column": "x"},
            {"source": "a", "column": "w"},
        ],
        0,
    )
    assert resp.status_code == 200
    # The write response is unchanged: no seq leaks into it.
    assert set(resp.json()) == {"generation", "entries", "reviews"}

    assert operations() == {
        "until_seq": 1,
        "records": [
            {
                "seq": 1,
                "type": "replace",
                "generation_before": 0,
                "generation_after": 1,
                "added": [
                    {"source": "a", "column": "w"},
                    {"source": "a", "column": "x"},
                    {"source": "b", "column": "y"},
                ],
                "removed": [],
            }
        ],
        "next_after_seq": 1,
    }


def test_replacement_records_removals_and_same_content_still_counts():
    put_registry(ENTRIES, 0)
    put_registry([], 1)  # clears the list: users/email removed
    # Identical content (empty -> empty) still advances the generation and
    # is still recorded, with an empty diff.
    put_registry([], 2)

    log = operations()
    assert log["until_seq"] == 3
    assert [record["seq"] for record in log["records"]] == [1, 2, 3]
    assert log["records"][1] == {
        "seq": 2,
        "type": "replace",
        "generation_before": 1,
        "generation_after": 2,
        "added": [],
        "removed": ENTRIES,
    }
    assert log["records"][2] == {
        "seq": 3,
        "type": "replace",
        "generation_before": 2,
        "generation_after": 3,
        "added": [],
        "removed": [],
    }


def test_purge_is_recorded_with_generation_columns_and_sorted_package_ids():
    gid = register()
    p1 = seal_ok(gid)
    p2 = seal_ok(gid)
    put_registry(ENTRIES, 0)  # generation 1 (the graph is now blocked)
    resp = purge(("users", "email"), ("users", "country"))
    assert resp.status_code == 200
    # The purge response is unchanged: just the sorted purged ids.
    assert resp.json() == {"purged": sorted([p1, p2])}

    log = operations()
    assert [record["type"] for record in log["records"]] == ["replace", "purge"]
    purge_record = log["records"][1]
    assert purge_record == {
        "seq": 2,
        "type": "purge",
        "generation": 1,  # the purge does not move the generation
        "columns": [
            {"source": "users", "column": "country"},
            {"source": "users", "column": "email"},
        ],
        "purged": sorted([p1, p2]),
    }


def test_empty_match_purge_is_not_recorded():
    gid = register()
    seal_ok(gid)
    assert purge(("ghost", "x")).json() == {"purged": []}
    assert operations() == {"until_seq": 0, "records": [], "next_after_seq": 0}


def test_both_types_share_one_seq_in_effect_order():
    gid = register()
    pid = seal_ok(gid)
    put_registry(ENTRIES, 0)          # seq 1
    assert purge(("ghost", "x")).json() == {"purged": []}  # not recorded
    purge(("users", "email"))         # seq 2
    put_registry([], 1)               # seq 3
    assert purge(("users", "email")).json() == {"purged": []}  # not recorded

    log = operations()
    assert [(r["seq"], r["type"]) for r in log["records"]] == [
        (1, "replace"),
        (2, "purge"),
        (3, "replace"),
    ]
    assert log["records"][1]["purged"] == [pid]


def test_records_carry_no_samples_or_graphs():
    gid = register()
    pid = seal_ok(gid)
    put_registry(ENTRIES, 0)
    purge(("users", "email"))

    replace_record, purge_record = operations()["records"]
    assert set(replace_record) == {
        "seq", "type", "generation_before", "generation_after",
        "added", "removed",
    }
    assert set(purge_record) == {
        "seq", "type", "generation", "columns", "purged",
    }
    # Nothing in the log references the sealed sample or the graph itself.
    assert pid not in str(replace_record)
    assert gid not in str(purge_record)


# ------------------------- failures leave no record --------------------------


def test_stale_replacement_leaves_no_record():
    put_registry(ENTRIES, 0)
    resp = put_registry([], 0)  # stale: current generation is 1
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "GENERATION_CONFLICT"
    log = operations()
    assert log["until_seq"] == 1
    assert len(log["records"]) == 1


def test_failed_replacement_leaves_neither_change_nor_record(faulty_conn):
    register()
    put_registry(ENTRIES, 0)
    # Fail the log insert inside the replacement transaction: the whole
    # replacement (entries, generation, conclusions, record) rolls back.
    faulty_conn.arm(lambda sql: sql.startswith("INSERT INTO operations"))
    resp = put_registry([], 1, http=no_raise)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL"

    assert client.get("/api/sensitive-columns").json() == {
        "generation": 1, "entries": ENTRIES}
    log = operations()
    assert log["until_seq"] == 1
    assert [r["type"] for r in log["records"]] == ["replace"]

    # Fault cleared: the same replacement succeeds and is recorded as seq 2.
    assert put_registry([], 1).status_code == 200
    log = operations()
    assert [r["seq"] for r in log["records"]] == [1, 2]
    assert log["records"][1]["removed"] == ENTRIES


def test_failed_purge_leaves_neither_change_nor_record(faulty_conn):
    gid = register()
    pid = seal_ok(gid)
    faulty_conn.arm(lambda sql: sql.startswith("INSERT INTO operations"))
    resp = purge(("users", "email"), http=no_raise)
    assert resp.status_code == 500
    assert resp.json()["error"]["code"] == "INTERNAL"

    # The package is untouched and no record was left behind.
    assert client.get(f"/api/packages/{pid}").status_code == 409  # unreleased
    assert get_store().get_package_row(pid)["purged"] == 0
    assert operations() == {"until_seq": 0, "records": [], "next_after_seq": 0}

    # Fault cleared: the same purge succeeds and is recorded as seq 1.
    assert purge(("users", "email")).json() == {"purged": [pid]}
    log = operations()
    assert [r["type"] for r in log["records"]] == ["purge"]
    assert log["records"][0]["purged"] == [pid]


# --------------------------------- paging ------------------------------------


def test_paging_with_after_seq_and_limit():
    gid = register()
    seal_ok(gid)
    put_registry(ENTRIES, 0)   # seq 1
    purge(("users", "email"))  # seq 2
    put_registry([], 1)        # seq 3

    page1 = operations(after_seq=0, limit=2)
    assert [r["seq"] for r in page1["records"]] == [1, 2]
    assert page1["until_seq"] == 3
    assert page1["next_after_seq"] == 2

    page2 = operations(after_seq=page1["next_after_seq"], limit=2)
    assert [r["seq"] for r in page2["records"]] == [3]
    assert page2["next_after_seq"] == 3

    # A final empty page keeps the cursor where it was.
    page3 = operations(after_seq=page2["next_after_seq"], limit=2)
    assert page3 == {"until_seq": 3, "records": [], "next_after_seq": 3}


def test_fixed_cutoff_never_mixes_in_later_operations():
    put_registry(ENTRIES, 0)  # seq 1
    put_registry([], 1)       # seq 2

    page = operations(after_seq=0, limit=1, until_seq=2)
    assert [r["seq"] for r in page["records"]] == [1]
    assert page["until_seq"] == 2
    assert page["next_after_seq"] == 1

    put_registry(ENTRIES, 2)  # seq 3, committed after the cutoff was read

    # Re-paging with the same fixed cutoff sees exactly the same horizon.
    page = operations(after_seq=page["next_after_seq"], limit=1, until_seq=2)
    assert [r["seq"] for r in page["records"]] == [2]
    assert page["until_seq"] == 2
    assert page["next_after_seq"] == 2
    page = operations(after_seq=2, limit=1, until_seq=2)
    assert page == {"until_seq": 2, "records": [], "next_after_seq": 2}

    # Without an explicit cutoff the new operation is visible.
    assert operations()["until_seq"] == 3


def test_until_seq_defaults_to_the_committed_maximum():
    put_registry(ENTRIES, 0)
    put_registry([], 1)
    log = operations()
    assert log["until_seq"] == 2
    assert [r["seq"] for r in log["records"]] == [1, 2]
    # An explicit cutoff at the committed maximum is accepted.
    assert operations(until_seq=2)["records"] == log["records"]
    # A smaller explicit cutoff truncates the page.
    assert [r["seq"] for r in operations(until_seq=1)["records"]] == [1]


def test_until_seq_beyond_the_committed_maximum_is_rejected():
    put_registry(ENTRIES, 0)  # committed max seq is 1
    resp = client.get("/api/operations", params={"until_seq": 2})
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "UNTIL_SEQ_NOT_COMMITTED"
    assert resp.json()["error"]["until_seq"] == 2
    assert resp.json()["error"]["committed_seq"] == 1


def test_until_seq_beyond_empty_log_is_rejected_but_zero_is_fine():
    resp = client.get("/api/operations", params={"until_seq": 1})
    assert resp.status_code == 400
    assert resp.json()["error"]["committed_seq"] == 0
    assert operations(until_seq=0) == {
        "until_seq": 0, "records": [], "next_after_seq": 0}


def test_empty_page_after_the_end_keeps_after_seq_as_cursor():
    put_registry(ENTRIES, 0)  # committed max seq is 1
    page = operations(after_seq=5)
    assert page == {"until_seq": 1, "records": [], "next_after_seq": 5}


def test_query_parameter_validation():
    put_registry(ENTRIES, 0)
    for params in (
        {"after_seq": -1},
        {"until_seq": -1},
        {"limit": 0},
        {"limit": 101},
        {"limit": "many"},
        {"after_seq": "soon"},
    ):
        resp = client.get("/api/operations", params=params)
        assert resp.status_code == 422, params
    # Boundary values are accepted.
    assert operations(after_seq=0, limit=1)["records"]
    assert len(operations(after_seq=0, limit=100)["records"]) == 1


# ------------------------------ persistence ----------------------------------


def test_records_survive_a_restart(tmp_path):
    db_path = str(tmp_path / "ops.db")
    reset_store(db_path)
    gid = register()
    pid = seal_ok(gid)
    put_registry(ENTRIES, 0)
    purge(("users", "email"))
    before = operations()

    # Simulate a restart: drop the process-wide store, reopen the same file.
    reset_store(db_path)
    assert operations() == before
    assert client.get(f"/api/packages/{pid}").status_code == 410


def test_database_from_before_the_log_starts_with_an_empty_log(tmp_path):
    db_path = str(tmp_path / "old.db")
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO meta VALUES ('generation', '0');
        CREATE TABLE sensitive_entries (
            source_id TEXT NOT NULL, column_name TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            PRIMARY KEY (source_id, column_name));
        CREATE TABLE graphs (
            id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL,
            nodes TEXT NOT NULL, export TEXT NOT NULL,
            review TEXT NOT NULL, review_generation INTEGER NOT NULL);
        CREATE TABLE packages (
            id TEXT PRIMARY KEY, ordinal INTEGER NOT NULL,
            graph_id TEXT NOT NULL, columns TEXT NOT NULL, rows TEXT NOT NULL,
            generation INTEGER NOT NULL, release_generation INTEGER);
        """
    )
    conn.close()

    reset_store(db_path)
    # The old database gained the log table; the log itself starts empty.
    assert operations() == {"until_seq": 0, "records": [], "next_after_seq": 0}
    # New operations are recorded from seq 1.
    put_registry(ENTRIES, 0)
    assert [r["seq"] for r in operations()["records"]] == [1]


# ------------------------------- fault fixture -------------------------------


class FaultyConnection:
    """Delegates to the real connection, failing an armed statement once."""

    def __init__(self, real):
        self._real = real
        self._predicate = None

    def arm(self, predicate):
        self._predicate = predicate

    def _check(self, sql):
        if self._predicate is not None and self._predicate(sql):
            self._predicate = None  # a transient fault fires exactly once
            raise sqlite3.OperationalError("injected storage fault")

    def execute(self, sql, parameters=()):
        self._check(sql)
        return self._real.execute(sql, parameters)

    def executemany(self, sql, seq):
        self._check(sql)
        return self._real.executemany(sql, seq)

    def commit(self):
        self._check("COMMIT")
        return self._real.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        # Mirror sqlite3.Connection: commit on success, rollback on error.
        if exc_type is None:
            self._real.commit()
        else:
            self._real.rollback()
        return False

    def __getattr__(self, name):
        return getattr(self._real, name)


@pytest.fixture
def faulty_conn():
    store = get_store()
    real = store._conn
    proxy = FaultyConnection(real)
    store._conn = proxy
    yield proxy
    store._conn = real
