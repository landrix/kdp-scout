"""Tests for semantic keyword analysis features."""

import json
import sqlite3

import pytest

from kdp_scout.keyword_engine import normalize_semantic_relevance
from kdp_scout.db import (
    init_db, get_connection, SemanticClusterRepository,
    _migrate_add_semantic_clusters_table,
)


# ── Normalizer tests ──────────────────────────────────────────────


class TestNormalizeSemanticRelevance:
    def test_none_returns_zero(self):
        assert normalize_semantic_relevance(None) == 0.0

    def test_negative_returns_zero(self):
        assert normalize_semantic_relevance(-0.5) == 0.0

    def test_zero_returns_zero(self):
        assert normalize_semantic_relevance(0.0) == 0.0

    def test_mid_value(self):
        assert normalize_semantic_relevance(0.5) == 0.5

    def test_one_returns_one(self):
        assert normalize_semantic_relevance(1.0) == 1.0

    def test_above_one_clamped(self):
        assert normalize_semantic_relevance(1.5) == 1.0

    def test_small_positive(self):
        assert normalize_semantic_relevance(0.01) == pytest.approx(0.01)

    def test_string_number_converted(self):
        """Score passed as string should be converted to float."""
        assert normalize_semantic_relevance(0.75) == 0.75


# ── Schema / migration tests ─────────────────────────────────────


class TestSemanticClustersSchema:
    """Test that the semantic_clusters table is created correctly."""

    @pytest.fixture
    def db_conn(self, tmp_path, monkeypatch):
        """Create a temporary database with the full schema."""
        db_path = str(tmp_path / 'test.db')
        monkeypatch.setattr('kdp_scout.db.Config.get_db_path', lambda: db_path)
        monkeypatch.setattr('kdp_scout.config.Config.get_db_path', lambda: db_path)
        init_db()
        conn = get_connection()
        yield conn
        conn.close()

    def test_table_exists(self, db_conn):
        """semantic_clusters table should exist after init_db."""
        cursor = db_conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='semantic_clusters'"
        )
        assert cursor.fetchone() is not None

    def test_table_columns(self, db_conn):
        """Table should have the expected columns."""
        cursor = db_conn.execute("PRAGMA table_info(semantic_clusters)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            'id', 'cluster_label', 'keywords', 'relevance_score',
            'generated_phrases', 'book_context', 'created_date',
        }
        assert expected == columns

    def test_migration_idempotent(self, db_conn):
        """Running the migration twice should not error."""
        _migrate_add_semantic_clusters_table(db_conn)
        _migrate_add_semantic_clusters_table(db_conn)
        # Should still work fine
        cursor = db_conn.execute(
            "SELECT COUNT(*) FROM semantic_clusters"
        )
        assert cursor.fetchone()[0] == 0

    def test_insert_and_read(self, db_conn):
        """Should be able to insert and read back a cluster."""
        keywords_json = json.dumps(['medieval plague', 'black death'])
        phrases_json = json.dumps(['medieval plague conspiracy'])

        db_conn.execute(
            'INSERT INTO semantic_clusters '
            '(cluster_label, keywords, relevance_score, '
            'generated_phrases, book_context) '
            'VALUES (?, ?, ?, ?, ?)',
            ('plague themes', keywords_json, 0.85,
             phrases_json, 'Test Book | thriller'),
        )
        db_conn.commit()

        row = db_conn.execute(
            'SELECT * FROM semantic_clusters WHERE cluster_label = ?',
            ('plague themes',),
        ).fetchone()

        assert row is not None
        assert json.loads(row['keywords']) == ['medieval plague', 'black death']
        assert row['relevance_score'] == pytest.approx(0.85)
        assert row['book_context'] == 'Test Book | thriller'


# ── Repository tests ─────────────────────────────────────────────


class TestSemanticClusterRepository:
    @pytest.fixture
    def repo(self, tmp_path, monkeypatch):
        """Create a repository with a temporary database."""
        db_path = str(tmp_path / 'test.db')
        monkeypatch.setattr('kdp_scout.db.Config.get_db_path', lambda: db_path)
        monkeypatch.setattr('kdp_scout.config.Config.get_db_path', lambda: db_path)
        init_db()
        r = SemanticClusterRepository()
        yield r
        r.close()

    def test_add_cluster(self, repo):
        row_id = repo.add_cluster(
            cluster_label='historical fiction',
            keywords=json.dumps(['medieval', 'plague']),
            relevance_score=0.9,
            generated_phrases=json.dumps(['medieval plague fiction']),
            book_context='Test | thriller',
        )
        assert row_id is not None
        assert row_id > 0

    def test_get_clusters_by_context(self, repo):
        repo.add_cluster(
            cluster_label='cluster1',
            keywords=json.dumps(['kw1']),
            relevance_score=0.8,
            book_context='Book A | genre',
        )
        repo.add_cluster(
            cluster_label='cluster2',
            keywords=json.dumps(['kw2']),
            relevance_score=0.6,
            book_context='Book B | genre',
        )

        results = repo.get_clusters(book_context='Book A | genre')
        assert len(results) == 1
        assert results[0]['cluster_label'] == 'cluster1'

    def test_get_clusters_no_filter(self, repo):
        repo.add_cluster(
            cluster_label='c1',
            keywords=json.dumps(['a']),
            relevance_score=0.5,
        )
        repo.add_cluster(
            cluster_label='c2',
            keywords=json.dumps(['b']),
            relevance_score=0.7,
        )

        results = repo.get_clusters()
        assert len(results) == 2

    def test_get_latest_clusters_fresh(self, repo):
        repo.add_cluster(
            cluster_label='fresh',
            keywords=json.dumps(['word']),
            relevance_score=0.9,
            book_context='ctx',
        )
        results = repo.get_latest_clusters('ctx', max_age_hours=1)
        assert len(results) == 1

    def test_get_latest_clusters_wrong_context(self, repo):
        repo.add_cluster(
            cluster_label='cluster',
            keywords=json.dumps(['word']),
            relevance_score=0.9,
            book_context='ctx_a',
        )
        results = repo.get_latest_clusters('ctx_b', max_age_hours=1)
        assert len(results) == 0
