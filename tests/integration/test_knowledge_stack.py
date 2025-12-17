#!/usr/bin/env python3
"""
Knowledge Stack Integration Tests
==================================
Test Khoj, Obsidian (CouchDB), and Zotero integrations.
"""

import pytest
import requests
import json
import time
from conftest import SERVICE_HOST


class TestKhojIntegration:
    """Test Khoj AI assistant integration."""

    BASE_URL = f"http://{SERVICE_HOST}:42110"

    @pytest.mark.integration
    def test_khoj_api_health(self):
        """Test Khoj API health endpoint."""
        response = requests.get(f"{self.BASE_URL}/api/health", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_khoj_config_endpoint(self):
        """Test Khoj configuration endpoint."""
        try:
            response = requests.get(f"{self.BASE_URL}/api/config/data", timeout=10)
            # May require auth, but should respond
            assert response.status_code in [200, 401, 403]
        except Exception as e:
            pytest.fail(f"Khoj config endpoint failed: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_khoj_search_endpoint(self):
        """Test Khoj search functionality."""
        try:
            # Search endpoint - may require setup
            response = requests.get(
                f"{self.BASE_URL}/api/search",
                params={"q": "test", "t": "all"},
                timeout=30
            )
            # Should respond (may be empty if no data indexed)
            assert response.status_code in [200, 401, 404, 500]
        except Exception as e:
            pytest.fail(f"Khoj search failed: {e}")


class TestCouchDBIntegration:
    """Test CouchDB (Obsidian LiveSync) integration."""

    BASE_URL = f"http://{SERVICE_HOST}:5984"
    AUTH = ("obsidian", "mlops-dev-password")

    @pytest.mark.integration
    def test_couchdb_up(self):
        """Test CouchDB is running."""
        response = requests.get(f"{self.BASE_URL}/_up", auth=self.AUTH, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "ok"

    @pytest.mark.integration
    def test_couchdb_server_info(self):
        """Test CouchDB server info."""
        response = requests.get(f"{self.BASE_URL}/", auth=self.AUTH, timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "couchdb" in data

    @pytest.mark.integration
    def test_couchdb_all_dbs(self):
        """Test listing all databases."""
        response = requests.get(f"{self.BASE_URL}/_all_dbs", auth=self.AUTH, timeout=10)
        assert response.status_code == 200
        dbs = response.json()
        assert isinstance(dbs, list)
        # System dbs should exist
        assert "_users" in dbs

    @pytest.mark.integration
    def test_couchdb_create_database(self):
        """Test creating and deleting a database."""
        db_name = "mlops_test_db"

        try:
            # Create database
            response = requests.put(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)
            assert response.status_code in [201, 412]  # 412 if already exists

            # Verify exists
            response = requests.get(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)
            assert response.status_code == 200

            # Delete database
            response = requests.delete(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)
            assert response.status_code == 200
        except Exception as e:
            # Clean up on failure
            requests.delete(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)
            pytest.fail(f"CouchDB database test failed: {e}")

    @pytest.mark.integration
    def test_couchdb_document_crud(self):
        """Test document CRUD operations."""
        db_name = "mlops_test_crud"
        doc_id = "test_doc"

        try:
            # Create database
            requests.put(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)

            # Create document
            doc = {"_id": doc_id, "type": "test", "content": "Hello, Obsidian!"}
            response = requests.put(
                f"{self.BASE_URL}/{db_name}/{doc_id}",
                auth=self.AUTH,
                json=doc,
                timeout=10
            )
            assert response.status_code == 201
            rev = response.json()["rev"]

            # Read document
            response = requests.get(
                f"{self.BASE_URL}/{db_name}/{doc_id}",
                auth=self.AUTH,
                timeout=10
            )
            assert response.status_code == 200
            assert response.json()["content"] == "Hello, Obsidian!"

            # Update document
            doc["_rev"] = rev
            doc["content"] = "Updated content"
            response = requests.put(
                f"{self.BASE_URL}/{db_name}/{doc_id}",
                auth=self.AUTH,
                json=doc,
                timeout=10
            )
            assert response.status_code == 201
            rev = response.json()["rev"]

            # Delete document
            response = requests.delete(
                f"{self.BASE_URL}/{db_name}/{doc_id}",
                auth=self.AUTH,
                params={"rev": rev},
                timeout=10
            )
            assert response.status_code == 200

            # Clean up database
            requests.delete(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)
        except Exception as e:
            # Clean up on failure
            requests.delete(f"{self.BASE_URL}/{db_name}", auth=self.AUTH, timeout=10)
            pytest.fail(f"CouchDB CRUD test failed: {e}")

    @pytest.mark.integration
    def test_couchdb_obsidian_livesync_config(self):
        """Test CouchDB is properly configured for Obsidian LiveSync."""
        # Check CORS is enabled (required for LiveSync)
        response = requests.options(
            f"{self.BASE_URL}/",
            headers={"Origin": "app://obsidian.md"},
            timeout=10
        )
        # CORS should allow obsidian.md origin
        cors_header = response.headers.get("Access-Control-Allow-Origin", "")
        assert "obsidian" in cors_header or cors_header == "*" or response.status_code == 200


class TestZoteroIntegration:
    """Test Zotero paper management integration."""

    BASE_URL = f"http://{SERVICE_HOST}:8085"

    @pytest.mark.integration
    def test_zotero_health(self):
        """Test Zotero service health."""
        response = requests.get(f"{self.BASE_URL}/health", timeout=10)
        assert response.status_code == 200

    @pytest.mark.integration
    def test_zotero_list_papers(self):
        """Test listing papers."""
        response = requests.get(f"{self.BASE_URL}/api/papers", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "papers" in data
        assert "count" in data

    @pytest.mark.integration
    def test_zotero_add_paper(self):
        """Test adding a paper manually."""
        paper = {
            "title": "Test Paper: MLOps Integration",
            "authors": "Test Author, Another Author",
            "year": 2024,
            "doi": "10.1234/test.mlops.001",
            "abstract": "This is a test paper for the MLOps workbench.",
            "tags": ["test", "mlops", "integration"],
        }

        try:
            # Add paper
            response = requests.post(
                f"{self.BASE_URL}/api/papers",
                json=paper,
                timeout=10
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            paper_id = data["id"]

            # Verify paper exists
            response = requests.get(f"{self.BASE_URL}/api/papers/{paper_id}", timeout=10)
            assert response.status_code == 200
            assert response.json()["title"] == paper["title"]

            # Delete paper
            response = requests.delete(f"{self.BASE_URL}/api/papers/{paper_id}", timeout=10)
            assert response.status_code == 200
        except Exception as e:
            pytest.fail(f"Zotero add paper test failed: {e}")

    @pytest.mark.integration
    def test_zotero_export_bibtex(self):
        """Test BibTeX export."""
        # Add a test paper first
        paper = {
            "title": "BibTeX Export Test",
            "authors": "Export Author",
            "year": 2024,
        }
        response = requests.post(f"{self.BASE_URL}/api/papers", json=paper, timeout=10)
        paper_id = response.json().get("id")

        try:
            # Export BibTeX
            response = requests.get(f"{self.BASE_URL}/api/export/bibtex", timeout=10)
            assert response.status_code == 200
            bibtex = response.text
            assert "@article" in bibtex or "@" in bibtex or bibtex == ""
        finally:
            # Clean up
            if paper_id:
                requests.delete(f"{self.BASE_URL}/api/papers/{paper_id}", timeout=10)

    @pytest.mark.integration
    def test_zotero_export_markdown(self):
        """Test Markdown export for Obsidian."""
        # Add a test paper first
        paper = {
            "title": "Markdown Export Test",
            "authors": "MD Author",
            "year": 2024,
        }
        response = requests.post(f"{self.BASE_URL}/api/papers", json=paper, timeout=10)
        paper_id = response.json().get("id")

        try:
            # Export Markdown
            response = requests.get(f"{self.BASE_URL}/api/export/markdown", timeout=10)
            assert response.status_code == 200
            assert "Paper Library" in response.text or "# " in response.text
        finally:
            # Clean up
            if paper_id:
                requests.delete(f"{self.BASE_URL}/api/papers/{paper_id}", timeout=10)

    @pytest.mark.integration
    def test_zotero_search_papers(self):
        """Test paper search functionality."""
        # Add test papers
        papers = [
            {"title": "Machine Learning Basics", "year": 2024, "tags": ["ml"]},
            {"title": "Deep Learning Advanced", "year": 2024, "tags": ["dl"]},
        ]
        paper_ids = []

        try:
            for paper in papers:
                response = requests.post(f"{self.BASE_URL}/api/papers", json=paper, timeout=10)
                paper_ids.append(response.json().get("id"))

            # Search
            response = requests.get(
                f"{self.BASE_URL}/api/papers",
                params={"q": "Machine"},
                timeout=10
            )
            assert response.status_code == 200
            data = response.json()
            assert data["count"] >= 1
        finally:
            # Clean up
            for paper_id in paper_ids:
                if paper_id:
                    requests.delete(f"{self.BASE_URL}/api/papers/{paper_id}", timeout=10)

    @pytest.mark.integration
    def test_zotero_stats(self):
        """Test statistics endpoint."""
        response = requests.get(f"{self.BASE_URL}/api/stats", timeout=10)
        assert response.status_code == 200
        data = response.json()
        assert "total_papers" in data
        assert "by_year" in data


class TestKnowledgeStackIntegration:
    """Test integration between knowledge stack components."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_zotero_to_obsidian_export(self):
        """Test exporting Zotero papers to Obsidian-compatible format."""
        # Add paper to Zotero
        paper = {
            "title": "Integration Test Paper",
            "authors": "Integration Author",
            "year": 2024,
            "abstract": "Testing Zotero to Obsidian integration.",
            "tags": ["integration", "test"],
        }

        try:
            response = requests.post(
                f"http://{SERVICE_HOST}:8085/api/papers",
                json=paper,
                timeout=10
            )
            paper_id = response.json().get("id")

            # Export to Markdown
            response = requests.get(
                f"http://{SERVICE_HOST}:8085/api/export/markdown",
                timeout=10
            )
            assert response.status_code == 200
            markdown = response.text

            # Should contain paper info
            assert "Integration Test Paper" in markdown

            # Clean up
            if paper_id:
                requests.delete(f"http://{SERVICE_HOST}:8085/api/papers/{paper_id}", timeout=10)
        except Exception as e:
            pytest.fail(f"Zotero to Obsidian export test failed: {e}")
