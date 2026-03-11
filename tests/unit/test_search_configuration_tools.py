"""
Unit tests for search configuration tools.

Tests critical paths for search configuration operations including:
- Search execution with configuration
- Error handling
- Query placeholder replacement
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.search_configuration_tools import (
    _replace_search_text_placeholder,
    execute_search_with_configuration,
)

pytestmark = pytest.mark.unit


class TestExecuteSearchWithConfiguration:
    """Tests for execute_search_with_configuration function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_execute_search_success(self):
        """Test successful search execution with configuration."""
        # Mock: get configuration
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "test_index",
                            "query": '{"multi_match": {"query": "%SearchText%", "fields": ["title"]}}',
                        },
                    }
                ]
            }
        }

        # Mock: search results
        mock_search_response = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_id": "doc1",
                        "_score": 1.5,
                        "_source": {"id": "doc1", "title": "Test Document 1"},
                    },
                    {
                        "_id": "doc2",
                        "_score": 1.2,
                        "_source": {"id": "doc2", "title": "Test Document 2"},
                    },
                ],
            },
            "took": 10,
            "max_score": 1.5,
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client = Mock()
        mock_client.search.return_value = mock_search_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="laptop",
                size=10,
            )
            result_data = json.loads(result)

            assert result_data["search_configuration_id"] == "config1"
            assert result_data["query_text"] == "laptop"
            assert result_data["total_hits"] == 2
            assert len(result_data["results"]) == 2
            # Verify placeholder was replaced
            search_call = mock_client.search.call_args
            search_body = search_call[1]["body"]
            assert "%SearchText%" not in json.dumps(search_body)
            assert "laptop" in json.dumps(search_body)

    @pytest.mark.asyncio
    async def test_execute_search_config_not_found(self):
        """Test error when configuration is not found."""
        mock_config_response = {"hits": {"hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="invalid", query_text="test"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_execute_search_invalid_query_json(self):
        """Test error when query JSON is invalid."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "test_index",
                            "query": "invalid json {",
                        },
                    }
                ]
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1", query_text="test"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Invalid query JSON" in result_data["error"]

    @pytest.mark.asyncio
    async def test_execute_search_with_custom_fields(self):
        """Test search execution with custom fields."""
        mock_config_response = {
            "hits": {
                "hits": [
                    {
                        "_id": "config1",
                        "_source": {
                            "name": "Test Config",
                            "index": "test_index",
                            "query": '{"match_all": {}}',
                        },
                    }
                ]
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [{"_id": "doc1", "_score": 1.0, "_source": {}}],
            },
            "took": 5,
        }

        mock_sr_client = Mock()
        mock_sr_client.get_search_configurations.return_value = mock_config_response

        mock_client = Mock()
        mock_client.search.return_value = mock_search_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.search_configuration_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await execute_search_with_configuration(
                search_configuration_id="config1",
                query_text="test",
                fields=["id", "title"],
            )
            result_data = json.loads(result)

            assert result_data["total_hits"] == 1
            # Verify fields were set in search
            search_call = mock_client.search.call_args
            search_body = search_call[1]["body"]
            assert search_body["_source"] == ["id", "title"]


class TestReplaceSearchTextPlaceholder:
    """Tests for _replace_search_text_placeholder function."""

    def test_replace_in_string(self):
        """Test placeholder replacement in string."""
        result = _replace_search_text_placeholder("Query: %SearchText%", "laptop")
        assert result == "Query: laptop"

    def test_replace_in_dict(self):
        """Test placeholder replacement in dictionary."""
        obj = {
            "query": {
                "multi_match": {
                    "query": "%SearchText%",
                    "fields": ["title"],
                }
            }
        }
        result = _replace_search_text_placeholder(obj, "laptop")
        assert result["query"]["multi_match"]["query"] == "laptop"

    def test_replace_in_list(self):
        """Test placeholder replacement in list."""
        obj = ["Query: %SearchText%", "Another: %SearchText%"]
        result = _replace_search_text_placeholder(obj, "phone")
        assert result[0] == "Query: phone"
        assert result[1] == "Another: phone"

    def test_replace_nested(self):
        """Test placeholder replacement in nested structures."""
        obj = {
            "bool": {
                "must": [
                    {"match": {"title": "%SearchText%"}},
                    {"match": {"description": "%SearchText%"}},
                ]
            }
        }
        result = _replace_search_text_placeholder(obj, "tablet")
        assert result["bool"]["must"][0]["match"]["title"] == "tablet"
        assert result["bool"]["must"][1]["match"]["description"] == "tablet"

    def test_no_replacement_needed(self):
        """Test when no placeholder exists."""
        obj = {"query": "simple query"}
        result = _replace_search_text_placeholder(obj, "laptop")
        assert result == obj
