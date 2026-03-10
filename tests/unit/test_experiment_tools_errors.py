"""
Unit tests for experiment tools error scenarios.

Tests error paths and edge cases for experiment operations including:
- Experiment not found scenarios
- Empty results handling
- Error status handling
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.experiment_tools import (
    get_experiment_results,
)

pytestmark = pytest.mark.unit


class TestExperimentToolsErrors:
    """Test experiment tool error scenarios."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_experiment_results_empty(self):
        """Test get_experiment_results with no results."""
        # Should return empty results, not error
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "results": [],  # Empty results
                            "searchConfigurationList": ["config1", "config2"],
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            # Should return valid JSON with empty results message
            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "PAIRWISE_COMPARISON"
            assert result_data["total_queries"] == 0
            assert "message" in result_data
            assert "No results found" in result_data["message"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_not_found(self):
        """Test get_experiment_results when experiment doesn't exist."""
        mock_experiment_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("nonexistent_exp")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Experiment not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_error_status(self):
        """Test get_experiment_results when experiment has ERROR status."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "ERROR",
                            "errorMessage": "Experiment execution failed",
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["status"] == "ERROR"
            assert "error_message" in result_data
            assert result_data["error_message"] == "Experiment execution failed"
            assert "message" in result_data
            assert "No results available" in result_data["message"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_pending_status(self):
        """Test get_experiment_results when experiment is still PENDING."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "PENDING",
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["status"] == "PENDING"
            assert "message" in result_data
            assert "still pending" in result_data["message"].lower()

    @pytest.mark.asyncio
    async def test_get_experiment_results_unknown_status(self):
        """Test get_experiment_results when status is not COMPLETED/ERROR/PENDING/RUNNING."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "CANCELLED",
                        },
                    }
                ],
            }
        }
        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response
        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager
            result = await get_experiment_results("exp1")
            result_data = json.loads(result)
            assert result_data["status"] == "CANCELLED"
            assert "message" in result_data
            assert "Results not available" in result_data["message"]
            assert "CANCELLED" in result_data["message"]
