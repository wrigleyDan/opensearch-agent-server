"""
Unit tests for judgment list tools.

Tests critical paths for judgment operations including:
- Pair extraction from experiments
"""

import json
from unittest.mock import Mock, patch

import pytest

from tools.judgment_list_tools import (
    extract_pairs_from_pairwise_experiment,
)

pytestmark = pytest.mark.unit


class TestExtractPairsFromPairwiseExperiment:
    """Tests for extract_pairs_from_pairwise_experiment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_extract_pairs_success(self):
        """Test successful extraction of pairs from pairwise experiment."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "results": [
                                {
                                    "query_text": "laptop",
                                    "snapshots": [
                                        {"docIds": ["doc1", "doc2"]},
                                        {"docIds": ["doc3", "doc4"]},
                                    ],
                                }
                            ],
                        }
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await extract_pairs_from_pairwise_experiment(
                experiment_id="exp1",
                max_docs_per_query=10,
            )
            result_data = json.loads(result)

            assert result_data["experiment_id"] == "exp1"
            assert result_data["total_pairs"] == 4
            assert len(result_data["pairs"]) == 4

    @pytest.mark.asyncio
    async def test_extract_pairs_experiment_not_found(self):
        """Test error when experiment is not found."""
        mock_experiment_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await extract_pairs_from_pairwise_experiment(
                experiment_id="invalid"
            )
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Experiment not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_extract_pairs_wrong_type(self):
        """Test error when experiment is not pairwise."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                        }
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.judgment_list_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await extract_pairs_from_pairwise_experiment(experiment_id="exp1")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "PAIRWISE_COMPARISON" in result_data["error"]
