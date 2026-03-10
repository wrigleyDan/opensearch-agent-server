"""
Unit tests for experiment tools.

Tests critical paths for experiment operations including:
- Experiment CRUD operations
- Experiment creation with different types
- Experiment retrieval with retry logic
- Results processing (pairwise and pointwise)
- Error handling
"""

import json
from collections.abc import Generator
from unittest.mock import Mock, patch

import pytest

from tools.experiment_tools import (
    _total_hits,
    get_experiment_results,
    list_experiment,
)

pytestmark = pytest.mark.unit


class TestTotalHitsHelper:
    """Tests for _total_hits helper (OpenSearch total as dict or int)."""

    def test_total_hits_dict_with_value(self):
        """_total_hits returns value when total is dict with 'value'."""
        assert _total_hits({"value": 5}) == 5
        assert _total_hits({"value": 0}) == 0

    def test_total_hits_dict_missing_value(self):
        """_total_hits returns 0 when dict has no 'value'."""
        assert _total_hits({}) == 0
        assert _total_hits({"relation": "eq"}) == 0

    def test_total_hits_int(self):
        """_total_hits returns int when total is raw int (older API)."""
        assert _total_hits(3) == 3
        assert _total_hits(0) == 0

    def test_total_hits_none(self):
        """_total_hits returns 0 when total is None."""
        assert _total_hits(None) == 0


class TestListExperiment:
    """Tests for list_experiment function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self) -> Generator[None, None, None]:
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_list_experiments_success(self):
        """Test successful listing of experiments."""
        mock_response = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {"_id": "exp1", "_source": {"name": "Experiment 1"}},
                    {"_id": "exp2", "_source": {"name": "Experiment 2"}},
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_experiment()
            result_data = json.loads(result)

            assert "hits" in result_data
            assert result_data["hits"]["total"]["value"] == 2
            mock_sr_client.get_experiments.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_experiments_error(self):
        """Test error handling when listing experiments fails."""
        mock_sr_client = Mock()
        mock_sr_client.get_experiments.side_effect = Exception("Connection error")

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await list_experiment()
            assert "Error listing experiments" in result


class TestGetExperimentResults:
    """Tests for get_experiment_results function."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_pairwise_results_success(self):
        """Test successful retrieval of pairwise experiment results."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1", "config2"],
                            "results": [
                                {
                                    "query_text": "laptop",
                                    "metrics": [
                                        {"metric": "jaccard", "value": 0.8},
                                        {"metric": "rbo50", "value": 0.75},
                                    ],
                                    "snapshots": [
                                        {
                                            "searchConfigurationId": "config1",
                                            "docIds": ["doc1", "doc2"],
                                        },
                                        {
                                            "searchConfigurationId": "config2",
                                            "docIds": ["doc3", "doc4"],
                                        },
                                    ],
                                },
                                {
                                    "query_text": "phone",
                                    "metrics": [
                                        {"metric": "jaccard", "value": 0.6},
                                    ],
                                    "snapshots": [
                                        {
                                            "searchConfigurationId": "config1",
                                            "docIds": ["doc5"],
                                        },
                                        {
                                            "searchConfigurationId": "config2",
                                            "docIds": ["doc6"],
                                        },
                                    ],
                                },
                            ],
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

            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "PAIRWISE_COMPARISON"
            assert result_data["total_queries"] == 2
            assert "aggregate_metrics" in result_data
            assert "top_performing_queries" in result_data
            assert "per_query_results" in result_data

    @pytest.mark.asyncio
    async def test_get_pointwise_results_success(self):
        """Test successful retrieval of pointwise experiment results."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "searchText": "laptop",
                            "metrics": [
                                {"metric": "NDCG@10", "value": 0.85},
                            ],
                            "documentIds": ["doc1", "doc2"],
                            "searchConfigurationId": "config1",
                        }
                    },
                    {
                        "_source": {
                            "searchText": "phone",
                            "metrics": [
                                {"metric": "NDCG@10", "value": 0.70},
                            ],
                            "documentIds": ["doc3"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 2}}},  # Count query
            mock_search_response,  # Full results query
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "POINTWISE_EVALUATION"
            assert result_data["total_queries"] == 2
            assert "aggregate_metrics" in result_data

    @pytest.mark.asyncio
    async def test_get_experiment_results_not_found(self):
        """Test error when experiment is not found."""
        mock_experiment_response = {"hits": {"total": {"value": 0}, "hits": []}}

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("invalid")
            result_data = json.loads(result)

            assert "error" in result_data
            assert "Experiment not found" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_error_status(self):
        """Test error when experiment has ERROR status."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "ERROR",
                            "errorMessage": "Test error",
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

    @pytest.mark.asyncio
    async def test_get_pairwise_results_empty_results(self):
        """Test critical path: pairwise experiment with no results."""
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

            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "PAIRWISE_COMPARISON"
            assert result_data["total_queries"] == 0
            assert "message" in result_data

    @pytest.mark.asyncio
    async def test_pairwise_results_null_metric_values(self):
        """Test critical path: pairwise results with null metric values."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test",
                                    "metrics": [
                                        {"metric": "jaccard", "value": None},
                                        {"metric": "rbo50", "value": 0.5},
                                    ],
                                    "snapshots": [],
                                },
                            ],
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

            # None/non-numeric metrics are skipped for aggregation; no crash.
            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "PAIRWISE_COMPARISON"
            assert result_data["total_queries"] == 1
            # Only numeric metric (rbo50) is aggregated; jaccard had None
            assert "aggregate_metrics" in result_data
            assert "rbo50" in result_data["aggregate_metrics"]
            assert result_data["aggregate_metrics"]["rbo50"]["mean"] == 0.5

    @pytest.mark.asyncio
    async def test_pairwise_results_empty_metrics_list(self):
        """Test critical path: pairwise results with empty metrics list."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test",
                                    "metrics": [],  # Empty metrics
                                    "snapshots": [],
                                },
                            ],
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

            assert result_data["total_queries"] == 1
            assert result_data["aggregate_metrics"] == {}
            assert result_data["primary_metric"] is None

    @pytest.mark.asyncio
    async def test_pairwise_results_missing_query_text(self):
        """Test critical path: pairwise results with missing query_text."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    # Missing query_text
                                    "metrics": [{"metric": "jaccard", "value": 0.8}],
                                    "snapshots": [],
                                },
                            ],
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

            assert result_data["total_queries"] == 1
            assert result_data["per_query_results"][0]["query_text"] == ""

    @pytest.mark.asyncio
    async def test_pairwise_results_malformed_metric_objects(self):
        """Test critical path: pairwise results with malformed metric objects."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test",
                                    "metrics": [
                                        {"metric": "jaccard", "value": 0.8},  # Valid
                                        {
                                            "metric": "rbo50"
                                        },  # Missing value (will be None)
                                        {
                                            "value": 0.5
                                        },  # Missing metric name (will be skipped)
                                        {},  # Empty object (will be skipped)
                                    ],
                                    "snapshots": [],
                                },
                            ],
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

            # Malformed/None metrics are skipped for aggregation; no crash.
            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "PAIRWISE_COMPARISON"
            assert result_data["total_queries"] == 1
            assert "aggregate_metrics" in result_data
            assert "jaccard" in result_data["aggregate_metrics"]
            assert result_data["aggregate_metrics"]["jaccard"]["mean"] == 0.8

    @pytest.mark.asyncio
    async def test_pairwise_results_single_value_statistics(self):
        """Test critical path: pairwise results with single value (stdev should be 0)."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test",
                                    "metrics": [{"metric": "jaccard", "value": 0.8}],
                                    "snapshots": [],
                                },
                            ],
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

            assert result_data["total_queries"] == 1
            assert result_data["aggregate_metrics"]["jaccard"]["std_dev"] == 0

    @pytest.mark.asyncio
    async def test_pairwise_results_negative_metric_values(self):
        """Test critical path: pairwise results with negative metric values."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test1",
                                    "metrics": [{"metric": "jaccard", "value": -0.5}],
                                    "snapshots": [],
                                },
                                {
                                    "query_text": "test2",
                                    "metrics": [{"metric": "jaccard", "value": 0.8}],
                                    "snapshots": [],
                                },
                            ],
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

            assert result_data["total_queries"] == 2
            assert result_data["aggregate_metrics"]["jaccard"]["min"] == -0.5
            assert result_data["aggregate_metrics"]["jaccard"]["max"] == 0.8

    @pytest.mark.asyncio
    async def test_pairwise_results_missing_snapshots(self):
        """Test critical path: pairwise results with missing snapshots."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test",
                                    "metrics": [{"metric": "jaccard", "value": 0.8}],
                                    # Missing snapshots
                                },
                            ],
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

            assert result_data["total_queries"] == 1
            assert result_data["per_query_results"][0]["snapshots"] == []

    @pytest.mark.asyncio
    async def test_pairwise_results_malformed_snapshots(self):
        """Test critical path: pairwise results with malformed snapshots."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test",
                                    "metrics": [{"metric": "jaccard", "value": 0.8}],
                                    "snapshots": [
                                        {
                                            "searchConfigurationId": "config1",
                                            "docIds": ["doc1"],
                                        },  # Valid
                                        {
                                            "searchConfigurationId": "config2"
                                        },  # Missing docIds
                                        {
                                            "docIds": ["doc3"]
                                        },  # Missing searchConfigurationId
                                        {},  # Empty snapshot
                                    ],
                                },
                            ],
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

            assert result_data["total_queries"] == 1
            assert len(result_data["per_query_results"][0]["snapshots"]) == 4

    @pytest.mark.asyncio
    async def test_pairwise_results_all_same_values(self):
        """Test critical path: pairwise results with all same metric values."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1"],
                            "results": [
                                {
                                    "query_text": "test1",
                                    "metrics": [{"metric": "jaccard", "value": 0.5}],
                                    "snapshots": [],
                                },
                                {
                                    "query_text": "test2",
                                    "metrics": [{"metric": "jaccard", "value": 0.5}],
                                    "snapshots": [],
                                },
                                {
                                    "query_text": "test3",
                                    "metrics": [{"metric": "jaccard", "value": 0.5}],
                                    "snapshots": [],
                                },
                            ],
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

            assert result_data["total_queries"] == 3
            assert result_data["aggregate_metrics"]["jaccard"]["mean"] == 0.5
            assert result_data["aggregate_metrics"]["jaccard"]["median"] == 0.5
            assert result_data["aggregate_metrics"]["jaccard"]["min"] == 0.5
            assert result_data["aggregate_metrics"]["jaccard"]["max"] == 0.5
            assert result_data["aggregate_metrics"]["jaccard"]["std_dev"] == 0

    @pytest.mark.asyncio
    async def test_pointwise_results_null_metric_values(self):
        """Test critical path: pointwise results with null metric values."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [
                                {"metric": "NDCG@10", "value": None},
                                {"metric": "MRR", "value": 0.5},
                            ],
                            "documentIds": ["doc1"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            # None/non-numeric metrics are skipped for aggregation; no crash.
            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "POINTWISE_EVALUATION"
            assert result_data["total_queries"] == 1
            assert "aggregate_metrics" in result_data
            assert "MRR" in result_data["aggregate_metrics"]
            assert result_data["aggregate_metrics"]["MRR"]["mean"] == 0.5

    @pytest.mark.asyncio
    async def test_pointwise_results_empty_metrics_list(self):
        """Test critical path: pointwise results with empty metrics list."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [],  # Empty metrics
                            "documentIds": ["doc1"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1
            assert result_data["aggregate_metrics"] == {}

    @pytest.mark.asyncio
    async def test_pointwise_results_missing_search_text(self):
        """Test critical path: pointwise results with missing searchText."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            # Missing searchText
                            "metrics": [{"metric": "NDCG@10", "value": 0.8}],
                            "documentIds": ["doc1"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1
            assert result_data["per_query_results"][0]["query_text"] == ""

    @pytest.mark.asyncio
    async def test_pointwise_results_missing_document_ids(self):
        """Test critical path: pointwise results with missing documentIds."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [{"metric": "NDCG@10", "value": 0.8}],
                            # Missing documentIds
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1
            assert result_data["per_query_results"][0]["document_ids"] == []

    @pytest.mark.asyncio
    async def test_pointwise_results_missing_search_configuration_id(self):
        """Test critical path: pointwise results with missing searchConfigurationId."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [{"metric": "NDCG@10", "value": 0.8}],
                            "documentIds": ["doc1"],
                            # Missing searchConfigurationId
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1
            assert result_data["per_query_results"][0]["search_configuration_id"] == ""

    @pytest.mark.asyncio
    async def test_pointwise_results_missing_primary_metric(self):
        """Test critical path: pointwise results without NDCG@10 metric."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [{"metric": "MRR", "value": 0.8}],  # No NDCG@10
                            "documentIds": ["doc1"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1
            assert (
                result_data["primary_metric"] == "MRR"
            )  # Should fall back to first available

    @pytest.mark.asyncio
    async def test_pointwise_results_single_value_statistics(self):
        """Test critical path: pointwise results with single value (stdev should be 0)."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [{"metric": "NDCG@10", "value": 0.8}],
                            "documentIds": ["doc1"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1
            assert result_data["aggregate_metrics"]["NDCG@10"]["std_dev"] == 0

    @pytest.mark.asyncio
    async def test_pointwise_results_malformed_metric_objects(self):
        """Test critical path: pointwise results with malformed metric objects."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_search_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_source": {
                            "searchText": "test",
                            "metrics": [
                                {"metric": "NDCG@10", "value": 0.8},  # Valid
                                {"metric": "MRR"},  # Missing value (will be None)
                                {"value": 0.5},  # Missing metric name (will be skipped)
                                {},  # Empty object (will be skipped)
                            ],
                            "documentIds": ["doc1"],
                            "searchConfigurationId": "config1",
                        }
                    },
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            # Malformed/None metrics are skipped for aggregation; no crash.
            assert result_data["experiment_id"] == "exp1"
            assert result_data["type"] == "POINTWISE_EVALUATION"
            assert result_data["total_queries"] == 1
            assert "aggregate_metrics" in result_data
            assert "NDCG@10" in result_data["aggregate_metrics"]
            assert result_data["aggregate_metrics"]["NDCG@10"]["mean"] == 0.8


class TestExperimentToolsLargeExperiments:
    """Critical path tests for large experiment performance."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_experiment_results_large_pointwise(self):
        """Test critical path: retrieving results from large pointwise experiment (1000+ queries)."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        # Simulate large result set (1000 queries)
        large_hits = []
        for i in range(1000):
            large_hits.append(
                {
                    "_source": {
                        "searchText": f"query_{i}",
                        "metrics": [
                            {"metric": "NDCG@10", "value": 0.5 + (i % 10) * 0.05}
                        ],
                        "documentIds": [f"doc{j}" for j in range(10)],
                        "searchConfigurationId": "config1",
                    }
                }
            )

        mock_search_response = {
            "hits": {
                "total": {"value": 1000},
                "hits": large_hits,
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 1000}}},
            mock_search_response,
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")
            result_data = json.loads(result)

            assert result_data["total_queries"] == 1000
            assert "aggregate_metrics" in result_data
            assert len(result_data["per_query_results"]) == 1000

    @pytest.mark.asyncio
    async def test_get_experiment_results_large_pairwise(self):
        """Test critical path: retrieving results from large pairwise experiment (500+ queries)."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "searchConfigurationList": ["config1", "config2"],
                            "results": [
                                {
                                    "query_text": f"query_{i}",
                                    "metrics": [
                                        {
                                            "metric": "jaccard",
                                            "value": 0.5 + (i % 10) * 0.05,
                                        },
                                        {
                                            "metric": "rbo50",
                                            "value": 0.4 + (i % 10) * 0.05,
                                        },
                                    ],
                                    "snapshots": [
                                        {
                                            "searchConfigurationId": "config1",
                                            "docIds": [f"doc{j}" for j in range(10)],
                                        },
                                        {
                                            "searchConfigurationId": "config2",
                                            "docIds": [
                                                f"doc{j + 10}" for j in range(10)
                                            ],
                                        },
                                    ],
                                }
                                for i in range(500)
                            ],
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

            assert result_data["total_queries"] == 500
            assert "aggregate_metrics" in result_data
            assert len(result_data["per_query_results"]) == 500


class TestExperimentToolsNetworkFailure:
    """Critical path tests for network failure recovery."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_experiment_results_network_failure_recovery(self):
        """Test critical path: get_experiment_results handles network failure gracefully."""
        mock_sr_client = Mock()
        mock_sr_client.get_experiments.side_effect = ConnectionError(
            "Network unreachable"
        )

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert "error" in result_data
            assert "Error retrieving experiment results" in result_data["error"]

    @pytest.mark.asyncio
    async def test_get_experiment_results_partial_network_failure(self):
        """Test critical path: get_experiment_results handles partial failure (metadata succeeds, results fail)."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "POINTWISE_EVALUATION",
                            "status": "COMPLETED",
                        },
                    }
                ],
            }
        }

        mock_sr_client = Mock()
        mock_sr_client.get_experiments.return_value = mock_experiment_response

        mock_client = Mock()
        mock_client.search.side_effect = [
            {"hits": {"total": {"value": 100}}},  # Count succeeds
            ConnectionError("Network error"),  # Results query fails
        ]

        mock_client_manager = Mock()
        mock_client_manager.get_search_relevance_client.return_value = mock_sr_client
        mock_client_manager.get_client.return_value = mock_client

        with patch(
            "tools.experiment_tools.get_client_manager"
        ) as mock_get_manager:
            mock_get_manager.return_value = mock_client_manager

            result = await get_experiment_results("exp1")

            assert isinstance(result, str)
            result_data = json.loads(result)
            assert "error" in result_data
            assert "Error processing pointwise results" in result_data["error"]


class TestExperimentToolsConcurrentRequests:
    """Critical path tests for concurrent request handling."""

    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        """Mock emitter to avoid dependencies."""
        with patch("utils.monitored_tool.get_ag_ui_emitter", return_value=None):
            yield

    @pytest.mark.asyncio
    async def test_get_experiment_results_concurrent_calls(self):
        """Test critical path: get_experiment_results handles concurrent calls."""
        mock_experiment_response = {
            "hits": {
                "total": {"value": 1},
                "hits": [
                    {
                        "_id": "exp1",
                        "_source": {
                            "type": "PAIRWISE_COMPARISON",
                            "status": "COMPLETED",
                            "results": [
                                {"query_text": "test", "metrics": [], "snapshots": []}
                            ],
                            "searchConfigurationList": ["config1"],
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

            import asyncio

            results = await asyncio.gather(
                get_experiment_results("exp1"),
                get_experiment_results("exp1"),
                get_experiment_results("exp1"),
            )

            # All should succeed
            for result in results:
                result_data = json.loads(result)
                assert result_data["total_queries"] == 1
