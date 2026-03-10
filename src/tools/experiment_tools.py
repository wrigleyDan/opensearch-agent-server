"""
Experiment Tools
Tools for managing search relevance experiments using the OpenSearch Python client.
Experiments allow offline testing of different search configurations.
"""

import json
import statistics

from opensearchpy import OpenSearch

from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import format_tool_error

logger = get_logger(__name__)

def _total_hits(total_value: dict | int | None) -> int:
    """Extract hit count from OpenSearch 'total' (dict with 'value' or raw int)."""
    if isinstance(total_value, dict):
        return int(total_value.get("value", 0) or 0)
    if total_value is None:
        return 0
    return int(total_value)


@monitored_tool(
    name="ListExperimentTool",
    description="Lists all search relevance experiments in OpenSearch",
)
def list_experiment() -> str:
    """
    Lists all available search relevance experiments.

    Returns:
        str: JSON string containing list of experiments
    """
    try:
        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        response = sr_client.get_experiments()

        return json.dumps(response, indent=2)
    except Exception as e:
        return format_tool_error(f"Error listing experiments: {str(e)}")


@monitored_tool(
    name="GetExperimentResultsTool",
    description="Retrieves and aggregates results from an offline evaluation experiment. Supports both PAIRWISE_COMPARISON and POINTWISE_EVALUATION experiments. Provides aggregate metrics, identifies top/bottom performing queries, and returns per-query details.",
)
def get_experiment_results(experiment_id: str) -> str:
    """
    Retrieves experiment results with aggregated metrics and per-query analysis.
    Handles both pairwise comparison and pointwise evaluation experiments.

    Args:
        experiment_id: ID of the experiment to retrieve results for

    Returns:
        str: JSON string with aggregate metrics, top/bottom performers, and per-query results
    """
    try:
        log_info_event(
            logger,
            f"[GetExperimentResultsTool] Starting retrieval for experiment: {experiment_id}",
            "tools.experiment.results_start",
            experiment_id=experiment_id,
        )
        client_manager = get_client_manager()
        client = client_manager.get_client()
        sr_client = client_manager.get_search_relevance_client()

        # First, get the experiment metadata to determine the type
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Fetching experiment metadata to determine type.",
            "tools.experiment.results_fetch_metadata",
        )
        exp_response = sr_client.get_experiments(experiment_id=experiment_id)
        exp_hits = _total_hits(exp_response.get("hits", {}).get("total", 0))

        if exp_hits == 0:
            log_warning_event(
                logger,
                f"[GetExperimentResultsTool] ✗ Experiment not found: {experiment_id}",
                "tools.experiment.results_not_found",
                experiment_id=experiment_id,
            )
            return json.dumps(
                {"experiment_id": experiment_id, "error": "Experiment not found"},
                indent=2,
            )

        exp_data = exp_response.get("hits", {}).get("hits", [])[0].get("_source", {})
        exp_type = exp_data.get("type", "")
        exp_status = exp_data.get("status", "")

        log_info_event(
            logger,
            f"[GetExperimentResultsTool] Experiment type: {exp_type}, status: {exp_status}",
            "tools.experiment.results_type_status",
            exp_type=exp_type,
            exp_status=exp_status,
        )

        # Check status
        if exp_status == "ERROR":
            error_message = exp_data.get("errorMessage", "No error message provided")
            log_error_event(
                logger,
                f"[GetExperimentResultsTool] ✗ Experiment failed: {error_message}",
                "tools.experiment.results_error_status",
                error=error_message,
                exc_info=False,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "status": "ERROR",
                    "error_message": error_message,
                    "message": "Experiment execution failed. No results available.",
                },
                indent=2,
            )
        elif exp_status in ["PENDING", "RUNNING", "PROCESSING"]:
            log_info_event(
                logger,
                "[GetExperimentResultsTool] Experiment still processing.",
                "tools.experiment.results_still_processing",
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "status": exp_status,
                    "message": f"Experiment is still {exp_status.lower()}. Results not yet available. Please try again later.",
                },
                indent=2,
            )
        elif exp_status != "COMPLETED":
            log_warning_event(
                logger,
                f"[GetExperimentResultsTool] Unexpected status: {exp_status}",
                "tools.experiment.results_unexpected_status",
                exp_status=exp_status,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "status": exp_status,
                    "message": f"Results not available for status '{exp_status}'. Only COMPLETED experiments have results.",
                },
                indent=2,
            )

        # Route to appropriate handler based on experiment type
        if exp_type == "PAIRWISE_COMPARISON":
            log_info_event(
                logger,
                "[GetExperimentResultsTool] Processing as PAIRWISE_COMPARISON.",
                "tools.experiment.results_pairwise",
            )
            return _get_pairwise_results(experiment_id, exp_data)
        elif exp_type == "POINTWISE_EVALUATION":
            log_info_event(
                logger,
                "[GetExperimentResultsTool] Processing as POINTWISE_EVALUATION.",
                "tools.experiment.results_pointwise",
            )
            return _get_pointwise_results(experiment_id, client)
        else:
            log_warning_event(
                logger,
                f"[GetExperimentResultsTool] Unsupported experiment type: {exp_type}",
                "tools.experiment.results_unsupported_type",
                exp_type=exp_type,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": exp_type,
                    "error": f"Unsupported experiment type: {exp_type}",
                    "message": "Only PAIRWISE_COMPARISON and POINTWISE_EVALUATION are currently supported.",
                },
                indent=2,
            )

    except Exception as e:
        error_msg = f"Error retrieving experiment results: {str(e)}"
        log_error_event(
            logger,
            f"[GetExperimentResultsTool] ERROR: {error_msg}",
            "tools.experiment.results_exception",
            error=e,
        )
        return format_tool_error(error_msg)


def _get_pairwise_results(experiment_id: str, exp_data: dict) -> str:
    """
    Process results for PAIRWISE_COMPARISON experiments.
    Results are embedded in the experiment document itself.
    """
    try:
        results = exp_data.get("results", [])
        search_config_ids = exp_data.get("searchConfigurationList", [])

        log_info_event(
            logger,
            f"[GetExperimentResultsTool] Found {len(results)} pairwise results",
            "tools.experiment.results_pairwise_count",
            count=len(results),
        )

        if len(results) == 0:
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": "PAIRWISE_COMPARISON",
                    "total_queries": 0,
                    "message": "No results found in pairwise experiment",
                },
                indent=2,
            )

        # Extract and aggregate metrics
        all_metrics: dict[str, list[float]] = {}
        per_query_results = []

        for result in results:
            query_text = result.get("query_text", "")
            metrics_list = result.get("metrics", [])
            snapshots = result.get("snapshots", [])

            # Convert metrics to dict
            query_metrics = {}
            for metric_obj in metrics_list:
                metric_name = metric_obj.get("metric")
                metric_value = metric_obj.get("value")
                query_metrics[metric_name] = metric_value

                # Collect for aggregation (only numeric values; skip None/non-numeric)
                if metric_name is not None and isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            # Format snapshots for readability
            comparison_snapshots = []
            for snapshot in snapshots:
                comparison_snapshots.append(
                    {
                        "search_configuration_id": snapshot.get(
                            "searchConfigurationId", ""
                        ),
                        "document_ids": snapshot.get("docIds", []),
                        "num_documents": len(snapshot.get("docIds", [])),
                    }
                )

            per_query_results.append(
                {
                    "query_text": query_text,
                    "metrics": query_metrics,
                    "snapshots": comparison_snapshots,
                }
            )

        # Calculate aggregate statistics
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Calculating aggregate statistics.",
            "tools.experiment.results_aggregate",
        )
        aggregate_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregate_metrics[metric_name] = {
                    "mean": round(statistics.mean(values), 4),
                    "median": round(statistics.median(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "std_dev": round(statistics.stdev(values), 4)
                    if len(values) > 1
                    else 0,
                }

        log_info_event(
            logger,
            "[GetExperimentResultsTool] Available metrics.",
            "tools.experiment.results_available_metrics",
            metrics=list(all_metrics.keys()),
        )

        # Sort by first available metric (e.g., jaccard, rbo50, etc.)
        primary_metric = list(all_metrics.keys())[0] if all_metrics else None

        if primary_metric:
            log_info_event(
                logger,
                "[GetExperimentResultsTool] Using primary metric for ranking.",
                "tools.experiment.results_primary_metric",
                primary_metric=primary_metric,
            )
            sorted_queries = sorted(
                per_query_results,
                key=lambda x: x["metrics"].get(primary_metric, 0),
                reverse=True,
            )
        else:
            sorted_queries = per_query_results

        # Add performance rank
        for idx, query_result in enumerate(sorted_queries):
            query_result["performance_rank"] = idx + 1

        # Get top 5 and bottom 5 performers
        top_n = min(5, len(sorted_queries))
        bottom_n = min(5, len(sorted_queries))

        top_performers = [
            {
                "query": q["query_text"],
                "rank": q["performance_rank"],
                **q["metrics"],
            }
            for q in sorted_queries[:top_n]
        ]

        underperforming = [
            {
                "query": q["query_text"],
                "rank": q["performance_rank"],
                **q["metrics"],
            }
            for q in sorted_queries[-bottom_n:]
        ]

        log_info_event(
            logger,
            "[GetExperimentResultsTool] Identified top and bottom performers.",
            "tools.experiment.results_top_bottom",
            top_count=len(top_performers),
            bottom_count=len(underperforming),
        )

        # Build final result
        result = {
            "experiment_id": experiment_id,
            "type": "PAIRWISE_COMPARISON",
            "total_queries": len(per_query_results),
            "search_configurations_compared": search_config_ids,
            "primary_metric": primary_metric,
            "aggregate_metrics": aggregate_metrics,
            "top_performing_queries": top_performers,
            "underperforming_queries": underperforming,
            "per_query_results": sorted_queries,
        }

        log_info_event(
            logger,
            "[GetExperimentResultsTool] ✓ Successfully aggregated pairwise results",
            "tools.experiment.results_pairwise_done",
            query_count=len(per_query_results),
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error processing pairwise results: {str(e)}"
        log_error_event(
            logger,
            "[GetExperimentResultsTool] ✗ ERROR.",
            "tools.experiment.results_pairwise_exception",
            error=e,
        )
        return format_tool_error(error_msg)


def _get_pointwise_results(
    experiment_id: str,
    client: OpenSearch,
) -> str:
    """
    Process results for POINTWISE_EVALUATION experiments.
    Results are stored in the search-relevance-evaluation-result index.
    """
    try:
        # First, get the total count of results
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Querying for result count.",
            "tools.experiment.results_pointwise_count_query",
        )
        count_query = {
            "query": {"match": {"experimentId": experiment_id}},
            "size": 0,
            "track_total_hits": True,
        }

        count_response = client.search(
            index="search-relevance-evaluation-result", body=count_query
        )

        total_hits = _total_hits(count_response.get("hits", {}).get("total", 0))
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Found result(s).",
            "tools.experiment.results_pointwise_found",
            total_hits=total_hits,
        )

        if total_hits == 0:
            log_warning_event(
                logger,
                "[GetExperimentResultsTool] No pointwise results found.",
                "tools.experiment.results_pointwise_empty",
                experiment_id=experiment_id,
            )
            return json.dumps(
                {
                    "experiment_id": experiment_id,
                    "type": "POINTWISE_EVALUATION",
                    "total_queries": 0,
                    "message": "No results found for this pointwise evaluation experiment",
                },
                indent=2,
            )

        # Now retrieve all results. TODO: Results may be capped by index
        # max_result_window (e.g. 10k); add pagination if needed for larger experiments.
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Retrieving all results.",
            "tools.experiment.results_pointwise_retrieve",
            total_hits=total_hits,
        )
        query_body = {
            "query": {"match": {"experimentId": experiment_id}},
            "size": total_hits,
            "track_total_hits": True,
        }

        response = client.search(
            index="search-relevance-evaluation-result", body=query_body
        )

        hits = response.get("hits", {}).get("hits", [])
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Retrieved hits from response.",
            "tools.experiment.results_pointwise_retrieved",
            hit_count=len(hits),
        )

        # Parse results and organize by metric
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Processing results and extracting metrics.",
            "tools.experiment.results_pointwise_process",
        )
        metrics_by_query: list[dict] = []
        all_metrics: dict[str, list[float]] = {}

        for hit in hits:
            source = hit.get("_source", {})
            query_text = source.get("searchText", "")
            metrics_list = source.get("metrics", [])

            # Convert metrics array to dict
            query_metrics = {}
            for metric_obj in metrics_list:
                metric_name = metric_obj.get("metric")
                metric_value = metric_obj.get("value")
                query_metrics[metric_name] = metric_value

                # Collect for aggregate calculation (only numeric; skip None/non-numeric)
                if metric_name is not None and isinstance(metric_value, (int, float)):
                    if metric_name not in all_metrics:
                        all_metrics[metric_name] = []
                    all_metrics[metric_name].append(metric_value)

            metrics_by_query.append(
                {
                    "query_text": query_text,
                    "metrics": query_metrics,
                    "document_ids": source.get("documentIds", []),
                    "search_configuration_id": source.get("searchConfigurationId", ""),
                    "timestamp": source.get("timestamp", ""),
                }
            )

        log_info_event(
            logger,
            "[GetExperimentResultsTool] Processed queries.",
            "tools.experiment.results_pointwise_processed",
            query_count=len(metrics_by_query),
        )
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Available metrics.",
            "tools.experiment.results_pointwise_metrics",
            metrics=list(all_metrics.keys()),
        )

        # Calculate aggregate statistics
        log_info_event(
            logger,
            "[GetExperimentResultsTool] Calculating aggregate statistics.",
            "tools.experiment.results_pointwise_aggregate",
        )
        aggregate_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                aggregate_metrics[metric_name] = {
                    "mean": round(statistics.mean(values), 4),
                    "median": round(statistics.median(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "std_dev": round(statistics.stdev(values), 4)
                    if len(values) > 1
                    else 0,
                }

        log_info_event(
            logger,
            "[GetExperimentResultsTool] Calculated aggregates for metrics.",
            "tools.experiment.results_pointwise_aggregates_done",
            metric_count=len(aggregate_metrics),
        )

        # Sort queries by primary metric (NDCG@10, or first available metric)
        primary_metric = "NDCG@10"
        if primary_metric not in all_metrics and all_metrics:
            primary_metric = list(all_metrics.keys())[0]

        log_info_event(
            logger,
            "[GetExperimentResultsTool] Using primary metric for ranking.",
            "tools.experiment.results_pointwise_primary_metric",
            primary_metric=primary_metric,
        )

        sorted_queries = sorted(
            metrics_by_query,
            key=lambda x: x["metrics"].get(primary_metric, 0),
            reverse=True,
        )

        # Add performance rank
        for idx, query_result in enumerate(sorted_queries):
            query_result["performance_rank"] = idx + 1

        # Get top 5 and bottom 5 performers
        top_n = 5
        bottom_n = 5
        top_performers = [
            {
                "query": q["query_text"],
                "rank": q["performance_rank"],
                **q["metrics"],
            }
            for q in sorted_queries[:top_n]
        ]

        underperforming = [
            {
                "query": q["query_text"],
                "rank": q["performance_rank"],
                **q["metrics"],
            }
            for q in sorted_queries[-bottom_n:]
        ]

        log_info_event(
            logger,
            "[GetExperimentResultsTool] Identified top and bottom performers.",
            "tools.experiment.results_pointwise_top_bottom",
            top_count=len(top_performers),
            bottom_count=len(underperforming),
        )

        # Build final result
        result = {
            "experiment_id": experiment_id,
            "type": "POINTWISE_EVALUATION",
            "total_queries": len(metrics_by_query),
            "primary_metric": primary_metric,
            "aggregate_metrics": aggregate_metrics,
            "top_performing_queries": top_performers,
            "underperforming_queries": underperforming,
            "per_query_results": sorted_queries,
        }

        log_info_event(
            logger,
            "[GetExperimentResultsTool] ✓ Successfully aggregated pointwise results",
            "tools.experiment.results_pointwise_done",
            query_count=len(metrics_by_query),
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Error processing pointwise results: {str(e)}"
        log_error_event(
            logger,
            "[GetExperimentResultsTool] ✗ ERROR.",
            "tools.experiment.results_pointwise_exception",
            error=e,
        )
        return format_tool_error(error_msg)
