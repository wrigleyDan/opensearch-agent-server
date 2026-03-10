"""
Judgment List Tools
Tools for managing judgment lists using the OpenSearch Python client.
"""

from __future__ import annotations

import json
from datetime import datetime

from utils.logging_helpers import (
    get_logger,
    log_error_event,
    log_info_event,
    log_warning_event,
)
from utils.monitored_tool import monitored_tool
from utils.opensearch_client import get_client_manager
from utils.tool_utils import log_tool_error

logger = get_logger(__name__)

# Maximum allowed length for judgment list names
MAX_JUDGMENT_NAME_LENGTH = 50
# Reserve space for date suffix (_yyyyMMdd = 9 chars: 1 underscore + 8 date)
DATE_SUFFIX_LENGTH = 9
MAX_NAME_WITH_DATE = MAX_JUDGMENT_NAME_LENGTH - DATE_SUFFIX_LENGTH  # 41 chars


def _validate_judgment_name(name: str) -> tuple[bool, str]:
    """
    Validate judgment list name length.

    Args:
        name: The judgment list name

    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(name) > MAX_JUDGMENT_NAME_LENGTH:
        return (
            False,
            f"Judgment list name is too long ({len(name)} characters). Maximum allowed is {MAX_JUDGMENT_NAME_LENGTH} characters. Please provide a shorter name.",
        )
    return True, ""


def _truncate_name_for_date(name: str) -> str:
    """
    Truncate judgment list name to allow room for date suffix (_yyyyMMdd).

    Args:
        name: The judgment list name

    Returns:
        Truncated name (max 41 characters to allow for 9-char date suffix)
    """
    if len(name) > MAX_NAME_WITH_DATE:
        truncated = name[:MAX_NAME_WITH_DATE]
        log_info_event(
            logger,
            "Truncated judgment list name to allow for date suffix.",
            "tools.judgment_list.name_truncated",
            name=name,
            truncated=truncated,
        )
        return truncated
    return name


def _append_date_suffix(name: str) -> str:
    """
    Append current date in yyyyMMdd format to judgment list name.

    Args:
        name: The judgment list name (should already be truncated)

    Returns:
        Name with date suffix
    """
    date_suffix = datetime.now().strftime("%Y%m%d")
    return f"{name}_{date_suffix}"


@monitored_tool(
    name="ExtractPairsFromPairwiseExperimentTool",
    description="Extracts query-document pairs from a pairwise experiment for LLM judgment generation. Returns pairs in the format required by GenerateLLMJudgmentsTool.",
)
def extract_pairs_from_pairwise_experiment(
    experiment_id: str,
    max_docs_per_query: int = 10,
    include_snapshot_index: int | None = None,
) -> str:
    """
    Extracts query-document pairs from a pairwise experiment.

    Args:
        experiment_id: ID of the pairwise experiment
        max_docs_per_query: Maximum number of documents to include per query (default: 10)
        include_snapshot_index: If specified, only include docs from this snapshot (0 or 1). If None, includes docs from both snapshots.

    Returns:
        str: JSON string of query-doc pairs formatted for generate_llm_judgments
    """
    try:
        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] Extracting pairs from experiment.",
            "tools.judgment_list.extract_start",
            experiment_id=experiment_id,
        )

        client_manager = get_client_manager()
        sr_client = client_manager.get_search_relevance_client()

        # Get the experiment
        response = sr_client.get_experiments(experiment_id=experiment_id)

        exp_hits = response.get("hits", {}).get("total", {}).get("value", 0)
        if exp_hits == 0:
            log_warning_event(
                logger,
                "[ExtractPairsFromPairwiseExperimentTool] ✗ Experiment not found.",
                "tools.judgment_list.extract_not_found",
                experiment_id=experiment_id,
            )
            return json.dumps(
                {"error": "Experiment not found", "experiment_id": experiment_id},
                indent=2,
            )

        exp_data = response.get("hits", {}).get("hits", [])[0].get("_source", {})
        exp_type = exp_data.get("type", "")

        # Validate it's a pairwise experiment
        if exp_type != "PAIRWISE_COMPARISON":
            log_error_event(
                logger,
                "[ExtractPairsFromPairwiseExperimentTool] ✗ Wrong experiment type.",
                "tools.judgment_list.extract_wrong_type",
                exp_type=exp_type,
                exc_info=False,
            )
            return json.dumps(
                {
                    "error": f"This tool only works with PAIRWISE_COMPARISON experiments. Found: {exp_type}",
                    "experiment_id": experiment_id,
                    "experiment_type": exp_type,
                },
                indent=2,
            )

        # Extract results
        results = exp_data.get("results", [])
        if not results:
            log_warning_event(
                logger,
                "[ExtractPairsFromPairwiseExperimentTool] ✗ No results in experiment.",
                "tools.judgment_list.extract_no_results",
            )
            return json.dumps(
                {
                    "error": "No results found in experiment",
                    "experiment_id": experiment_id,
                },
                indent=2,
            )

        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] Found queries in experiment.",
            "tools.judgment_list.extract_queries",
            query_count=len(results),
        )

        # Extract query-doc pairs
        pairs = []
        query_doc_set = set()  # Track unique pairs to avoid duplicates

        for result in results:
            query_text = result.get("query_text", "")
            snapshots = result.get("snapshots", [])

            for snapshot_idx, snapshot in enumerate(snapshots):
                # Skip if we're filtering by snapshot index
                if (
                    include_snapshot_index is not None
                    and snapshot_idx != include_snapshot_index
                ):
                    continue

                doc_ids = snapshot.get("docIds", [])

                # Limit docs per query
                for doc_id in doc_ids[:max_docs_per_query]:
                    # Create unique key to avoid duplicates
                    pair_key = (query_text, doc_id)
                    if pair_key not in query_doc_set:
                        query_doc_set.add(pair_key)
                        pairs.append({"query": query_text, "doc_id": doc_id})

        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] Extracted unique query-doc pairs.",
            "tools.judgment_list.extract_pairs",
            pair_count=len(pairs),
        )

        result = {
            "experiment_id": experiment_id,
            "experiment_type": exp_type,
            "total_queries": len(results),
            "total_pairs": len(pairs),
            "max_docs_per_query": max_docs_per_query,
            "pairs": pairs,
        }

        log_info_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] ✓ Successfully extracted pairs",
            "tools.judgment_list.extract_done",
        )
        return json.dumps(result, indent=2)

    except Exception as e:
        log_error_event(
            logger,
            "[ExtractPairsFromPairwiseExperimentTool] ERROR.",
            "tools.judgment_list.extract_error",
            error=e,
        )
        return json.dumps(
            {"error": f"Error extracting pairs from pairwise experiment: {str(e)}"},
            indent=2,
        )


