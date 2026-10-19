import json
from pathlib import Path

from pydantic import ValidationError
from tqdm import tqdm

from .models import RagDataset, StudentSearchResults, MinimalSource


def source_iou(source_a: MinimalSource, source_b: MinimalSource) -> float:
    """Calculate character-range IoU for sources in the same file."""

    if source_a.file_path != source_b.file_path:
        return 0.0

    intersection_start = max(
        source_a.first_character_index,
        source_b.first_character_index,
    )
    intersection_end = min(
        source_a.last_character_index,
        source_b.last_character_index,
    )

    intersection = max(0, intersection_end - intersection_start)

    union_start = min(
        source_a.first_character_index,
        source_b.first_character_index,
    )
    union_end = max(
        source_a.last_character_index,
        source_b.last_character_index,
    )
    union = union_end - union_start

    if union <= 0:
        return 0.0

    return float(intersection / union)


def source_matches(retrieved: MinimalSource, expected: MinimalSource) -> bool:
    """Check whether a retrieved source matches a ground-truth source."""

    return source_iou(retrieved, expected) >= 0.05


def evaluate_search_results(
    student_search_results_path: str,
    dataset_path: str,
) -> dict[int, float]:
    """Calculate Recall@k against a ground-truth dataset."""

    results_path = Path(student_search_results_path)
    ground_truth_path = Path(dataset_path)

    if not results_path.is_file():
        raise FileNotFoundError("Search results file"
                                f" not found: {results_path}")

    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            "Ground-truth dataset"
            f" not found: {ground_truth_path}"
        )

    try:
        with results_path.open("r", encoding="utf-8") as file:
            results_data = json.load(file)

        with ground_truth_path.open("r", encoding="utf-8") as file:
            dataset_data = json.load(file)

        student_results = StudentSearchResults(**results_data)
        ground_truth = RagDataset(**dataset_data)

    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON: {error}") from error
    except ValidationError as error:
        raise ValueError(f"Invalid dataset structure: {error}") from error

    if student_results.k <= 0:
        raise ValueError("The saved search results must have k > 0.")

    results_by_id = {
        result.question_id: result for result in student_results.search_results
    }

    # Evaluate only cutoffs available in the saved results.
    cutoffs = [cutoff for cutoff in (1, 3, 5, 10)
               if cutoff <= student_results.k]

    if not cutoffs:
        raise ValueError("No valid Recall@k cutoff is available.")

    relevant_questions = [
        question
        for question in ground_truth.rag_questions
        if hasattr(question, "sources") and question.sources
    ]

    if not relevant_questions:
        raise ValueError(
            "The ground-truth " "dataset contains no questions with sources."
        )

    recall_totals = {cutoff: 0.0 for cutoff in cutoffs}

    for question in tqdm(
        relevant_questions,
        desc="Evaluating Recall@k",
        unit="question",
    ):
        result = results_by_id.get(question.question_id)
        retrieved_sources = (result.retrieved_sources
                             if result is not None else [])

        for cutoff in cutoffs:
            top_sources = retrieved_sources[:cutoff]
            found_count = 0

            for expected in question.sources:
                if any(
                    source_matches(retrieved, expected)
                    for retrieved in top_sources
                ):
                    found_count += 1

            recall_totals[cutoff] += found_count / len(question.sources)

    question_count = len(relevant_questions)

    return {cutoff: total / question_count
            for cutoff, total in recall_totals.items()}
