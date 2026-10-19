from .models import (
    Chunk,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
)
from transformers import AutoTokenizer, AutoModelForCausalLM
from .retriever import load_index, search, chunk_to_source
from functools import lru_cache
import json
from pathlib import Path
import torch
import time
from typing import Any


def build_context(chunks: list[Chunk]) -> str:
    """Build a numbered context from retrieved chunks."""
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[SOURCE {i}]\n" f"File: {chunk.file_path}\n" f"{chunk.content}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> list[dict[str, str]]:
    """Create the prompt for answering a question."""
    prompt = "You are a RAG agent for vLLM repository.\n"
    prompt += (
        "Your task is to provide a natural-language answer "
        "to the user's query based only on the provided sources "
        "retrieved during search.\n"
    )
    prompt += (
        "Answer the question directly and concisely using "
        "the provided sources.\n"
        "Prefer 2 to 4 complete sentences. "
        "Include technical details only when relevant.\n"
        "Do not provide long lists or examples unless "
        "the question specifically asks for them.\n"
        "Finish your answer with a complete sentence.\n"
    )
    prompt += (
        "If the sources do not contain enough information to answer, "
        "say that the provided sources do not contain enough information.\n"
    )
    prompt += (
        "Cite the specific source number that directly supports "
        "each important factual claim, using [SOURCE n].\n"
        "Do not cite a source unless it explicitly supports "
        "the claim being made.\n"
        "Distinguish examples from requirements. "
        "Do not claim that a configuration option is required "
        "unless the sources explicitly say so.\n"
        "If a claim cannot be supported by the provided sources, "
        "omit it.\n"
    )
    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": f"Sources:\n{context}\n\nQuestion: {question}",
        },
    ]
    return messages


def prepare_prompt(question: str, k: int = 3) -> str:
    """Retrieve relevant chunks and prepare the prompt."""

    chunks, bm25 = load_index()

    found_chunks = search(
        query=question,
        chunks=chunks,
        bm25=bm25,
        k=k,
    )

    context = build_context(found_chunks)

    prompt = build_prompt(question, context)

    return prompt


def generate_answer(prompt: str) -> str:
    """Generate an answer using the language model."""
    torch.set_num_threads(8)
    tokenizer, model = load_model()

    messages = [
        {
            "role": "user",
            "content": "\nAnswer.",
        }
    ]

    text = tokenizer.apply_chat_template(
        prompt + messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.perf_counter() - start

    prompt_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][prompt_len:]

    print(f"Generation time: {elapsed:.2f} seconds")

    answer: str = str(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return answer.strip()


def answer_question(question: str, k: int = 3) -> dict:
    """Answer a question using the RAG pipeline."""
    chunks, bm25 = load_index()

    found_chunks = search(question, chunks, bm25, k)

    prompt = prepare_prompt(question)
    answer = generate_answer(prompt)

    answer = answer.strip()

    sources = [chunk_to_source(chunk).model_dump() for chunk in found_chunks]

    return {"answer": answer}



def test_generation() -> None:
    questions = [
        "What does the hello function do?",
        "What is GPU memory used for?",
        "Who invented Python?",
    ]

    for question in questions:
        result = answer_question(question, k=2)

        print("Question:", result["question"])
        print("Answer:", result["answer"])
        print("Sources:", result["retrieved_sources"])
        print("-" * 40)


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any]:
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    return tokenizer, model


def answer_dataset(
    dataset_path: str,
    k: int = 3,
) -> StudentSearchResultsAndAnswer:
    """Generate answers for all retrieved questions."""

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = StudentSearchResults(**data)
    results: list[MinimalAnswer] = []

    for item in dataset.search_results:
        generated = answer_question(
            question=item.question,
            k=k,
        )

        result = MinimalAnswer(
            question_id=item.question_id,
            question=item.question,
            retrieved_sources=item.retrieved_sources,
            answer=generated["answer"],
        )

        results.append(result)

    return StudentSearchResultsAndAnswer(
        search_results=results,
        k=dataset.k,
    )


def save_generation_results(
    results: StudentSearchResultsAndAnswer,
    dataset_path: str,
    save_dir: str,
) -> Path:
    """Save generated answers to a JSON file."""

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / Path(dataset_path).name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results.model_dump(),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def run_generation_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 3,
) -> Path:
    """Run the complete generation pipeline for a dataset."""

    results = answer_dataset(dataset_path=dataset_path, k=k)
    print("DEBUG:", results)

    output_path = save_generation_results(
        results=results, dataset_path=dataset_path, save_dir=save_directory
    )

    return output_path


if __name__ == "__main__":
    output_path = run_generation_dataset(
        dataset_path="data/datasets/test.json",
        save_directory="data/output/generation_results",
        k=3,
    )

    print("Saved to:", output_path)
