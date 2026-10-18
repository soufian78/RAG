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
    """Build context from retrieved chunks."""
    context_parts = []

    for chunk in chunks:
        context_parts.append(f"File: {chunk.file_path}\n" f"{chunk.content}")

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
    """Create the prompt for answering a question."""
    return f"""You are a question-answering assistant.

        Rules:
        1. Use only the provided context.
        2. Copy the relevant answer sentence exactly from the context.
        3. Do not add or invent information.
        4. If the answer is not explicitly present, respond exactly:
        I don't know based on the provided context.
        5. Output only the answer, without explanations.

        Context:
        {context}

        Question:
        {question}
        """


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
            "content": prompt + "\nAnswer briefly in one sentence.",
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
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
            max_new_tokens=20,
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

    context = build_context(found_chunks)

    prompt = build_prompt(question, context)

    answer = generate_answer(prompt)

    fallback = "I don't know based on the provided context."

    answer = answer.strip()

    if fallback in answer:
        answer = fallback

    elif answer not in context:
        answer = fallback

    sources = [chunk_to_source(chunk).model_dump() for chunk in found_chunks]

    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": sources,
    }


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
