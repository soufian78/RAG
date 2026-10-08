from .retriever import load_index, search
from .models import Chunk, RagDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from .retriever import load_index, search, chunk_to_source
from functools import lru_cache
import json
from pathlib import Path




def build_context(chunks: list[Chunk]) -> str:
    context_parts = []

    for chunk in chunks:
        context_parts.append(
            f"File: {chunk.file_path}\n"
            f"{chunk.content}"
        )

    return "\n\n".join(context_parts)


def build_prompt(question: str, context: str) -> str:
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
    tokenizer, model = load_model()
    messages = [
        {
            "role": "user",
            "content": prompt + "\nGive one short answer. Do not repeat."
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    inputs = tokenizer(text, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

def answer_question(question: str, k: int = 3) -> dict:

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

    sources = [
        chunk_to_source(chunk).model_dump()
        for chunk in found_chunks
    ]

 
    return {
        "question": question,
        "answer": answer,
        "retrieved_sources": sources
    }

def test_generation():
    questions = [
        "What does the hello function do?",
        "What is GPU memory used for?",
        "Who invented Python?"
    ]

    for question in questions:
        result = answer_question(question, k=2)

        print("Question:", result["question"])
        print("Answer:", result["answer"])
        print("Sources:", result["retrieved_sources"])
        print("-" * 40)

@lru_cache(maxsize=1)
def load_model():
    model_name = "Qwen/Qwen3-0.6B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    model.eval()

    return tokenizer, model

def answer_dataset(dataset_path: str, k: int = 3) -> list[dict]:

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dataset = RagDataset(**data)

    results = []

    for question in dataset.rag_questions:

        result = answer_question(
            question=question.question,
            k=k
        )

        result["question_id"] = question.question_id

        results.append(result)

def save_generation_results(
    results: list[dict],
    dataset_path: str,
    save_dir: str
) -> Path:

    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / Path(dataset_path).name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    return output_path

def run_generation_dataset(
        dataset_path: str,
        save_directory: str,
        k: int = 3
    ) -> Path:

    results = answer_dataset(
        dataset_path=dataset_path,
        k=k
    )

    output_path = save_generation_results(
        results=results,
        dataset_path=dataset_path,
        save_dir=save_directory
    )

    return output_path


if __name__ == "__main__":
    output_path = run_generation_dataset(
        dataset_path="data/datasets/test.json",
        save_directory="data/output/generation_results",
        k=3
    )

    print("Saved to:", output_path)