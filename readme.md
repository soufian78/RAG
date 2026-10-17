*This project has been created as part of the 42 curriculum by skabny.*

# RAG Against the Machine

## Description

**RAG Against the Machine** is a Retrieval-Augmented Generation (RAG) project designed to answer questions about technical documentation and source code.

The goal is to build a system that retrieves relevant information from a local knowledge base and uses a language model to generate answers grounded in the retrieved context.

Instead of asking a language model to answer questions using only its pretrained knowledge, the system searches a collection of documents first. It then provides the most relevant text to the model, helping it produce answers based on the available information.

The project uses the vLLM documentation and source code as its knowledge base.

### Main features

- Index Python source files and Markdown documentation.
- Split documents into smaller, searchable chunks.
- Retrieve relevant chunks using BM25.
- Rank results using relevance scores and additional weighting.
- Generate context-based answers with a local language model.
- Evaluate retrieval quality using Recall@k.
- Support searches over documentation and source-code datasets.

## System Architecture

The project follows a Retrieval-Augmented Generation pipeline composed of two main stages: indexing and question answering.

### 1. Indexing pipeline

The indexing stage prepares the knowledge base before questions are asked.

1. **Document loading:** The indexer recursively reads Python (`.py`) and Markdown (`.md`) files.
2. **Chunking:** Each document is split into smaller chunks while preserving useful structural boundaries.
3. **Tokenization:** Text is converted into searchable terms.
4. **Index construction:** A BM25 index is built from the chunks, with additional weighting for filenames and Markdown headings.
5. **Storage:** The chunks and search index are saved for later use.

### 2. Question-answering pipeline

When a user submits a question, the system performs the following steps:

1. **Question tokenization:** The question is converted into terms compatible with the retrieval index.
2. **Retrieval:** BM25 scores the indexed chunks against the question.
3. **Ranking:** The most relevant chunks are selected, taking into account the additional ranking rules.
4. **Context construction:** The selected chunks are combined into a context containing their file paths and text.
5. **Prompt construction:** A prompt instructs the language model to answer using only the supplied context.
6. **Answer generation:** Qwen3-0.6B generates the response.
7. **Evaluation:** Retrieved sources and generated answers can be saved in JSON format for evaluation.

### Architecture diagram

```text
                 INDEXING
                    |
          Python and Markdown files
                    |
              Document loader
                    |
                 Chunking
                    |
                Tokenization
                    |
             BM25 index builder
                    |
         Saved chunks and BM25 index
                    |
                    v
              USER QUESTION
                    |
              Tokenization
                    |
             BM25 retrieval
                    |
           Ranking and top-k
                    |
             Context builder
                    |
              Prompt builder
                    |
            Qwen3-0.6B model
                    |
             Generated answer
                    |
            JSON / Evaluation
```

The indexing stage is performed before retrieval. The saved index can then be loaded and reused for multiple queries without rebuilding it for every question.

## Chunking Strategy

Chunking is an important part of the project because retrieval quality depends on how documents are divided.

If chunks are too small, important information may be separated from its context. If chunks are too large, retrieval can return irrelevant information and consume more of the language model's context window.

The project uses different strategies for Python source code and Markdown documentation.

### Python files

Python files are parsed using Python's Abstract Syntax Tree (AST), through the `ast` module.

The chunker uses top-level syntax nodes and their line positions to identify meaningful code sections. This helps preserve functions, classes, and other structural elements instead of splitting the code at arbitrary character positions.

Large sections are divided further when necessary to respect the configured chunk-size limit.

### Markdown files

Markdown documents are divided using structural boundaries:

- Markdown headings.
- Paragraph boundaries.
- A maximum chunk size.

This approach helps keep related documentation together and preserves useful information about the section in which a chunk appears.

### Chunk metadata

Each chunk stores:

- `file_path`: the original file path.
- `start`: the starting position in the source text.
- `end`: the ending position in the source text.
- `content`: the text contained in the chunk.

This metadata makes it possible to identify the original source of retrieved information and convert chunks into source references for evaluation.

The indexer uses a configurable maximum chunk size, with a limit of 100 to 2,000 characters in the current implementation.

## Retrieval Method

The project uses **BM25 (Best Matching 25)** as its lexical retrieval algorithm.

BM25 is a ranking function commonly used in information retrieval. It compares a query with documents by considering term frequency, inverse document frequency, and document length.

### How BM25 works

1. The question is tokenized into terms.
2. The system compares these terms with the tokens in each indexed chunk.
3. BM25 calculates a relevance score for each chunk.
4. The chunks are sorted by their scores.
5. The system returns the top-k results.

A higher score generally indicates a stronger lexical match between the question and the chunk.

### Ranking improvements

Several additional mechanisms improve the relevance of retrieved chunks:

- **Filename weighting:** Terms found in filenames receive additional weight because filenames often identify the subject of a document.
- **Markdown heading weighting:** Terms from the last relevant Markdown heading receive additional weight because headings summarize the topic of a section.
- **Documentation boost:** Chunks whose paths contain `/docs/` receive an additional score boost.
- **Tokenization improvements:** The tokenizer handles lowercase text, underscores, and code-related naming patterns to improve matching between questions and technical content.

These mechanisms supplement BM25 rather than replacing it.

### Why BM25?

BM25 was selected because it is lightweight, fast, and suitable for a local documentation search system. It provides a useful baseline without requiring an embedding model or a separate vector database.

Its main limitation is that it relies primarily on lexical overlap. A question and a relevant chunk may use different words to describe the same concept, making semantic matches more difficult.

## Performance Analysis

The system is evaluated using **Recall@k**, which measures whether relevant sources appear among the top-k retrieved results.

Recall@k is calculated as:

\[
\operatorname{Recall@k} =
\frac{\text{Questions with a relevant source in top-k}}
{\text{Total questions}}
\]

For example, Recall@5 measures how often at least one relevant source is found within the first five results.

### Documentation dataset

The latest reported results on the documentation dataset contain 100 questions.

| Metric | Score |
|---|---:|
| Recall@1 | 61.0% |
| Recall@3 | 79.0% |
| Recall@5 | 83.0% |
| Recall@10 | 84.0% |

<text></text>

The results show that the system retrieves at least one relevant source for 83% of questions within the top five results.

Increasing k from 5 to 10 improves recall by only one percentage point. This suggests that most of the retrieval improvement is achieved within the first five results on this dataset.

### Source-code dataset

The latest reported results on the source-code dataset contain 99 questions.

| Metric | Score |
|---|---:|
| Recall@5 | 44.4% |
| Recall@10 | 47.5% |

<text></text>

The lower scores indicate that retrieving relevant source-code sections is more challenging than retrieving documentation sections with the current configuration.

### Analysis

The evaluation process helped identify several issues:

- Incorrect source paths caused relevant chunks to be missed.
- Duplicate or incorrectly structured paths affected the indexing process.
- Basic tokenization did not always match technical identifiers effectively.
- Filename and heading weighting improved the ranking of relevant chunks.
- The source-code dataset remains an area for improvement.

The main optimization objective is to improve Recall@5 while keeping retrieval efficient.

*Note: These figures represent the latest reported evaluation results during development. They should be recalculated after any changes to the indexer, tokenizer, ranking logic, or datasets.*

## Design Decisions

### Local language model

The project uses `Qwen/Qwen3-0.6B` through Hugging Face Transformers.

A small model makes local experimentation more accessible and avoids relying on a hosted generation API. However, its relatively limited capacity can affect answer quality and inference speed.

### BM25 instead of vector search

BM25 was chosen as a lightweight starting point. It is straightforward to implement, easy to evaluate, and effective when questions share terminology with the relevant documentation.

A future version could compare BM25 with dense retrieval or a hybrid approach combining lexical and semantic search.

### Pydantic data models

Pydantic models define the structure of chunks, questions, source references, and evaluation results.

They help validate data and maintain consistent JSON structures between the indexing, retrieval, generation, and evaluation stages.

### Separation of responsibilities

The project separates its main responsibilities into modules:

- **Indexer:** Loads documents, chunks text, tokenizes content, and builds the index.
- **Retriever:** Loads the index, ranks chunks, and returns relevant sources.
- **Generator:** Builds context and prompts, then generates answers.
- **Models:** Defines the data structures used across the pipeline.
- **CLI:** Exposes the project operations through command-line commands.

This separation makes the system easier to test, understand, and improve.

### Context-grounded generation

The generation prompt instructs the model to use only the retrieved context and to avoid inventing information.

When the provided context does not support an answer, the intended fallback is:

`I don't know based on the provided context.`

This approach aims to reduce unsupported answers, although prompt instructions alone cannot guarantee that the model will never hallucinate.

## Challenges Faced

### 1. Incorrect file paths

**Problem:** Incorrect or duplicated source paths caused retrieval and evaluation problems.

**Solution:** The indexing paths were corrected, and duplicate path issues were checked before evaluating the system again.

### 2. Improving Recall@k

**Problem:** Initial retrieval results were weak, especially when relevant sources were not ranked highly.

**Solution:** The BM25 implementation was improved through tokenizer changes, filename weighting, Markdown heading weighting, and a documentation-path boost.

### 3. Tokenizing technical identifiers

**Problem:** Technical identifiers can contain underscores, CamelCase names, and abbreviations. Basic whitespace tokenization does not handle these patterns well.

**Solution:** Tokenization was improved to recognize more useful terms within technical text and code identifiers.

### 4. Source-code retrieval

**Problem:** The source-code dataset produced lower Recall@k scores than the documentation dataset.

**Solution:** The project uses the same evaluation process to identify this gap. Further improvements could include AST-aware retrieval, better identifier normalization, and combining lexical retrieval with semantic search.

### 5. Slow answer generation

**Problem:** Local language-model inference can be slow, especially when running on a CPU.

**Solution:** The implementation uses a small language model and limits the number of generated tokens. Further work could focus on inference optimization, prompt length, and model configuration.

## Instructions

### Requirements

- Python 3.13 or a compatible Python version.
- [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.
- Sufficient disk space and memory for the language model and project dependencies.
- Access to download the model on the first run, unless it is already cached.

### Installation

Clone the repository and enter the project directory:

```bash
git clone <YOUR_REPOSITORY_URL>
cd rag
```

Install the project dependencies:

```bash
uv sync
```

Make sure the raw documentation and source code are available in the expected directory:

```text
data/
└── raw/
    └── vllm-0.10.1/
```

The exact directory structure should match the paths configured in the project.

### Build the index

Run the index command using the project's CLI entry point:

```bash
uv run python -m src.main index --raw_path data/raw --max_chunk 2000
```

This command reads the supported files, creates chunks, builds the BM25 index, and saves the processed data.

### Search for relevant chunks

Search the indexed knowledge base:

```bash
uv run python -m src.main search --query "What does the hello function do?" --k 5
```

The command should return the most relevant chunks for the question.

### Search a dataset

Run retrieval over a question dataset:

```bash
uv run python -m src.main search_dataset
```

The exact dataset paths and other parameters can be configured according to the CLI implementation.

### Generate answers

Run the dataset-answering command:

```bash
uv run python -m src.main answer_dataset
```

This runs the retrieval and generation workflow and saves the resulting answers and source references.

**Important:** Replace `src.main` with the actual Python module containing `fire.Fire(CLI)` if your repository uses a different entry point. The command options and defaults should match the current CLI implementation.

## Example Usage

Suppose the user asks:

> What does the hello function do?

The pipeline retrieves relevant chunks from the indexed files. The generator then receives the question and retrieved context.

An illustrative answer could be:

```text
The hello function prints a message.
```

This is an example of the expected interaction, not a guarantee of the output for every index or model configuration.

If the retrieved context does not contain the answer, the system is instructed to return:

```text
I don't know based on the provided context.
```

## Evaluation

The project includes an evaluation tool for measuring retrieval performance against the expected source references.

For example, evaluate the documentation dataset with k = 10 and a maximum context length of 2,000:

```bash
./moulinette-bin evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 \
  --max_context_length 2000
```

Use the corresponding source-code dataset files when evaluating code retrieval.

The evaluation results can be used to compare changes to the tokenizer, chunking strategy, and ranking mechanism.

## Resources

### Documentation and technical references

- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) — Original research introducing retrieval-augmented generation for knowledge-intensive NLP tasks.
- [Okapi BM25](https://en.wikipedia.org/wiki/Okapi_BM25) — Overview of the BM25 ranking function.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — Documentation for loading and running language models.
- [Qwen model collection](https://huggingface.co/Qwen) — Models and documentation from the Qwen family.
- [Pydantic documentation](https://docs.pydantic.dev/) — Data validation and Python model definitions.
- [Python `ast` documentation](https://docs.python.org/3/library/ast.html) — Abstract Syntax Trees used to analyze Python source code.
- [Python `re` documentation](https://docs.python.org/3/library/re.html) — Regular expressions used for text processing.
- [uv documentation](https://docs.astral.sh/uv/) — Python package and environment management.
- [vLLM documentation](https://docs.vllm.ai/) — Documentation used as the project's knowledge base.

### Use of AI

AI tools were used as development assistance during the project.

Their use included:

- **Understanding concepts:** Explaining RAG architecture, BM25, tokenization, chunking, and Recall@k.
- **Debugging:** Helping investigate indexing, path handling, and retrieval issues.
- **Code improvement:** Discussing possible improvements to tokenization, ranking, prompts, and inference performance.
- **Documentation:** Assisting with docstrings and the organization and writing of this README.

AI-generated suggestions were treated as assistance rather than authoritative results. Implementation decisions and retrieval performance were checked against the project code and evaluation outputs.

The retrieval results reported in this README come from the project's recorded evaluation runs, not from AI-generated estimates.

## Future Improvements

Potential improvements include:

- Improve Recall@5 on the source-code dataset.
- Experiment with chunk sizes and overlap strategies.
- Add AST-aware search for functions, classes, and identifiers.
- Compare BM25 against dense and hybrid retrieval.
- Optimize local language-model inference.
- Add automated tests for indexing, retrieval, and answer generation.
- Evaluate answer correctness and groundedness in addition to retrieval recall.

## Author

`skabny`

Student at 1337.