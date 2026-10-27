# MiniMind Chinese-to-English Translation Agent

A comprehensive Chinese→English translation system that combines a local vLLM deployment with an intelligent, auditable agent workflow.  
This repo provides a scalable vLLM deployment, a LangGraph-based translation pipeline with multi-pass validation/fixing, and tools to produce a **MiniMind-style English dataset** suitable for pretraining, instruction tuning, RAG indexing and multilingual benchmarking.

## Motivation — why this project exists

Although `jingyaogong/minimind_dataset` is an impressive Chinese dataset, many teams, benchmarks, and models prefer or require high-quality English instruction-style data. This project builds a faithful English counterpart that:

- Preserves original metadata, conversation structure, and instruction intent so comparisons to the Chinese original remain valid.
- Enables English-only teams and benchmark suites to use MiniMind-style content without language barriers.
- Helps research on language-specific behavior, bias, and cross-lingual performance by providing a parallel dataset.
- Ships a reusable, auditable pipeline (vLLM + LangGraph + validation) that dataset maintainers can adopt or adapt.

**Goals**
- Produce high-fidelity English dataset for pretraining and instruction tuning.
- Maintain provenance and licensing clarity.
- Flag low-confidence translations for human review.
- Provide JSONL output compatible with standard LLM training stacks.

## 🚀 Features

- **vLLM Deployment**: Docker-based deployment of Qwen3-4B-Instruct model with GPU acceleration
- **Intelligent Translation Agent**: LangGraph-powered workflow with translation validation and fixing
- **Chinese Text Detection**: Automatic detection of Chinese characters using regex patterns
- **Quality Assurance**: Multi-pass translation with validation and correction steps
- **Batch Processing**: Efficient concurrent processing of large datasets
- **OpenAI Compatible API**: Standard OpenAI API interface for easy integration

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [vLLM Deployment Setup](#vllm-deployment-setup)
- [Agent Setup](#agent-setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

## 🔧 Prerequisites

- Docker and Docker Compose
- NVIDIA GPU with CUDA support
- Python 3.12+
- At least 8GB GPU memory for Qwen3-4B model

## 🐳 vLLM Deployment Setup

### 1. GPU Configuration

The deployment is configured to use GPU #1. Update the device ID in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          device_ids: ["GPU-9997ab17-e4ca-cd9f-3786-01659de406f0"]  # Update this
          capabilities: ["gpu"]
```

### 2. Volume Configuration

Update the cache paths in `docker-compose.yml` for your system:

```yaml
volumes:
  # Windows example - update paths as needed
  - "T:\\hf-cache\\hub:/root/.cache/huggingface/hub"
  - "T:\\vllm-cache:/root/.cache/vllm"
```

### 3. Start vLLM Server

```bash
docker-compose up -d
```

The vLLM server will be available at `http://localhost:8000` with OpenAI-compatible API endpoints.

### 4. Verify Deployment

Check if the model is loaded:

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer my-token-1234"
```

## 🤖 Agent Setup

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv sync --locked
```

### 2. Environment Configuration

Create a `.env` file with the following configuration:

```env
# vLLM Server Configuration
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_API_KEY=my-token-1234
MODEL_ID=Qwen/Qwen3-4B-Instruct-2507-FP8

# Processing Configuration
CONCURRENCY=10
OUTPUT_PATH=./out/pretrain_hq.en.jsonl
INPUT_FILE=pretrain_hq.jsonl

# Retry and validation policy
REQUEST_ATTEMPTS=6
REQUEST_TIMEOUT=120
RETRY_BASE=1
RETRY_MAX=60
MAX_TOKENS=2048
MAX_FIXES=2
# Remove any old CHECK_POINT setting; resume is automatic.
```

## ⚙️ Configuration

The agent supports the following configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | vLLM server endpoint |
| `OPENAI_API_KEY` | `my-token-1234` | API key for authentication |
| `MODEL_ID` | `Qwen/Qwen3-4B-Instruct-2507-FP8` | Model identifier |
| `CONCURRENCY` | `10` | Number of concurrent translation tasks |
| `OUTPUT_PATH` | `./out/pretrain_hq.en.jsonl` | Output file path |
| `INPUT_FILE` | `pretrain_hq.jsonl` | Local input JSONL file path |
| `REQUEST_ATTEMPTS` | `6` | Maximum HTTP attempts per translation/correction segment, including the first |
| `REQUEST_TIMEOUT` | `120` | OpenAI SDK request timeout in seconds |
| `RETRY_BASE` / `RETRY_MAX` | `1` / `60` | Exponential full-jitter backoff base/cap in seconds; Retry-After is honored up to the cap |
| `MAX_TOKENS` | `2048` | Output token budget per segment; length-truncated responses fail validation |
| `MAX_FIXES` | `2` | Maximum correction passes for residual Chinese |
| `CHECK_POINT` | `0` | Legacy skip control: nonzero values are rejected to prevent data loss |

## 🎯 Usage

### Basic Translation

1. **Prepare Input Data**: Create a JSONL file with records containing a `text` field:
   ```json
   {"text": "这是一个中文句子。"}
   {"text": "Another English sentence that will remain unchanged."}
   {"text": "混合的中文和English text."}
   ```

2. **Run the Translation Pipeline**:
   ```bash
   python -m app.run_pipeline
   ```

3. **Output**: The agent will create an output file with translated text:
   ```json
   {"text": "This is a Chinese sentence."}
   {"text": "Another English sentence that will remain unchanged."}
   {"text": "Mixed Chinese and English text."}
   ```

### Agent Workflow

The agent implements a sophisticated workflow:

1. **Detection**: Checks if text contains Chinese characters
2. **Translation**: Translates Chinese text to English using the system prompt
3. **Validation**: Verifies that translation doesn't contain Chinese characters
4. **Fixing**: If Chinese characters remain, applies a fixing pass
5. **Durable result**: Commits the source/translation pair to SQLite; failures stay explicit and retryable
6. **Publication**: Exports the English JSONL in source order only when every record succeeds

### Completeness, overload recovery, and paired output

The pipeline snapshots and validates the entire local JSONL input before sending requests.
Every nonblank JSONL record must be an object containing a string `text`; malformed input
aborts before translation. Empty strings and duplicate records are retained. The zero-based
`source_id` identifies a record's position among nonblank JSONL records, **not** its text content.

A bounded queue feeds `CONCURRENCY` workers, so the runner never creates a task for every
input row at once. HTTP 408/409/429, 5xx, connection errors and timeouts retry with exponential
backoff and jitter. Empty responses and non-`stop` finish reasons also retry. Other HTTP 4xx
errors fail immediately. Retry exhaustion or residual Chinese after correction marks the
record **failed**, never successful with blank/original Chinese fallback. Other records can
finish; the process exits nonzero if any record remains unsuccessful.

For `OUTPUT_PATH=./out/pretrain_hq.en.jsonl`, the artifacts are:

| File | Purpose |
|---|---|
| `pretrain_hq.en.jsonl.sqlite` | Durable source snapshot, successful translations, failure details and record-attempt counts; retain for resume |
| `pretrain_hq.en.jsonl.pairs.jsonl` | Exactly one entry for every source record after a finished/gracefully interrupted run, including failed/pending entries |
| `pretrain_hq.en.jsonl.summary.json` | `running`, `incomplete`, or `complete`, input SHA-256, total/success/failed/pending counts |
| `pretrain_hq.en.jsonl` | Final training dataset, published only when all source records have validated translations |

Example successful pair (hash shortened here):

```json
{"source_id":0,"source_sha256":"...","source":{"text":"你好","category":"greeting"},"text_en":"Hello","status":"success","attempts":1,"error":null}
```

Failed/pending entries have `text_en: null`; failed entries include an error. `attempts`
counts completed record attempts across runs, not individual HTTP calls. Source hashes cover
the canonical full source record, including metadata. Export checks contiguous IDs, source
hashes, translation validity and equal counts. Training rows preserve the original metadata
and replace only `text`; row N corresponds to source row N, even when requests finish out of
order. Plain text remains plain text; existing conversation boundary tokens and standard role
headers are preserved outside model control.

To resume after overload, an interruption, or a server restart, run the same command again:

```bash
uv run --locked python -m app.run_pipeline
```

Only failed/pending records are retried. Completed results are reused, and files are rebuilt
in source order rather than appended, so reruns do not duplicate rows. SQLite commits each
result with full synchronous durability. A process crash before a result is committed may
repeat that model request, but cannot duplicate the final row. An OS file lock rejects a
second writer using the same output path. Use a local filesystem with reliable SQLite/OS
locking; retain the SQLite database and any active WAL sidecars together.

The full input-byte hash and model/prompt/token-budget/correction configuration must match
on resume. Concurrency, retry counts, timeout and endpoint may be changed to recover an
overloaded server; use the same actual model behind the endpoint. Changed source data or
translation settings require a new output path. Do not edit the input while a run is active:
translation uses the validated snapshot.

**Migration from older versions:** remove `CHECK_POINT` (the old default silently skipped
10,000 rows) and choose a fresh `OUTPUT_PATH`. Old append-only results have no source IDs
and may be reordered or incomplete, so they are rejected as an automatic resume source.
Keep them for comparison; regenerate from the complete original input.

Atomic replacement prevents partially written JSONL files. `summary.status == "complete"`
is the publication signal and is written last; the files are not a multi-file transaction.
After an abrupt kill, sidecar exports may be stale or absent: rerun to rebuild them from
SQLite. On a failed resume, an older complete training file can remain on disk; consumers
must check the summary, not just file existence.

These checks guarantee record accounting and structural completeness, not semantic
translation accuracy. Review sampled translations separately. Persistently unavailable
vLLM or invalid completions leave an explicit incomplete run rather than fabricated pairs.

### LangGraph version

The runtime is pinned to **LangGraph 1.2.17**, the latest stable release verified on
2026-10-27 ([PyPI](https://pypi.org/project/langgraph/1.2.17/),
[upstream releases](https://github.com/langchain-ai/langgraph/releases)). The existing
`StateGraph` API remains in use; Python 3.12+ is still required by this project.
`uv.lock` and hash-pinned `requirements.txt` are regenerated together.

### Direct API Testing

Test the vLLM deployment directly:

```bash
# Simple completion
python test.py

# Streaming completion
python test-streaming.py
```

## 🧪 Testing

Run offline regression tests against the real LangGraph and OpenAI SDK packages:

```bash
uv sync --locked
uv run --locked python -m unittest discover -s tests -v
uv pip check
```

The tests simulate HTTP overload/timeouts, retry exhaustion, invalid completions,
interruptions, resume, duplicate input rows, reordered completions, metadata retention,
input changes, legacy checkpoints and ledger corruption. CI runs Python 3.12 and 3.13
using the hash-pinned requirements. These tests do not replace a real GPU/vLLM load test.


### Test vLLM Connection

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="my-token-1234"
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B-Instruct-2507-FP8",
    messages=[{"role": "user", "content": "Translate: 你好世界"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

### Test Agent Components

```python
from app.detector import has_chinese
from app.vllm_client import OpenAIClient

# Test Chinese detection
print(has_chinese("Hello world"))  # False
print(has_chinese("你好世界"))      # True

# Test vLLM client
client = OpenAIClient()
result = await client.chat("Translate to English:", "你好")
print(result)  # Should output English translation
```

## 📁 Project Structure

```
├── app/                          # Agent implementation
│   ├── config.py                # Configuration management
│   ├── detector.py              # Chinese text detection
│   ├── graph.py                 # LangGraph workflow definition
│   ├── ledger.py                # Durable pairs, integrity checks and atomic exports
│   ├── run_pipeline.py          # Main pipeline runner
│   └── vllm_client.py           # OpenAI-compatible vLLM client
├── prompts/                     # System prompts
│   ├── translate_system.txt     # Translation prompt
│   └── fix_system.txt          # Translation fixing prompt
├── docker-compose.yml           # vLLM deployment configuration
├── pyproject.toml              # Python project configuration
├── requirements.txt            # Python dependencies
├── test.py                     # vLLM API test script
├── test-streaming.py           # Streaming API test script
└── README.md                   # This file
```

## 🔍 Key Components

### vLLM Deployment
- **Model**: Qwen3-4B-Instruct-2507-FP8 (optimized for inference)
- **GPU Memory**: 92% utilization for maximum efficiency
- **Concurrency**: Up to 48 sequences for high throughput
- **API**: OpenAI-compatible endpoints for seamless integration

### Translation Agent
- **Framework**: LangGraph for robust workflow management
- **Detection**: Regex-based Chinese character detection
- **Translation**: Professional translation with context awareness
- **Quality Control**: Multi-pass validation and correction
- **Performance**: Concurrent processing with configurable limits

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🆘 Troubleshooting

### Common Issues

1. **GPU Not Detected**: Ensure NVIDIA drivers and Docker GPU support are properly configured
2. **Out of Memory**: Reduce `max_num_seqs` or `gpu_memory_utilization` in docker-compose.yml
3. **Connection Refused**: Verify vLLM container is running: `docker-compose ps`
4. **Translation Quality**: Adjust temperature and max_tokens in the configuration

### Performance Tuning

- **Concurrency**: Adjust `CONCURRENCY` based on your GPU memory and processing needs
- **Batch Size**: Modify `max_num_seqs` in docker-compose.yml for optimal throughput
- **Memory Usage**: Tune `gpu_memory_utilization` based on available GPU memory

For more detailed troubleshooting, check the Docker logs:
```bash
docker-compose logs vllm
```