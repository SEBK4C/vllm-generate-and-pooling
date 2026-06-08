# vLLM Generate + Pooling Hybrid Research Fork

This repository is the `vllm-generate-and-pooling` research fork of
[vLLM](https://github.com/vllm-project/vllm). It keeps the core vLLM serving
stack and adds an experimental path for one resident generative model to serve
both generation and LAST-token hidden-state pooling from the same logical model
load.

The immediate target is Gemma 4 12B QAT:

```text
google/gemma-4-12B-it-qat-w4a16-ct
```

The long-term goal is a single vLLM server process with one Gemma 4 model copy in
GPU memory serving:

1. OpenAI-compatible chat/completions.
2. OpenAI-compatible embeddings from normalized full-size hidden states.
3. A best-effort Gemma multimodal generation transcription shim.

This is not upstream vLLM's default behavior. It is a research path for exploring
whether embeddings can be treated as a second request type over shared prefill
machinery instead of as a second model instance.

## What this fork changes

Current upstream vLLM generally treats `generate` and `pooling` as separate runner
modes. This fork adds an explicit experimental flag:

```bash
--enable-generate-and-pooling
```

With that flag enabled, the server keeps the model on the generation runner and
adds a lightweight pooling path over the final hidden states. The initial pooling
implementation is intentionally conservative:

- Pooling task: `embed` only.
- Pooling type: `LAST` token only.
- Output size: `3840` dimensions only for the Gemma 4 12B target.
- Normalization: required.
- Matryoshka/truncation: rejected.
- Mixed generation+pooling microbatches: rejected until the scheduler milestone
  implements a safe task-homogeneous batching policy.
- `--async-scheduling`: still experimental for this hybrid path and not required
  in the baseline command.

LAST-token normalized hidden-state embeddings are an untrained research baseline.
Do not treat them as production-quality retrieval embeddings until the W4A16 QAT
quality gate has passed against BF16/reference embeddings on the target retrieval
workload.

## Install from this repository

Use the fork checkout as the install path. From the repository root:

```bash
cd /path/to/vllm-generate-and-pooling
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

If you are developing C/C++ kernels or rebuilding native extensions, omit
`VLLM_USE_PRECOMPILED=1`:

```bash
uv pip install -e . --torch-backend=auto
```

## Run the hybrid server

Baseline command for the Gemma 4 12B QAT target:

```bash
vllm serve google/gemma-4-12B-it-qat-w4a16-ct \
  --runner generate \
  --enable-generate-and-pooling \
  --pooling-task embed \
  --pooling-output-dim 3840 \
  --pooling-normalize true \
  --pooling-type LAST \
  --enable-generative-audio-transcription \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image": 4, "audio": 1}' \
  --host 0.0.0.0 \
  --port 8000
```

Do not add `--async-scheduling` to the required baseline until mixed
`SamplingParams` and `PoolingParams` traffic has passed the M4 scheduler SLOs.

## Use the endpoints

### Chat completions

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-12B-it-qat-w4a16-ct",
    "messages": [
      {"role": "user", "content": "Explain solar panel degradation in one paragraph."}
    ],
    "max_tokens": 128
  }'
```

### Embeddings

```bash
curl http://localhost:8000/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-12B-it-qat-w4a16-ct",
    "input": "Represent this query for retrieval: solar panel degradation",
    "encoding_format": "float"
  }'
```

Expected embedding behavior in this research path:

- One embedding per input item.
- Length is `3840`.
- Values are finite floats.
- Vector norm should be approximately `1.0`.
- No generated text is returned.

`encoding_format: "base64"` should follow the existing vLLM/OpenAI-compatible
embedding serialization path.

### Pooling debug endpoint

```bash
curl http://localhost:8000/pooling \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "google/gemma-4-12B-it-qat-w4a16-ct",
    "input": "debug the raw pooling path",
    "encoding_format": "float"
  }'
```

### Audio transcription

The current flag advertises the planned Gemma generative transcription path:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=google/gemma-4-12B-it-qat-w4a16-ct \
  -F file=@sample.wav \
  -F response_format=json
```

The transcription shim is best-effort multimodal generation, not Whisper-compatible
ASR. Timestamp, segment, verbose JSON, and logprob options must return clear HTTP
400 errors unless they are explicitly implemented later.

## Validation gates before claiming success

Before treating this as complete, run and document:

1. W4A16 QAT embedding quality against BF16/reference embeddings.
2. VRAM comparison between one hybrid instance and two independent model-serving
   instances.
3. `/v1/chat/completions` correctness.
4. `/v1/embeddings` shape, finite-value, and normalization checks.
5. `/v1/audio/transcriptions` Gemma-shim correctness after the shim is fully
   implemented.
6. Mixed chat+embedding load test with concrete p99 inter-token and embedding
   latency SLOs.
7. Logs confirming one logical Gemma model load and no duplicate model copy.

See `docs/contributing/generate-and-pooling-implementation-plan.md` for the
milestone plan and `docs/contributing/generate-and-pooling-agent-guide.md` for
agent operating rules.

## What remains from upstream vLLM

This fork still uses vLLM's core strengths:

- PagedAttention for efficient KV-cache memory management.
- Continuous batching, chunked prefill, and prefix caching.
- CUDA/HIP graph execution paths.
- Quantization support including compressed-tensors checkpoints.
- OpenAI-compatible serving APIs.
- Hugging Face model integration.
- Tensor, pipeline, data, expert, and context parallelism.
- Multimodal model support where the underlying model and renderer support it.

For general vLLM documentation, see <https://docs.vllm.ai/>. For the upstream
project, see <https://github.com/vllm-project/vllm>.

## Contributing

This fork is research-oriented. Agents and humans should read the hybrid guide
before changing runner, scheduler, pooling, embedding, or transcription code:

```text
docs/contributing/generate-and-pooling-agent-guide.md
```

If a change is intended for upstream vLLM, split it into a minimal upstreamable
patch and follow the upstream contribution process.

## Citation

If you use vLLM for research, cite the original vLLM paper:

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```
