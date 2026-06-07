# Generate and Pooling Implementation Plan

This fork explores one resident Gemma 4 model serving generation, embeddings,
`/pooling`, and best-effort generative transcription. The production target is
`google/gemma-4-12B-it-qat-w4a16-ct`, but the implementation must fail closed if
W4A16 hidden-state embeddings do not pass the retrieval quality gate.

## Current Path Map

Treat path names as current observations, not stable APIs:

- CLI `vllm serve` arguments flow through OpenAI CLI args into
  `EngineArgs`/`AsyncEngineArgs`, then into `ModelConfig`.
- `ModelConfig` resolves `runner_type`; this patch keeps hybrid mode on the
  `generate` runner and adds explicit feature flags instead of adding a new
  runner enum.
- `AsyncLLM.get_supported_tasks()` asks `EngineCore`, which asks the model
  executor, which asks each GPU worker/model runner.
- `api_server.build_app()` and `init_app_state()` already register generation,
  speech-to-text, and pooling routers/states based on the advertised
  `supported_tasks` tuple.
- The V2 GPU runner already separates model forward, `sample_tokens()`, and
  `pool()`, making it the safest first runner for hybrid dispatch.

## Milestones

1. **M0 quality gate:** benchmark normalized LAST-token hidden states from the
   W4A16 checkpoint against BF16/reference embeddings. Stop if retrieval
   quality is below the predeclared threshold.
2. **M1 task union:** expose `--enable-generate-and-pooling` and advertise
   generation plus `embed` without destructive `--convert embed` conversion.
3. **M2 dual runner init:** initialize both sampler and lightweight LAST-token
   pooling helpers for the same loaded model when hybrid mode is enabled.
4. **M3 encode dispatch:** route homogeneous pooling scheduler steps to
   `pool()` and homogeneous generation steps to `sample_tokens()`.
5. **M4 scheduling SLOs:** split or prioritize mixed traffic so chat p99
   inter-token latency and embedding p99 latency meet concrete SLOs. Until then,
   mixed generation/pooling microbatches are rejected rather than executed
   unsafely.
6. **M5 transcription shim:** replace ASR/Whisper-shaped assumptions with a
   Gemma multimodal generation shim that returns clear 400 errors for unsupported
   timestamps, segments, verbose JSON, and logprob options.
7. **M6 value proof:** compare VRAM against two independent model copies and
   verify one logical Gemma model load.

## Initial Patch Scope

The first code patch starts M1-M3 for the V2 runner:

- Adds hybrid CLI/config flags.
- Defaults hybrid pooling to `embed`, `LAST`, normalized full-size 3840 output.
- Rejects Matryoshka/truncated embedding dimensions in hybrid mode.
- Advertises `generate` plus `embed`, and optionally `transcription`, from the
  same runner.
- Initializes sampler and `PoolingRunner` together for hybrid V2 execution.
- Adds per-scheduler-step generation/pooling flags and rejects mixed hybrid
  microbatches until the M4 scheduler policy is implemented.
- Adds executor and engine-core `pool()` dispatch so pooling-only hybrid steps
  consume the same forward hidden states without calling the sampler or MTP
  proposal path.

## Known Limitations

- LAST-token normalization is an untrained research baseline, not a validated
  retrieval product.
- The transcription flag currently advertises the endpoint for the planned shim;
  full Gemma generative transcription still requires the M5 serving-path work.
- `--async-scheduling` remains experimental for hybrid mode until M4 mixed-load
  tests prove both `SamplingParams` and `PoolingParams` are safe.
- Mixed generation/pooling microbatches intentionally fail closed in this patch.
