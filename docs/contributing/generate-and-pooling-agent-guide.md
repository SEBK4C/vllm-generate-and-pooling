# Generate and Pooling Research Agent Guide

> Follow this guide when working on the `vllm-generate-and-pooling` fork's
> hybrid Gemma 4 serving effort: one resident model serving generation,
> embeddings, and generative transcription.

## Operating Rules

- Start from the companion implementation plan in
  `docs/contributing/generate-and-pooling-implementation-plan.md` and keep it
  updated when changing milestone scope or known limitations.
- Treat file paths in research prompts as hypotheses. Before editing, locate the
  current modules with `rg` searches for `runner_type`, `supported_tasks`,
  `PoolingParams`, `SamplingParams`, `OpenAIServingTranscription`,
  `PoolingRunner`, `AsyncLLM.encode`, and `GPUModelRunner`. Document the actual
  path mapping in the research report.
- Run an embedding-quality go/no-go gate before scheduler or runner work.
  Compare W4A16 hidden-state embeddings from
  `google/gemma-4-12B-it-qat-w4a16-ct` against BF16 or other reference
  embeddings on representative retrieval tasks. If QAT embeddings fail the
  threshold, pause and report that the single-copy QAT design may not satisfy
  the product goal.
- Describe LAST-token plus normalization embeddings as an untrained research
  baseline until retrieval benchmarks prove production quality.
- Define concrete M4 latency SLOs before implementation. At minimum, measure
  chat p99 inter-token latency under concurrent embedding load and embedding
  p99 latency under active chat traffic. Evaluate decode-priority scheduling,
  prefill chunking, task-homogeneous microbatches, and fairness across request
  types.
- Treat `--async-scheduling` as experimental for the hybrid path until verified
  with both `SamplingParams` and `PoolingParams`. Do not make it part of the
  required baseline until M4 passes.
- Treat the Gemma transcription shim as best-effort multimodal generation, not
  Whisper-compatible ASR. Unsupported timestamp, segment, verbose JSON, or
  logprob options must return clear HTTP 400 errors.
- Include a VRAM comparison in the definition of done: the hybrid server must
  show one logical Gemma model load and materially less memory than two
  independent model copies.
