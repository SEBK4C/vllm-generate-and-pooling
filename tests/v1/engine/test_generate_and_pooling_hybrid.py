# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.engine.arg_utils import EngineArgs
from vllm.pooling_params import PoolingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_generate_and_pooling_cli_flags_parse():
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())

    args = parser.parse_args(
        [
            "--enable-generate-and-pooling",
            "--pooling-task",
            "embed",
            "--pooling-output-dim",
            "3840",
            "--pooling-normalize",
            "true",
            "--pooling-type",
            "LAST",
            "--enable-generative-audio-transcription",
        ]
    )
    engine_args = EngineArgs.from_cli_args(args)

    assert engine_args.enable_generate_and_pooling is True
    assert engine_args.pooling_task == "embed"
    assert engine_args.pooling_output_dim == 3840
    assert engine_args.pooling_normalize is True
    assert engine_args.pooling_type == "LAST"
    assert engine_args.enable_generative_audio_transcription is True


def test_hybrid_runner_advertises_generate_embed_and_transcription():
    runner = GPUModelRunner.__new__(GPUModelRunner)
    runner.model_config = SimpleNamespace(
        runner_type="generate",
        enable_generate_and_pooling=True,
        enable_generative_audio_transcription=True,
    )
    runner.enable_generation = True
    runner.enable_pooling = True
    runner.model_state = SimpleNamespace(
        get_supported_generation_tasks=lambda: ["generate"]
    )
    runner.model = object()

    assert runner.get_supported_tasks() == ("generate", "transcription", "embed")


def test_hybrid_pooling_rejects_truncated_dimensions():
    model_config = SimpleNamespace(
        enable_generate_and_pooling=True,
        pooling_output_dim=3840,
        pooler_config=SimpleNamespace(dimensions=None, use_activation=None),
        is_matryoshka=False,
        served_model_name="google/gemma-4-12B-it-qat-w4a16-ct",
    )
    params = PoolingParams(task="embed", dimensions=1024)

    with pytest.raises(ValueError, match="does not support Matryoshka"):
        params.verify(model_config)


def test_hybrid_pooling_accepts_full_output_dimension_without_truncation():
    model_config = SimpleNamespace(
        enable_generate_and_pooling=True,
        pooling_output_dim=3840,
        pooler_config=SimpleNamespace(dimensions=None, use_activation=None),
        is_matryoshka=False,
        served_model_name="google/gemma-4-12B-it-qat-w4a16-ct",
    )
    params = PoolingParams(task="embed", dimensions=3840)

    params.verify(model_config)

    assert params.dimensions is None
    assert params.use_activation is True
