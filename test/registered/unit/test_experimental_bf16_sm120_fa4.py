import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.attention_registry import (
    experimental_sm120_gdn_backends,
)


class ExperimentalBF16FA4Tests(unittest.TestCase):
    def runner(self, compute=torch.bfloat16, cache=torch.bfloat16):
        return SimpleNamespace(
            dtype=compute,
            kv_cache_dtype=cache,
            prefill_attention_backend_str="fa4",
            decode_attention_backend_str="fa4",
        )

    def test_default_remains_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("fa4", experimental_sm120_gdn_backends(self.runner()))

    def test_opt_in_accepts_matching_bf16(self):
        with patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_SM120_GDN_FA4": "1"}):
            self.assertIn("fa4", experimental_sm120_gdn_backends(self.runner()))

    def test_fp8_cache_is_rejected(self):
        with patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_SM120_GDN_FA4": "1"}):
            with self.assertRaisesRegex(ValueError, "FP8 is unsupported"):
                experimental_sm120_gdn_backends(self.runner(cache=torch.float8_e4m3fn))

    def test_fp16_cache_is_outside_this_experiment(self):
        with patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_SM120_GDN_FA4": "1"}):
            with self.assertRaises(ValueError):
                experimental_sm120_gdn_backends(
                    self.runner(torch.float16, torch.float16)
                )

    def test_mixed_compute_and_cache_are_rejected(self):
        with patch.dict(os.environ, {"SGLANG_EXPERIMENTAL_SM120_GDN_FA4": "1"}):
            with self.assertRaisesRegex(ValueError, "matching compute"):
                experimental_sm120_gdn_backends(self.runner(compute=torch.float16))


if __name__ == "__main__":
    unittest.main()
