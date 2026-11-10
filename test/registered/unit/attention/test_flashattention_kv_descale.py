"""Regression tests for runtime KV dtype and checkpoint descales."""

import unittest
from types import SimpleNamespace

import torch


class KVDescaleTests(unittest.TestCase):
    def prepare(self, dtype, ver=4):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionBackend,
        )

        backend = SimpleNamespace(
            kv_cache_dtype_str="bf16" if dtype == torch.bfloat16 else "fp8_e4m3",
            kv_cache_dtype=dtype,
            kv_cache_is_mxfp8=False,
        )
        layer = SimpleNamespace(
            head_dim=256, k_scale=torch.tensor(0.5), v_scale=torch.tensor(0.25)
        )
        return FlashAttentionBackend.prepare_paged_mha_query(
            backend,
            torch.ones(1, 256, dtype=torch.bfloat16),
            None,
            None,
            layer,
            logical_batch_size=2,
            kv_head_num=2,
            is_prefill=False,
            ver=ver,
        )

    def test_bf16_ignores_checkpoint_fp8_scales(self):
        for ver in (3, 4):
            q, _, _, ks, vs = self.prepare(torch.bfloat16, ver)
            self.assertEqual(q.dtype, torch.bfloat16)
            self.assertIsNone(ks)
            self.assertIsNone(vs)

    def test_fp8_keeps_checkpoint_scales(self):
        q, _, _, ks, vs = self.prepare(torch.float8_e4m3fn)
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(ks.shape), (2, 2))
        self.assertTrue(torch.all(ks == 0.5))
        self.assertTrue(torch.all(vs == 0.25))


if __name__ == "__main__":
    unittest.main()
