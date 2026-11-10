"""CPU policy checks for the opt-in SM120 FA4 prefill tile."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from cutlass import BFloat16

from sglang.kernels.ops.attention.fa4_sm120.flash_fwd import (
    FlashAttentionForwardSm120,
)
from sglang.kernels.ops.attention.fa4_sm120.runtime import (
    Sm120ForwardPolicy,
    _get_prefill_tile_config,
    _get_prefill_tile_override,
)
from sglang.srt.environ import envs

SWITCH = "SGLANG_FA4_SM120_PREFILL_TILE"


class SM120PrefillTileTests(unittest.TestCase):
    def setUp(self):
        _get_prefill_tile_override.cache_clear()
        _get_prefill_tile_config.cache_clear()
        self.addCleanup(_get_prefill_tile_override.cache_clear)
        self.addCleanup(_get_prefill_tile_config.cache_clear)

    @contextmanager
    def configured(self, value):
        # Each context simulates a separately configured server startup.
        with patch.dict(os.environ, {SWITCH: value}):
            _get_prefill_tile_override.cache_clear()
            try:
                yield
            finally:
                _get_prefill_tile_override.cache_clear()

    def select(self, **overrides):
        args = dict(
            head_dim=256,
            head_dim_v=256,
            tile_mn=None,
            has_bias=False,
            total_q_rows=8192 * 16,
            num_sms=188,
            num_batch=1,
            seqlen_q=8192,
            seqlen_k=24576,
            num_head_kv=2,
            qhead_per_kvhead=8,
            is_causal=True,
            is_local=False,
            window_size_left=None,
            window_size_right=0,
            pack_gqa=True,
            paged_kv=True,
        )
        args.update(overrides)
        return Sm120ForwardPolicy.select_config(**args)

    def test_unset_defaults_to_native_policy(self):
        with patch.dict(os.environ):
            os.environ.pop(SWITCH, None)
            self.assertEqual(envs.SGLANG_FA4_SM120_PREFILL_TILE.get(), "auto")
            actual = self.select()
            with self.configured("auto"):
                self.assertEqual(actual, self.select())
            self.assertEqual((actual.tile_m, actual.tile_n), (64, 64))

    def test_opt_in_uses_supported_native_four_warp_kernel(self):
        with self.configured("64x48"):
            config = self.select()
        self.assertEqual(config.compile_key, (64, 48, 1, 128))
        self.assertEqual(FlashAttentionForwardSm120.num_dma_threads, 32)
        self.assertTrue(
            FlashAttentionForwardSm120.can_implement(
                BFloat16, 256, 256, 64, 48, 1, 128, True, paged_kv=True
            )
        )

    def test_explicit_tile_takes_precedence(self):
        with self.configured("64x48"):
            actual = self.select(tile_mn=(32, 64))
        self.assertEqual((actual.tile_m, actual.tile_n), (32, 64))

    def test_decode_and_unqualified_workloads_keep_native_policy(self):
        cases = (
            dict(seqlen_q=1),
            dict(seqlen_q=None),
            dict(paged_kv=False),
            dict(is_causal=False, window_size_right=None),
            dict(is_local=True, window_size_left=4096),
        )
        for args in cases:
            with self.subTest(args=args):
                with self.configured("auto"):
                    native = self.select(**args)
                with self.configured("64x48"):
                    self.assertEqual(native, self.select(**args))

    def test_supported_tiles_are_not_limited_to_64x48(self):
        for value, expected in (("64x32", (64, 32)), ("32x64", (32, 64))):
            with self.subTest(value=value), self.configured(value):
                actual = self.select()
                self.assertEqual((actual.tile_m, actual.tile_n), expected)

    def test_supported_head_dimensions_are_not_hardcoded(self):
        with self.configured("64x48"):
            actual = self.select(head_dim=128, head_dim_v=128)
        self.assertEqual((actual.tile_m, actual.tile_n), (64, 48))

    def test_invalid_format_and_resource_exhaustion_are_rejected(self):
        for value in ("bad", "1", "64x", "64x48x32", "0x48", "-1x48"):
            with self.subTest(value=value), self.configured(value):
                with self.assertRaisesRegex(ValueError, "positive"):
                    self.select()
        for value in ("64x128", "64x47", "1x16"):
            with self.subTest(value=value), self.configured(value):
                with self.assertRaisesRegex(ValueError, "kernel constraints"):
                    self.select()

    def test_explicit_tile_overrides_environment_even_if_invalid(self):
        with self.configured("bad"):
            actual = self.select(tile_mn=(32, 64))
        self.assertEqual((actual.tile_m, actual.tile_n), (32, 64))

    def test_relative_bias_policy_is_preserved(self):
        with self.configured("64x48"):
            actual = self.select(has_bias=True, head_dim=128, head_dim_v=128)
            self.assertEqual((actual.tile_m, actual.tile_n), (64, 128))
            with self.assertRaisesRegex(ValueError, "up to 128"):
                self.select(has_bias=True)

    def test_startup_setting_is_stable_and_restart_changes_compile_key(self):
        selector = FlashAttentionForwardSm120.get_fwd_tile_size
        with self.configured("auto"):
            native = self.select()
            with patch.dict(os.environ, {SWITCH: "64x48"}):
                self.assertEqual(native, self.select())
        with self.configured("64x48"):
            tuned = self.select()
        self.assertNotEqual(native.compile_key, tuned.compile_key)
        with self.configured("auto"):
            self.assertEqual(native, self.select())
        self.assertIs(selector, FlashAttentionForwardSm120.get_fwd_tile_size)

    def test_layers_reuse_config_and_validate_each_geometry_once(self):
        field = envs.SGLANG_FA4_SM120_PREFILL_TILE
        with (
            self.configured("64x48"),
            patch.object(field, "get", wraps=field.get) as read_env,
            patch.object(
                FlashAttentionForwardSm120,
                "can_implement",
                wraps=FlashAttentionForwardSm120.can_implement,
            ) as validate,
        ):
            first = self.select()
            for _ in range(40):
                repeated = self.select(
                    num_batch=8, seqlen_q=53, seqlen_k=34000, total_q_rows=1792 * 16
                )
                self.assertIs(first, repeated)
            self.assertEqual(first.compile_key, (64, 48, 1, 128))
            self.assertEqual(read_env.call_count, 1)
            self.assertEqual(validate.call_count, 1)
            other = self.select(head_dim=128, head_dim_v=128)
            self.assertIsNot(first, other)
            self.assertEqual((other.tile_m, other.tile_n), (64, 48))
            self.assertIs(other, self.select(head_dim=128, head_dim_v=128))
            self.assertEqual(read_env.call_count, 1)
            self.assertEqual(validate.call_count, 2)

    def test_auto_reads_environment_once_without_override_validation(self):
        field = envs.SGLANG_FA4_SM120_PREFILL_TILE
        with (
            self.configured("auto"),
            patch.object(field, "get", wraps=field.get) as read_env,
            patch.object(
                FlashAttentionForwardSm120,
                "can_implement",
                wraps=FlashAttentionForwardSm120.can_implement,
            ) as validate,
        ):
            first = self.select()
            for _ in range(40):
                self.assertEqual(first, self.select())
            self.assertEqual((first.tile_m, first.tile_n), (64, 64))
            self.assertEqual(read_env.call_count, 1)
            validate.assert_not_called()

    def test_decode_and_explicit_tiles_do_not_resolve_override(self):
        field = envs.SGLANG_FA4_SM120_PREFILL_TILE
        with (
            self.configured("bad"),
            patch.object(field, "get", wraps=field.get) as read_env,
        ):
            decode = self.select(seqlen_q=1)
            self.assertNotEqual((decode.tile_m, decode.tile_n), (64, 48))
            explicit = self.select(tile_mn=(32, 64))
            self.assertEqual((explicit.tile_m, explicit.tile_n), (32, 64))
            bias = self.select(has_bias=True, head_dim=128, head_dim_v=128)
            self.assertEqual((bias.tile_m, bias.tile_n), (64, 128))
            read_env.assert_not_called()


if __name__ == "__main__":
    unittest.main()
