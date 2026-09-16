"""Direct-mask contract: decoding, placement/clipping, precedence and caching."""

import unittest

import numpy as np

import app_harness as H  # sets up an offline env and puts the repo root on sys.path
import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional

from src import init_mask as init_mask_lib

IMAGE_H, IMAGE_W = 20, 30


def pattern():
    """Tight 3x4 mask with a hole and a disconnected corner pixel."""
    data = np.zeros((3, 4), bool)
    data[0, 0] = True
    data[1, 1:3] = True
    data[2, 3] = True
    return data


def bitmap_at(x, y, data=None):
    data = pattern() if data is None else data
    return sly.Bitmap(
        data=data, origin=sly.PointLocation(row=y, col=x), extra_validation=False
    )


def mask_payload(x, y, data=None):
    return {"origin": [x, y], "data": H.encode_mask(pattern() if data is None else data)}


def placed(x, y, data=None, height=IMAGE_H, width=IMAGE_W):
    """Reference full-frame mask, built independently of the implementation."""
    data = pattern() if data is None else data
    expected = np.zeros((height, width), np.uint8)
    for row, col in np.argwhere(data):
        frame_row, frame_col = row + y, col + x
        if 0 <= frame_row < height and 0 <= frame_col < width:
            expected[frame_row, frame_col] = 255
    return expected


class DecodeTest(unittest.TestCase):
    def test_decodes_positioned_bitmap(self):
        bitmap = init_mask_lib.decode_context_mask(mask_payload(6, 3))
        self.assertIsInstance(bitmap, sly.Bitmap)
        self.assertEqual((bitmap.origin.col, bitmap.origin.row), (6, 3))
        np.testing.assert_array_equal(bitmap.data, pattern())

    def test_accepts_integral_float_origin(self):
        bitmap = init_mask_lib.decode_context_mask({**mask_payload(6, 3), "origin": [6.0, 3.0]})
        self.assertEqual((bitmap.origin.col, bitmap.origin.row), (6, 3))

    def test_malformed_payloads_are_explicit_errors(self):
        empty = np.zeros((3, 4), bool)
        cases = {
            "not an object": "mask",
            "origin missing": {"data": H.encode_mask(pattern())},
            "origin not a pair": {"origin": [1, 2, 3], "data": H.encode_mask(pattern())},
            "origin not numeric": {"origin": ["6", "3"], "data": H.encode_mask(pattern())},
            "origin fractional": {"origin": [6.5, 3], "data": H.encode_mask(pattern())},
            "origin boolean": {"origin": [True, False], "data": H.encode_mask(pattern())},
            "data missing": {"origin": [6, 3]},
            "data empty": {"origin": [6, 3], "data": ""},
            "data not a string": {"origin": [6, 3], "data": 42},
            "data not base64": {"origin": [6, 3], "data": "!!!not base64!!!"},
            "data not a bitmap": {"origin": [6, 3], "data": "aGVsbG8gd29ybGQ="},
            "mask without pixels": {"origin": [6, 3], "data": H.encode_mask(empty)},
        }
        for label, payload in cases.items():
            with self.subTest(label):
                with self.assertRaises(init_mask_lib.MaskDecodeError):
                    init_mask_lib.decode_context_mask(payload)


class PlacementTest(unittest.TestCase):
    def test_exact_placement_at_nonzero_origin(self):
        mask = init_mask_lib.place_mask_on_frame(bitmap_at(6, 3), IMAGE_H, IMAGE_W)
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(mask.shape, (IMAGE_H, IMAGE_W))
        np.testing.assert_array_equal(mask, placed(6, 3))
        self.assertEqual(sorted(np.unique(mask).tolist()), [0, 255])

    def test_matches_legacy_predictor_input_for_inside_masks(self):
        bitmap = bitmap_at(6, 3)
        np.testing.assert_array_equal(
            init_mask_lib.place_mask_on_frame(bitmap, IMAGE_H, IMAGE_W),
            functional.bitmap_to_mask(bitmap, IMAGE_H, IMAGE_W),
        )

    def test_clips_negative_origin(self):
        mask = init_mask_lib.place_mask_on_frame(bitmap_at(-1, -1), IMAGE_H, IMAGE_W)
        np.testing.assert_array_equal(mask, placed(-1, -1))
        # the pixel at pattern (0, 0) falls outside and must be dropped
        self.assertEqual(mask.sum() // 255, 3)

    def test_clips_overflowing_mask(self):
        mask = init_mask_lib.place_mask_on_frame(bitmap_at(IMAGE_W - 2, IMAGE_H - 2), IMAGE_H, IMAGE_W)
        np.testing.assert_array_equal(mask, placed(IMAGE_W - 2, IMAGE_H - 2))
        self.assertEqual(mask.sum() // 255, 2)

    def test_mask_fully_outside_is_an_error(self):
        with self.assertRaises(init_mask_lib.MaskDecodeError):
            init_mask_lib.place_mask_on_frame(bitmap_at(IMAGE_W + 5, 3), IMAGE_H, IMAGE_W)

    def test_mask_without_pixels_inside_is_an_error(self):
        data = np.zeros((3, 4), bool)
        data[0, 0] = True  # only pixel is clipped away at origin (-1, -1)
        with self.assertRaises(init_mask_lib.MaskDecodeError):
            init_mask_lib.place_mask_on_frame(bitmap_at(-1, -1, data), IMAGE_H, IMAGE_W)


class CacheKeyTest(unittest.TestCase):
    def test_prefers_local_figure_id(self):
        self.assertEqual(
            init_mask_lib.get_cache_key({"local_figure_id": "local-1", "figure_id": 7}), "local-1"
        )

    def test_falls_back_to_legacy_figure_id(self):
        self.assertEqual(init_mask_lib.get_cache_key({"figure_id": 7}), 7)

    def test_without_identity(self):
        self.assertIsNone(init_mask_lib.get_cache_key({}))


class ResolveTest(unittest.TestCase):
    def setUp(self):
        self.cache = {}
        self.api = H.StubApi()

    def resolve(self, context):
        return init_mask_lib.resolve_init_mask(context, self.api, self.cache, IMAGE_H, IMAGE_W)

    def legacy_annotation(self, figure_id, x, y):
        return {
            "objects": [
                {
                    "id": figure_id,
                    "classTitle": "legacy",
                    "geometryType": "bitmap",
                    "bitmap": {"origin": [x, y], "data": H.encode_mask(pattern())},
                }
            ]
        }

    def test_mask_wins_over_figure_id_without_any_download(self):
        context = {
            "init_figure": True,
            "image_id": 77,
            "figure_id": 101,
            "local_figure_id": "local-1",
            "mask": mask_payload(6, 3),
        }
        self.api.annotation_json = H.AnnotationDownloadForbidden("download attempted")
        np.testing.assert_array_equal(self.resolve(context), placed(6, 3))
        self.assertEqual(self.api.annotation_calls, [])
        self.assertEqual(self.api.image_info_calls, [])

    def test_continuation_reuses_cached_mask_by_local_figure_id(self):
        first = {
            "init_figure": True,
            "image_id": 77,
            "local_figure_id": "local-1",
            "mask": mask_payload(6, 3),
        }
        self.resolve(first)
        self.assertIsInstance(self.cache["local-1"], sly.Bitmap)
        continuation = {"image_id": 77, "local_figure_id": "local-1"}
        np.testing.assert_array_equal(self.resolve(continuation), placed(6, 3))
        self.assertEqual(self.api.annotation_calls, [])

    def test_continuation_falls_back_to_legacy_figure_id_key(self):
        self.resolve({"init_figure": True, "image_id": 77, "figure_id": 101, "mask": mask_payload(6, 3)})
        self.assertIn(101, self.cache)
        np.testing.assert_array_equal(
            self.resolve({"image_id": 77, "figure_id": 101}), placed(6, 3)
        )
        self.assertEqual(self.api.annotation_calls, [])

    def test_unknown_continuation_key_has_no_init_mask(self):
        self.assertIsNone(self.resolve({"image_id": 77, "local_figure_id": "other"}))

    def test_legacy_image_figure_id_download_fallback(self):
        self.api.annotation_json = self.legacy_annotation(101, 12, 8)
        context = {"init_figure": True, "image_id": 77, "figure_id": 101}
        np.testing.assert_array_equal(self.resolve(context), placed(12, 8))
        self.assertEqual(self.api.annotation_calls, [77])
        self.assertIn(101, self.cache)

    def test_legacy_video_figure_download_fallback(self):
        self.api.video_figure = H.VideoFigure(
            {
                "bitmap": {"origin": [12, 8], "data": H.encode_mask(pattern())},
                "geometryType": "bitmap",
            }
        )
        context = {"init_figure": True, "video": {"video_id": 5, "frame_index": 2}, "figure_id": 101}
        np.testing.assert_array_equal(self.resolve(context), placed(12, 8))
        self.assertEqual(self.api.video_figure_calls, [101])
        self.assertEqual(self.api.annotation_calls, [])

    def test_malformed_mask_never_falls_back_to_download(self):
        self.api.annotation_json = self.legacy_annotation(101, 12, 8)
        context = {
            "init_figure": True,
            "image_id": 77,
            "figure_id": 101,
            "mask": {"origin": [6, 3], "data": "not-a-bitmap"},
        }
        with self.assertRaises(init_mask_lib.MaskDecodeError):
            self.resolve(context)
        self.assertEqual(self.api.annotation_calls, [])
        self.assertEqual(self.cache, {})

    def test_init_figure_without_mask_and_figure_id(self):
        self.assertIsNone(self.resolve({"init_figure": True, "image_id": 77}))
        self.assertEqual(self.api.annotation_calls, [])

    def test_request_without_any_initial_object(self):
        self.assertIsNone(self.resolve({"image_id": 77}))
        self.assertEqual(self.api.annotation_calls, [])


if __name__ == "__main__":
    unittest.main()
