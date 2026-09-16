"""Production Smart Tool routes: request -> init mask -> predictor -> response.

The app module is imported exactly as in production (see ``app_harness``); only
weights, the GUI/service startup, the image cache, ``sly.Api`` and the ClickSEG
predictor are replaced by recording doubles. No model is loaded and no GPU is
used, so these tests verify mask/transport behaviour only.
"""

import unittest

import numpy as np

import app_harness as H
import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional

IMAGE_ID = 77
VIDEO_ID, FRAME_INDEX = 5, 2
IMAGE_H, IMAGE_W = 20, 30
CROP = H.crop(2, 1, 25, 15)  # inclusive crop: 15 rows x 24 cols
CROP_H, CROP_W = 15, 24
LEGACY_FIGURE_ID = 101


def pattern():
    data = np.zeros((3, 4), bool)
    data[0, 0] = True
    data[1, 1:3] = True
    data[2, 3] = True
    return data


def mask_payload(x, y, data=None):
    return {"origin": [x, y], "data": H.encode_mask(pattern() if data is None else data)}


def expected_init_mask(x, y, data=None):
    """Full-frame placement of the tight mask, then the production crop."""
    data = pattern() if data is None else data
    full = np.zeros((IMAGE_H, IMAGE_W), np.uint8)
    for row, col in np.argwhere(data):
        frame_row, frame_col = row + y, col + x
        if 0 <= frame_row < IMAGE_H and 0 <= frame_col < IMAGE_W:
            full[frame_row, frame_col] = 255
    return full[CROP[0]["y"] : CROP[1]["y"] + 1, CROP[0]["x"] : CROP[1]["x"] + 1]


def legacy_annotation(x, y):
    return {
        "objects": [
            {
                "id": LEGACY_FIGURE_ID,
                "classTitle": "legacy",
                "geometryType": "bitmap",
                "bitmap": {"origin": [x, y], "data": H.encode_mask(pattern())},
            }
        ]
    }


def image_context(**overrides):
    context = {
        "crop": CROP,
        "positive": [{"x": 7, "y": 5}],
        "negative": [{"x": 20, "y": 12}],
        "image_id": IMAGE_ID,
        "request_uid": "uid-1",
    }
    context.update(overrides)
    return context


def video_context(**overrides):
    context = {
        "crop": CROP,
        "positive": [{"x": 7, "y": 5}],
        "negative": [],
        "video": {"video_id": VIDEO_ID, "frame_index": FRAME_INDEX},
        "request_uid": "uid-1",
    }
    context.update(overrides)
    return context


class SmartToolRouteTest(unittest.TestCase):
    def setUp(self):
        self.app = H.OfflineSmartToolApp()
        self.app.put_image(IMAGE_ID, H.solid_image(IMAGE_H, IMAGE_W))
        self.app.put_frame(VIDEO_ID, FRAME_INDEX, H.solid_image(IMAGE_H, IMAGE_W))

    def assertNoGeometryDownloads(self):
        self.assertEqual(
            self.app.api.download_calls,
            {
                "annotation.download_json": [],
                "video.figure.get_info_by_id": [],
                "image.get_info_by_id": [],
            },
        )

    # -- direct mask -----------------------------------------------------
    def test_direct_mask_without_figure_id(self):
        response = self.app.post(
            "/smart_segmentation",
            image_context(init_figure=True, local_figure_id="local-1", mask=mask_payload(6, 3)),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(self.app.predict_calls), 1)
        call = self.app.predict_calls[0]
        np.testing.assert_array_equal(call.init_mask, expected_init_mask(6, 3))
        self.assertEqual(call.crop_shape, (CROP_H, CROP_W))
        # clicks reach the predictor in crop coordinates, unchanged behaviour
        self.assertEqual(call.clicks, [(5, 4, True), (18, 11, False)])
        self.assertNoGeometryDownloads()

    def test_direct_mask_wins_over_supplied_figure_id(self):
        self.app.api.annotation_json = H.AnnotationDownloadForbidden("must not download")
        response = self.app.post(
            "/smart_segmentation",
            image_context(
                init_figure=True,
                figure_id=LEGACY_FIGURE_ID,
                local_figure_id="local-1",
                mask=mask_payload(6, 3),
            ),
        )
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(self.app.last_init_mask, expected_init_mask(6, 3))
        self.assertNoGeometryDownloads()

    def test_direct_mask_is_clipped_to_the_image(self):
        response = self.app.post(
            "/smart_segmentation",
            image_context(init_figure=True, local_figure_id="local-1", mask=mask_payload(-1, -1)),
        )
        self.assertEqual(response.status_code, 200)
        init_mask = self.app.last_init_mask
        np.testing.assert_array_equal(init_mask, expected_init_mask(-1, -1))
        # origin (-1, -1): one mask pixel is clipped by the image bounds and two
        # more fall outside the crop, the single remaining pixel keeps its place
        self.assertEqual(
            np.argwhere(init_mask == 255).tolist(), [[1 - CROP[0]["y"], 2 - CROP[0]["x"]]]
        )
        self.assertNoGeometryDownloads()

    def test_continuation_request_reuses_the_cached_mask(self):
        first = image_context(
            init_figure=True, local_figure_id="local-1", mask=mask_payload(6, 3)
        )
        self.assertEqual(self.app.post("/smart_segmentation", first).status_code, 200)
        # a later click request carries neither mask nor init_figure
        second = image_context(
            local_figure_id="local-1", positive=[{"x": 8, "y": 6}], negative=[]
        )
        self.assertEqual(self.app.post("/smart_segmentation", second).status_code, 200)
        self.assertEqual(len(self.app.predict_calls), 2)
        np.testing.assert_array_equal(
            self.app.predict_calls[1].init_mask, expected_init_mask(6, 3)
        )
        self.assertNoGeometryDownloads()

    def test_continuation_uses_legacy_figure_id_when_local_id_is_absent(self):
        first = image_context(
            init_figure=True, figure_id=LEGACY_FIGURE_ID, mask=mask_payload(6, 3)
        )
        self.assertEqual(self.app.post("/smart_segmentation", first).status_code, 200)
        second = image_context(figure_id=LEGACY_FIGURE_ID)
        self.assertEqual(self.app.post("/smart_segmentation", second).status_code, 200)
        np.testing.assert_array_equal(
            self.app.predict_calls[1].init_mask, expected_init_mask(6, 3)
        )
        self.assertNoGeometryDownloads()

    def test_malformed_mask_is_rejected_without_predicting(self):
        self.app.api.annotation_json = legacy_annotation(12, 8)
        response = self.app.post(
            "/smart_segmentation",
            image_context(
                init_figure=True,
                figure_id=LEGACY_FIGURE_ID,
                mask={"origin": [6, 3], "data": "not-a-bitmap"},
            ),
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertEqual(payload["success"], False)
        self.assertIsNone(payload["origin"])
        self.assertIsNone(payload["bitmap"])
        self.assertIsInstance(payload["error"], str)
        self.assertTrue(payload["error"])
        self.assertEqual(self.app.predict_calls, [])
        self.assertNoGeometryDownloads()

    def test_mask_outside_of_the_image_is_rejected_and_not_cached(self):
        response = self.app.post(
            "/smart_segmentation",
            image_context(
                init_figure=True,
                local_figure_id="local-1",
                mask=mask_payload(IMAGE_W + 4, 3),
            ),
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["success"], False)
        self.assertEqual(self.app.predict_calls, [])
        # the rejected mask must not be served to the next request of the session
        continuation = image_context(local_figure_id="local-1")
        self.assertEqual(self.app.post("/smart_segmentation", continuation).status_code, 200)
        self.assertIsNone(self.app.last_init_mask)

    def test_direct_mask_without_any_figure_identity(self):
        first = image_context(init_figure=True, mask=mask_payload(6, 3))
        self.assertEqual(self.app.post("/smart_segmentation", first).status_code, 200)
        np.testing.assert_array_equal(self.app.last_init_mask, expected_init_mask(6, 3))
        # nothing to cache it under: a later request without mask has no init mask
        self.assertEqual(self.app.post("/smart_segmentation", image_context()).status_code, 200)
        self.assertIsNone(self.app.last_init_mask)
        self.assertNoGeometryDownloads()

    # -- video -----------------------------------------------------------
    def test_video_frame_direct_mask(self):
        response = self.app.post(
            "/smart_segmentation",
            video_context(init_figure=True, local_figure_id="local-9", mask=mask_payload(6, 3)),
        )
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(self.app.last_init_mask, expected_init_mask(6, 3))
        self.assertIn(("download_frame", VIDEO_ID, FRAME_INDEX), self.app.cache.calls)
        self.assertNoGeometryDownloads()

    def test_video_legacy_figure_id_fallback(self):
        self.app.api.video_figure = H.VideoFigure(
            {
                "bitmap": {"origin": [12, 8], "data": H.encode_mask(pattern())},
                "geometryType": "bitmap",
            }
        )
        response = self.app.post(
            "/smart_segmentation",
            video_context(init_figure=True, figure_id=LEGACY_FIGURE_ID),
        )
        self.assertEqual(response.status_code, 200)
        np.testing.assert_array_equal(self.app.last_init_mask, expected_init_mask(12, 8))
        self.assertEqual(self.app.api.video_figure_calls, [LEGACY_FIGURE_ID])

    # -- legacy image path ----------------------------------------------
    def test_legacy_image_figure_id_fallback_is_unchanged(self):
        self.app.api.annotation_json = legacy_annotation(12, 8)
        response = self.app.post(
            "/smart_segmentation",
            image_context(init_figure=True, figure_id=LEGACY_FIGURE_ID),
        )
        self.assertEqual(response.status_code, 200)
        init_mask = self.app.last_init_mask
        np.testing.assert_array_equal(init_mask, expected_init_mask(12, 8))
        # identical to the pre-existing SDK download/placement/crop pipeline
        legacy = functional.download_init_mask(self.app.api, LEGACY_FIGURE_ID, IMAGE_ID)
        np.testing.assert_array_equal(
            init_mask,
            functional.crop_image(CROP, functional.bitmap_to_mask(legacy, IMAGE_H, IMAGE_W)),
        )
        self.assertEqual(self.app.api.annotation_calls, [IMAGE_ID, IMAGE_ID])

    def test_legacy_continuation_is_served_from_cache_without_a_second_download(self):
        self.app.api.annotation_json = legacy_annotation(12, 8)
        init = image_context(init_figure=True, figure_id=LEGACY_FIGURE_ID)
        self.assertEqual(self.app.post("/smart_segmentation", init).status_code, 200)
        # later click request of the same legacy session: figure_id only, no init_figure
        continuation = image_context(figure_id=LEGACY_FIGURE_ID, positive=[{"x": 8, "y": 6}])
        self.assertEqual(self.app.post("/smart_segmentation", continuation).status_code, 200)
        np.testing.assert_array_equal(
            self.app.predict_calls[1].init_mask, expected_init_mask(12, 8)
        )
        self.assertEqual(self.app.api.annotation_calls, [IMAGE_ID])

    def test_request_without_any_initial_object_still_predicts(self):
        response = self.app.post("/smart_segmentation", image_context())
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(self.app.last_init_mask)
        self.assertNoGeometryDownloads()

    # -- response transport ---------------------------------------------
    def test_bitmap_response_transport_is_unchanged(self):
        pred = np.zeros((CROP_H, CROP_W), bool)
        pred[4:7, 5:9] = True
        self.app.pred_mask = pred
        response = self.app.post(
            "/smart_segmentation",
            image_context(init_figure=True, local_figure_id="local-1", mask=mask_payload(6, 3)),
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["success"], True)
        self.assertIsNone(payload["error"])
        # origin is the crop offset plus the tight bbox of the predicted mask
        self.assertEqual(payload["origin"], {"x": CROP[0]["x"] + 5, "y": CROP[0]["y"] + 4})
        np.testing.assert_array_equal(
            sly.Bitmap.base64_2_data(payload["bitmap"]), np.ones((3, 4), bool)
        )

    def test_empty_prediction_response_is_unchanged(self):
        self.app.pred_mask = np.zeros((CROP_H, CROP_W), bool)
        response = self.app.post(
            "/smart_segmentation",
            image_context(init_figure=True, local_figure_id="local-1", mask=mask_payload(6, 3)),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"origin": None, "bitmap": None, "success": True, "error": None},
        )

    def test_request_without_clicks_is_unchanged(self):
        response = self.app.post(
            "/smart_segmentation",
            image_context(positive=[], negative=[], init_figure=True, mask=mask_payload(6, 3)),
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"origin": None, "bitmap": None, "success": True, "error": None},
        )
        self.assertEqual(self.app.predict_calls, [])


class SmartToolBatchRouteTest(unittest.TestCase):
    def setUp(self):
        self.app = H.OfflineSmartToolApp()
        self.app.put_image(IMAGE_ID, H.solid_image(IMAGE_H, IMAGE_W))

    def test_batch_mask_legacy_and_malformed_items(self):
        self.app.api.annotation_json = legacy_annotation(12, 8)
        states = [
            image_context(init_figure=True, local_figure_id="local-1", mask=mask_payload(6, 3)),
            image_context(init_figure=True, figure_id=LEGACY_FIGURE_ID),
            image_context(init_figure=True, mask={"origin": "nope", "data": "x"}),
        ]
        response = self.app.post("/smart_segmentation_batch", {"states": states})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload), 3)
        self.assertEqual([item["success"] for item in payload], [True, True, False])
        self.assertEqual([item["origin"] for item in payload], [None, None, None])
        self.assertTrue(payload[2]["error"])
        self.assertEqual(len(self.app.predict_calls), 2)
        np.testing.assert_array_equal(
            self.app.predict_calls[0].init_mask, expected_init_mask(6, 3)
        )
        np.testing.assert_array_equal(
            self.app.predict_calls[1].init_mask, expected_init_mask(12, 8)
        )
        # the mask item needs no geometry download; legacy item downloads once
        self.assertEqual(self.app.api.annotation_calls, [IMAGE_ID])
        self.assertEqual(self.app.api.image_info_calls, [])


if __name__ == "__main__":
    unittest.main()
