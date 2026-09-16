"""Smart Tool initialization mask (direct-mask contract).

The labeling tool sends the initial object of a Smart Tool session as a tight
binary bitmap in image / video frame coordinates::

    context = {
        "init_figure": True,
        "mask": {"origin": [x, y], "data": "<encoded bitmap>"},
        ...
    }

``origin`` is the top-left ``x, y`` position of the tight mask in the full image
or frame and ``data`` uses the same encoded representation as the Supervisely
Bitmap JSON geometry. The mask is preferred over ``figure_id`` whenever both are
supplied, so no annotation / figure download is needed to start a session.

``figure_id`` initialization (``init_figure`` without ``mask``) is kept as a
deprecated compatibility path for legacy callers.
"""

from typing import Optional, Tuple

import numpy as np
import supervisely as sly
from supervisely.nn.inference.interactive_segmentation import functional

MASK_FIELD = "mask"


class MaskDecodeError(ValueError):
    """The supplied init mask is malformed, empty or outside of the frame."""


def get_cache_key(context: dict):
    """Continuation cache identity: ``local_figure_id``, legacy ``figure_id`` otherwise."""
    key = context.get("local_figure_id")
    if key is None:
        key = context.get("figure_id")
    return key


def _origin_to_xy(origin) -> Tuple[int, int]:
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        raise MaskDecodeError(
            "init mask 'origin' must be a [x, y] pair, got: {!r}".format(origin)
        )
    xy = []
    for value in origin:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MaskDecodeError(
                "init mask 'origin' must contain integers, got: {!r}".format(origin)
            )
        if isinstance(value, float) and not float(value).is_integer():
            raise MaskDecodeError(
                "init mask 'origin' must contain integers, got: {!r}".format(origin)
            )
        xy.append(int(value))
    return xy[0], xy[1]


def decode_context_mask(mask) -> sly.Bitmap:
    """Validate and decode the contract mask into a positioned :class:`sly.Bitmap`.

    :raises MaskDecodeError: mask payload is not a decodable non-empty bitmap.
    """
    if not isinstance(mask, dict):
        raise MaskDecodeError(
            "init mask must be an object with 'origin' and 'data', got: {}".format(
                type(mask).__name__
            )
        )
    x, y = _origin_to_xy(mask.get("origin"))
    data = mask.get("data")
    if not isinstance(data, str) or data == "":
        raise MaskDecodeError("init mask 'data' must be a non-empty encoded string")
    try:
        array = sly.Bitmap.base64_2_data(data)
    except Exception as exc:  # malformed base64 / zlib / image payload
        raise MaskDecodeError("init mask 'data' can not be decoded: {}".format(exc))
    if not isinstance(array, np.ndarray) or array.ndim != 2 or array.size == 0:
        raise MaskDecodeError("init mask 'data' must decode to a non-empty 2D bitmap")
    try:
        return sly.Bitmap(
            data=array.astype(bool),
            origin=sly.PointLocation(row=y, col=x),
            extra_validation=False,
        )
    except ValueError as exc:
        raise MaskDecodeError("init mask is empty: {}".format(exc))


def place_mask_on_frame(bitmap: sly.Bitmap, height: int, width: int) -> np.ndarray:
    """Place a tight bitmap on the full image / frame, clipping it to the bounds.

    Returns a ``uint8`` ``(height, width)`` mask with ``0`` / ``255`` values, i.e.
    exactly the predictor input produced by ``functional.bitmap_to_mask`` for
    fully inside masks.

    :raises MaskDecodeError: the mask has no pixels inside the frame.
    """
    if height <= 0 or width <= 0:
        raise MaskDecodeError(
            "can not place init mask on a {}x{} frame".format(width, height)
        )
    data = bitmap.data
    x_from, y_from = bitmap.origin.col, bitmap.origin.row
    dst_left, dst_top = max(0, x_from), max(0, y_from)
    dst_right = min(width, x_from + data.shape[1])
    dst_bottom = min(height, y_from + data.shape[0])
    if dst_right <= dst_left or dst_bottom <= dst_top:
        raise MaskDecodeError("init mask is fully outside of the image / frame")
    src = data[
        dst_top - y_from : dst_bottom - y_from, dst_left - x_from : dst_right - x_from
    ]
    if not src.any():
        raise MaskDecodeError("init mask has no pixels inside the image / frame")
    mask = np.zeros((height, width), np.uint8)
    mask[dst_top:dst_bottom, dst_left:dst_right] = src * 255
    return mask


def _download_legacy_init_mask(api: sly.Api, context: dict) -> sly.Bitmap:
    figure_id = context.get("figure_id")
    sly.logger.warn(
        "Deprecated Smart Tool initialization: no 'mask' in the request, "
        "falling back to downloading the figure geometry by figure_id.",
        extra={"figure_id": figure_id},
    )
    if context.get("image_id") is not None:
        return functional.download_init_mask(api, figure_id, context["image_id"])
    figure = api.video.figure.get_info_by_id(figure_id)
    return sly.Bitmap.from_json(figure.geometry)


def resolve_init_mask(
    context: dict,
    api: sly.Api,
    init_mask_cache,
    img_height: int,
    img_width: int,
) -> Optional[np.ndarray]:
    """Resolve the full-frame init mask for a Smart Tool request.

    Preference order:

    1. contract ``mask`` from the request (no annotation / figure download),
    2. deprecated ``init_figure`` + ``figure_id`` download (images and videos),
    3. mask cached by ``local_figure_id`` / legacy ``figure_id`` (continuation).

    :return: ``uint8`` full image / frame mask with ``0`` / ``255`` values or
        ``None`` when the request carries no initial object.
    :raises MaskDecodeError: the supplied ``mask`` is unusable; there is no
        silent fallback to ``figure_id`` in that case.
    """
    cache_key = get_cache_key(context)
    mask_json = context.get(MASK_FIELD)
    has_legacy_source = (
        context.get("image_id") is not None or context.get("video") is not None
    )
    if mask_json is not None:
        bitmap = decode_context_mask(mask_json)
        sly.logger.debug(
            "Smart Tool init mask taken from the request payload.",
            extra={"cache_key": cache_key, "origin": mask_json.get("origin")},
        )
        if cache_key is not None:
            init_mask_cache[cache_key] = bitmap
    elif context.get("init_figure") is True and has_legacy_source:
        if context.get("figure_id") is None:
            sly.logger.warn(
                "Smart Tool initialization without 'mask' and without 'figure_id': "
                "prediction will start without an initial mask."
            )
            return None
        bitmap = _download_legacy_init_mask(api, context)
        if cache_key is not None:
            init_mask_cache[cache_key] = bitmap
    elif cache_key is not None and init_mask_cache.get(cache_key) is not None:
        bitmap = init_mask_cache.get(cache_key)
        sly.logger.debug(
            "Smart Tool init mask restored from cache.", extra={"cache_key": cache_key}
        )
    else:
        return None
    return place_mask_on_frame(bitmap, img_height, img_width)
