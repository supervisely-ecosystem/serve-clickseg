"""Baseline vs head reproducer for the direct-mask Smart Tool contract.

Runs the same Smart Tool request against the *baseline* production module
(``src/main.py`` at ``VERIFY_BASE_SHA``, taken from git or from the verbatim
fixture next to this file) and against the current ``src/main.py``:

1. a first Smart Tool request that carries the contract ``mask`` and no
   ``figure_id``: the baseline never hands the mask to the predictor and
   instead downloads the annotation, the head passes the exactly placed and
   cropped mask and performs no download;
2. a first request that carries both ``mask`` and ``figure_id``: the baseline
   ignores the mask and feeds the downloaded figure geometry to the predictor,
   the head prefers the mask and performs no download.

Exits non-zero when any of those observations no longer holds, so a revert of
the fix (or a regression of the baseline expectation) fails the command.

Usage: ``python tests/unit/baseline_regression.py``
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app_harness as H  # noqa: E402

FALLBACK_BASE_SHA = "b802638de28f40ad2b6a07697d72c520e1e0a969"
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "baseline_src_main.py.txt"

IMAGE_ID = 77
IMAGE_SIZE = (20, 30)  # height, width
CROP = H.crop(2, 1, 25, 15)
LEGACY_FIGURE_ID = 101

# tight 3x4 init mask placed at x=6, y=3 in full image coordinates
MASK_PATTERN = np.zeros((3, 4), bool)
MASK_PATTERN[0, 0] = True
MASK_PATTERN[1, 1:3] = True
MASK_PATTERN[2, 3] = True
MASK_ORIGIN = [6, 3]
# same pattern, shifted: what the legacy figure_id download would return
LEGACY_ORIGIN = [12, 8]

failures = []
report = []


def observe(title, text):
    report.append("{}: {}".format(title, text))
    print("{}: {}".format(title, text), flush=True)


def require(condition, message):
    if not condition:
        failures.append(message)
        print("FAIL: " + message, flush=True)


def baseline_source() -> Path:
    """Baseline ``src/main.py``: git object when available, verbatim fixture otherwise."""
    fixture_bytes = FIXTURE.read_bytes()
    base_sha = os.environ.get("VERIFY_BASE_SHA") or FALLBACK_BASE_SHA
    try:
        from_git = subprocess.check_output(
            ["git", "show", "{}:src/main.py".format(base_sha)],
            cwd=str(H.REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        from_git = None
    if from_git is None:
        observe(
            "Baseline source",
            "git object {}:src/main.py unavailable, using fixture {} (sha256 {})".format(
                base_sha, FIXTURE.name, hashlib.sha256(fixture_bytes).hexdigest()[:16]
            ),
        )
    else:
        observe(
            "Baseline source",
            "{}:src/main.py from git, sha256 {}, identical to fixture: {}".format(
                base_sha,
                hashlib.sha256(from_git).hexdigest()[:16],
                from_git == fixture_bytes,
            ),
        )
        require(
            from_git == fixture_bytes,
            "preserved baseline fixture differs from {}:src/main.py".format(base_sha),
        )
    path = Path(os.environ["SLY_APP_DATA_DIR"]) / "baseline_src_main.py"
    path.write_bytes(fixture_bytes if from_git is None else from_git)
    return path


def legacy_annotation():
    return {
        "objects": [
            {
                "id": LEGACY_FIGURE_ID,
                "classTitle": "legacy",
                "geometryType": "bitmap",
                "bitmap": {
                    "origin": LEGACY_ORIGIN,
                    "data": H.encode_mask(MASK_PATTERN),
                },
            }
        ]
    }


def expected_crop_mask(origin):
    """Init mask the predictor must receive: placed on the image, then cropped."""
    full = np.zeros(IMAGE_SIZE, np.uint8)
    x, y = origin
    full[y : y + MASK_PATTERN.shape[0], x : x + MASK_PATTERN.shape[1]] = MASK_PATTERN * 255
    return full[CROP[0]["y"] : CROP[1]["y"] + 1, CROP[0]["x"] : CROP[1]["x"] + 1]


def context(figure_id=None, with_mask=True):
    ctx = {
        "crop": CROP,
        "positive": [{"x": 7, "y": 5}],
        "negative": [],
        "image_id": IMAGE_ID,
        "init_figure": True,
        "local_figure_id": "local-1",
        "request_uid": "uid-1",
    }
    if figure_id is not None:
        ctx["figure_id"] = figure_id
    if with_mask:
        ctx["mask"] = {"origin": MASK_ORIGIN, "data": H.encode_mask(MASK_PATTERN)}
    return ctx


def run_request(app, ctx):
    """Returns (status_code, payload_or_error_text)."""
    try:
        response = app.post("/smart_segmentation", ctx)
    except Exception as exc:  # the baseline route raises out of the handler
        return None, "{}: {}".format(type(exc).__name__, exc)
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, response.text


def check_mask_only(app, label):
    app.api.annotation_json = None  # any annotation download is a contract violation
    status, payload = run_request(app, context(figure_id=None, with_mask=True))
    observe(
        "{} / mask without figure_id".format(label),
        "result={} {} predictor_calls={} api_calls={}".format(
            status, payload, len(app.predict_calls), app.api.download_calls
        ),
    )
    return status, payload


def check_mask_and_figure_id(app, label):
    app.api.annotation_json = legacy_annotation()
    status, payload = run_request(app, context(figure_id=LEGACY_FIGURE_ID, with_mask=True))
    init_mask = app.predict_calls[-1].init_mask if app.predict_calls else None
    observe(
        "{} / mask and figure_id".format(label),
        "result={} {} predictor_init_mask_pixels={} api_calls={}".format(
            status,
            payload,
            sorted(map(tuple, np.argwhere(init_mask == 255).tolist()))
            if init_mask is not None
            else None,
            app.api.download_calls,
        ),
    )
    return init_mask


def main():
    base_path = baseline_source()

    baseline = H.OfflineSmartToolApp(source_path=base_path, name="clickseg_app_baseline")
    baseline.put_image(IMAGE_ID, H.solid_image(*IMAGE_SIZE))
    status, payload = check_mask_only(baseline, "baseline")
    require(
        not baseline.predict_calls,
        "baseline unexpectedly passed a direct mask to the predictor",
    )
    require(
        baseline.api.annotation_calls == [IMAGE_ID],
        "baseline was expected to download the annotation by figure_id, calls: {}".format(
            baseline.api.download_calls
        ),
    )
    require(
        status is None or status >= 400,
        "baseline was expected to fail the mask-only request, got {} {}".format(status, payload),
    )
    baseline_downloaded = check_mask_and_figure_id(baseline, "baseline")
    require(
        baseline_downloaded is not None
        and np.array_equal(baseline_downloaded, expected_crop_mask(LEGACY_ORIGIN)),
        "baseline was expected to feed the downloaded figure geometry to the predictor",
    )
    require(
        not np.array_equal(baseline_downloaded, expected_crop_mask(MASK_ORIGIN)),
        "baseline unexpectedly honoured the supplied mask",
    )

    head = H.OfflineSmartToolApp(name="clickseg_app_head")
    head.put_image(IMAGE_ID, H.solid_image(*IMAGE_SIZE))
    status, payload = check_mask_only(head, "head")
    require(status == 200, "head must accept a direct mask without figure_id, got {}".format(status))
    require(
        len(head.predict_calls) == 1
        and np.array_equal(head.predict_calls[-1].init_mask, expected_crop_mask(MASK_ORIGIN)),
        "head must pass the exactly placed and cropped mask to the predictor",
    )
    require(
        head.api.download_calls
        == {
            "annotation.download_json": [],
            "video.figure.get_info_by_id": [],
            "image.get_info_by_id": [],
        },
        "head must not download any geometry, calls: {}".format(head.api.download_calls),
    )
    head_mask = check_mask_and_figure_id(head, "head")
    require(
        head_mask is not None and np.array_equal(head_mask, expected_crop_mask(MASK_ORIGIN)),
        "head must prefer the supplied mask over figure_id",
    )
    require(
        head.api.annotation_calls == [],
        "head must not download the annotation when a mask is supplied",
    )

    if failures:
        print("\nBASELINE REGRESSION FAILED ({} problems)".format(len(failures)), flush=True)
        return 1
    print("\nBASELINE REGRESSION OK: baseline drops the direct mask, head honours it", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
