import json

import cv2
import numpy as np
import pytest

from stego_ai.cli import _synthetic_cover, project_description
from stego_ai.models import extract_features
from stego_ai.pipeline import build_arg_parser
from stego_ai.stego_algorithms import payload_capacity_bytes, process_image


@pytest.fixture()
def cover_path(tmp_path):
    path = tmp_path / "cover.png"
    assert cv2.imwrite(str(path), _synthetic_cover(size=512, seed=7))
    return path


@pytest.mark.parametrize("algorithm", ["lsb", "dct", "dwt"])
def test_text_round_trip_through_png(cover_path, tmp_path, algorithm):
    message = "round trip: şifre değil, steganografi"
    output = tmp_path / f"{algorithm}.png"

    process_image(
        str(cover_path),
        algorithm,
        action="embed",
        secret_message=message,
        seed=123,
        out_path=str(output),
    )
    recovered = process_image(str(output), algorithm, action="extract", seed=123)

    assert recovered == message


def test_capacity_accounts_for_transform_overhead():
    assert payload_capacity_bytes(512, 512, "lsb") > payload_capacity_bytes(512, 512, "dwt")
    assert payload_capacity_bytes(512, 512, "dwt") > payload_capacity_bytes(512, 512, "dct")
    assert payload_capacity_bytes(16, 16, "dct") == 0


def test_feature_contracts(cover_path):
    assert extract_features(str(cover_path), method="dct").shape == (64,)
    assert extract_features(str(cover_path), method="residual_hist").shape == (18,)
    assert extract_features(str(cover_path), method="residual_cooc").shape == (162,)


def test_default_pipeline_avoids_optional_boosters():
    args = build_arg_parser().parse_args(["--cover-dir", "covers", "--work-dir", "runs"])
    assert args.models == ["rf", "logreg", "svm"]


def test_description_states_security_boundary():
    description = project_description()
    encoded = json.dumps(description).lower()
    assert "not encryption" in encoded
    assert "integrity" in encoded

