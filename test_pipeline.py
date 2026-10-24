import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "material": "Li10GeP2S12_LGPS",
    "SBI": 11.58,
    "AFC": 0.025,
    "LLSD": 2.55,
    "LLB": 5.71,
    "LASD": 2.26,
}

MOCK_VALID_CIF = """
data_LiCoO2
_cell_length_a 2.815
_cell_length_b 2.815
_cell_length_c 14.05
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_symmetry_space_group_name_H-M 'R -3 m'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li 0.0 0.0 0.5
Co 0.0 0.0 0.0
O 0.0 0.0 0.24
"""

MOCK_INVALID_NO_LI_CIF = """
data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
Na 0.0 0.0 0.0
Cl 0.5 0.5 0.5
"""


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_predict_features():
    res = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_class"] in (0, 1)
    assert 0.0 <= body["superionic_probability"] <= 1.0
    assert body["model_name"] == "Random Forest"


def test_missing_and_unknown_fields_rejected():
    # Missing field
    incomplete = dict(SAMPLE_PAYLOAD)
    del incomplete["LASD"]
    assert client.post("/predict", json=incomplete).status_code == 422

    # Extra unknown field
    surplus = dict(SAMPLE_PAYLOAD)
    surplus["temperature"] = 300.0
    assert client.post("/predict", json=surplus).status_code == 422


def test_predict_cif_success():
    files = {"file": ("LiCoO2.cif", MOCK_VALID_CIF, "chemical/x-cif")}
    res = client.post("/predict-cif", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["material"] == "LiCoO2"
    assert "LLSD" in body["descriptors"]
    assert 0.0 <= body["superionic_probability"] <= 1.0


def test_predict_cif_rejected_if_no_lithium():
    files = {"file": ("NaCl.cif", MOCK_INVALID_NO_LI_CIF, "chemical/x-cif")}
    res = client.post("/predict-cif", files=files)
    assert res.status_code == 422
    assert "no Lithium" in res.json()["detail"]