import json
import math
import joblib
import pandas as pd
from typing import Literal
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from descriptors import extract_sendek_descriptors

app = FastAPI(
    title="Sendek Lithium Superionic Conductor Screener",
    description="Machine learning prediction API for solid-state battery electrolytes",
    version="1.0.0",
)

# Enable CORS for web and client interfaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load metadata and model pipeline
with open("model_metadata.json", "r") as f:
    METADATA = json.load(f)

FEATURES_ORDER = METADATA["features"]
MODEL = joblib.load("model.joblib")


class MaterialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    material: str
    SBI: float = Field(..., description="Sublattice packing / bonding index")
    AFC: float = Field(..., description="Average framework coordination")
    LLSD: float = Field(..., description="Li-Li shortest distance (Å)")
    LLB: float = Field(..., description="Li-Li shortest bond metric")
    LASD: float = Field(..., description="Li-anion shortest distance (Å)")


class PredictionResponse(BaseModel):
    material: str
    predicted_class: Literal[0, 1]
    superionic_probability: float
    model_name: str
    paper_reference_score: float
    descriptors: dict
    interpretation: str


def compute_paper_score(descriptors: dict) -> float:
    """Computes the 2017/2019 Sendek published logistic regression equation."""
    z = (
            -4.48937
            + 1.05 * descriptors["LLB"]
            + 3.20 * descriptors["SBI"]
            + 0.55 * descriptors["AFC"]
            + 0.40 * descriptors["LASD"]
            + 0.20 * descriptors["LLSD"]
    )
    return round(1.0 / (1.0 + math.exp(-z)), 4)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": METADATA["model_name"],
        "cv_f1": METADATA["cross_validation_f1"]
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_from_features(data: MaterialInput):
    """Predicts superionicity from pre-computed numerical descriptors."""
    desc_dict = {
        "SBI": data.SBI,
        "AFC": data.AFC,
        "LLSD": data.LLSD,
        "LLB": data.LLB,
        "LASD": data.LASD
    }

    X = pd.DataFrame([desc_dict])[FEATURES_ORDER]
    pred = int(MODEL.predict(X)[0])
    prob = float(MODEL.predict_proba(X)[0, 1])
    paper_score = compute_paper_score(desc_dict)

    return PredictionResponse(
        material=data.material,
        predicted_class=pred,
        superionic_probability=round(prob, 4),
        model_name=METADATA["model_name"],
        paper_reference_score=paper_score,
        descriptors=desc_dict,
        interpretation=(
            "Likely superionic lithium conductor (σ ≥ 10⁻⁴ S/cm)" if pred == 1
            else "Non-superionic conductor based on structural descriptors"
        )
    )


@app.post("/predict-cif", response_model=PredictionResponse)
async def predict_from_cif(file: UploadFile = File(...)):
    """Receives an uploaded .cif file, extracts Sendek descriptors, and runs ML inference."""
    if not file.filename.endswith(".cif"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a .cif format file.")

    content = await file.read()
    cif_str = content.decode("utf-8")

    try:
        desc_dict = extract_sendek_descriptors(cif_str)
    except ValueError as err:
        raise HTTPException(status_code=422, detail=str(err))

    material_name = file.filename.replace(".cif", "")
    X = pd.DataFrame([desc_dict])[FEATURES_ORDER]

    pred = int(MODEL.predict(X)[0])
    prob = float(MODEL.predict_proba(X)[0, 1])
    paper_score = compute_paper_score(desc_dict)

    return PredictionResponse(
        material=material_name,
        predicted_class=pred,
        superionic_probability=round(prob, 4),
        model_name=METADATA["model_name"],
        paper_reference_score=paper_score,
        descriptors=desc_dict,
        interpretation=(
            "Likely superionic lithium conductor (σ ≥ 10⁻⁴ S/cm)" if pred == 1
            else "Non-superionic conductor based on structural descriptors"
        )
    )