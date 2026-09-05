# ═══════════════════════════════════════════════════════════════════
# app.py — FastAPI Backend
# ═══════════════════════════════════════════════════════════════════
# Car Price Predictor ANN ka production API
# Routes:
# GET  /              → landing page
# GET  /train         → TrainingPipeline().run_pipeline()
# GET  /predict       → CSV upload form
# POST /predict       → batch CSV → predicted prices table
# GET  /predict/manual → manual input form
# POST /predict/manual → single car → predicted price JSON
###==============================================================

import os
import sys
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from carprice.pipeline.training_pipeline import TrainingPipeline
from carprice.utils.DL_utils.model.estimator import CarPriceModel
from carprice.utils.main_utils.utils import load_object
from carprice.cloud.hf_syncer import pull_model_from_huggingface
from carprice.exception.exception import CarPriceException
from carprice.logging.logger import logging

load_dotenv()

# ── LIFESPAN EVENT ────────────────────────────────────────────────
# IMP: on_event deprecated → lifespan use karo (modern approach)
# startup pe final_model/ check karo
# nahi hai → HuggingFace se download karo
# Railway/Render pe final_model/ nahi hoti (gitignore mein)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ───────────────────────────────────────────────────
    if not os.path.exists("final_model/model.keras"):
        logging.info("final_model/ missing → downloading from HuggingFace")
        pull_model_from_huggingface()
        logging.info("Models ready ✅")
    else:
        logging.info("final_model/ exists → skipping download")
    yield
    # ── SHUTDOWN ──────────────────────────────────────────────────
    logging.info("App shutting down")

# ── FASTAPI APP ───────────────────────────────────────────────────
app = FastAPI(
    title="Car Price Predictor — ANN",
    description="Used car price prediction using ANN regression pipeline",
    version="0.0.1",
    lifespan=lifespan
)

# ── CORS ──────────────────────────────────────────────────────────
# allow_origins=["*"] → sab domains allowed
# production mein specific domain do
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── TEMPLATES ─────────────────────────────────────────────────────
templates = Jinja2Templates(directory="./templates")


# ══════════════════════════════════════════════════════════════════
# PYDANTIC INPUT MODEL — Manual Prediction
# ══════════════════════════════════════════════════════════════════
class CarInput(BaseModel):
    """
    Manual prediction ke liye input schema.
    Pydantic automatic type validation karta hai.
    Missing field → 422 Unprocessable Entity

    Feature values come from actual car data:
    vehicle_age     : int   → car kitni saal purani hai
    km_driven       : float → log1p(km) — DataIngestion mein transform kiya tha
    seller_type     : str   → "Dealer" / "Individual"
    fuel_type       : str   → "Petrol" / "Diesel" / "CNG"
    transmission_type: str  → "Manual" / "Automatic"
    mileage         : float → kmpl
    engine          : float → cc
    max_power       : float → bhp
    seats           : float → number of seats
    brand           : str   → car brand e.g. "Maruti"
    """
    vehicle_age:       int
    km_driven:         float
    seller_type:       str
    fuel_type:         str
    transmission_type: str
    mileage:           float
    engine:            float
    max_power:         float
    seats:             float
    brand:             str


# ══════════════════════════════════════════════════════════════════
# ROUTE 1: / → Landing Page
# ══════════════════════════════════════════════════════════════════
@app.get("/", tags=["Home"])
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# ══════════════════════════════════════════════════════════════════
# ROUTE 2: /train → Training Pipeline
# ══════════════════════════════════════════════════════════════════
@app.get("/train", tags=["Training"])
async def train_route():
    """
    Poori training pipeline trigger karta hai.
    CSV → Ingestion → Validation → Transformation → ANN Train → HF push
    """
    try:
        logging.info("Train route called")
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        return Response("Training completed successfully")
    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# ROUTE 3: GET /predict → CSV Upload Form
# ══════════════════════════════════════════════════════════════════
@app.get("/predict", tags=["Prediction"])
async def predict_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict.html"
    )


# ══════════════════════════════════════════════════════════════════
# ROUTE 4: POST /predict → Batch CSV Prediction
# ══════════════════════════════════════════════════════════════════
@app.post("/predict", tags=["Prediction"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    """
    CSV upload → batch prediction → HTML table

    CSV mein yeh columns hone chahiye:
    vehicle_age, km_driven, seller_type, fuel_type,
    transmission_type, mileage, engine, max_power, seats, brand
    """
    try:
        logging.info(f"Predict route called — file: {file.filename}")

        df = pd.read_csv(file.file)
        logging.info(f"Uploaded CSV shape: {df.shape}")

        # model + preprocessor load karo
        preprocessor = load_object("final_model/preprocessor.pkl")
        model_path   = "final_model/model.keras"

        car_model = CarPriceModel(
            preprocessor=preprocessor,
            model_path=model_path
        )

        # predict → actual rupees
        y_pred = car_model.predict(df)
        # IMP: CarPriceModel.predict() → expm1() apply karta hai
        # output already actual rupees mein hai

        df["predicted_price (₹)"] = y_pred.astype(int)

        os.makedirs("prediction_output", exist_ok=True)
        df.to_csv("prediction_output/output.csv", index=False)

        table_html = df.to_html(classes="table table-striped")

        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={"table": table_html}
        )

    except Exception as e:
        raise CarPriceException(e, sys)


# ══════════════════════════════════════════════════════════════════
# ROUTE 5: GET /predict/manual → Manual Input Form
# ══════════════════════════════════════════════════════════════════
@app.get("/predict/manual", tags=["Prediction"])
async def predict_manual_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="predict_manual.html"
    )


# ══════════════════════════════════════════════════════════════════
# ROUTE 6: POST /predict/manual → Single Car Prediction
# ══════════════════════════════════════════════════════════════════
@app.post("/predict/manual", tags=["Prediction"])
async def predict_manual(data: CarInput):
    """
    Manual input → single car price prediction

    IMP: km_driven user raw km dega → log1p apply karo
    DataIngestion mein km_driven log1p kiya tha
    yahan bhi karna hoga — same transformation

    Returns:
        JSON: {
            predicted_price: int (actual rupees),
            formatted: "₹4,50,000"
        }
    """
    try:
        logging.info("Manual predict route called")

        # pydantic → dict → DataFrame
        input_dict = data.model_dump()

        # IMP: km_driven → user raw km deta hai
        # DataIngestion mein log1p apply tha → yahan bhi apply karo
        input_dict["km_driven"] = np.log1p(input_dict["km_driven"])

        input_df = pd.DataFrame([input_dict])
        logging.info(f"Input shape: {input_df.shape}")

        preprocessor = load_object("final_model/preprocessor.pkl")
        model_path   = "final_model/model.keras"

        car_model = CarPriceModel(
            preprocessor=preprocessor,
            model_path=model_path
        )

        prediction = car_model.predict(input_df)[0]
        # [0] → single prediction nikalo
        # CarPriceModel → expm1() apply karta hai → actual rupees

        predicted_price = int(prediction)
        formatted_price = f"₹{predicted_price:,}"

        logging.info(f"Prediction: {formatted_price}")

        return {
            "predicted_price": predicted_price,
            "formatted":       formatted_price
        }

    except Exception as e:
        raise CarPriceException(e, sys)


# ── ENTRY POINT ───────────────────────────────────────────────────
if __name__ == "__main__":
    app_run(app, host="0.0.0.0", port=8000)