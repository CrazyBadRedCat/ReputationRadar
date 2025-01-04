from fastapi import APIRouter, UploadFile, Query, File, Depends
from typing import Annotated
from app.models import ModelsDispatcher
from app.schemas import (
    FitUploadCSVResponse,
    FitRequest,
    FitResponse,
    PredictRequest,
    PredictResponse,
    SetRequest,
    SetResponse,
    ModelGetResponse,
)

router = APIRouter()


def get_models_dispatcher() -> ModelsDispatcher:
    from main import models_dispatcher
    return models_dispatcher


@router.post("/fit/upload_csv", response_model=FitUploadCSVResponse)
async def upload_csv(
    operation_id: Annotated[str, Query(..., description="Operation ID for the dataset upload")],
    file: Annotated[UploadFile, File(..., description="CSV file to upload")],
    models_dispatcher: Annotated[ModelsDispatcher, Depends(get_models_dispatcher)],
):
    await models_dispatcher.store_dataset(operation_id, file)
    return {"operation_id": operation_id}


@router.post("/fit", response_model=FitResponse)
async def fit_model(
    request: FitRequest,
    models_dispatcher: Annotated[ModelsDispatcher, Depends(get_models_dispatcher)],
):
    models_dispatcher.train(request.id, request.operation_id, request.hyperparameters)
    models_dispatcher.wait_for_training(request.id)
    return {"message": f"Model '{request.id}' trained and saved"}


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    models_dispatcher: Annotated[ModelsDispatcher, Depends(get_models_dispatcher)],
):
    texts, predictions, probabilities = models_dispatcher.predict(request.X)

    return PredictResponse(
        texts=texts,
        predictions=predictions,
        probabilities=probabilities
    )


@router.post("/set", response_model=SetResponse)
async def set_model(
    request: SetRequest,
    models_dispatcher: Annotated[ModelsDispatcher, Depends(get_models_dispatcher)],
):
    models_dispatcher.set_current_model(request.id)
    return {"message": f"Model '{request.id}' loaded"}


@router.get("/models", response_model=ModelGetResponse)
async def list_models(
    models_dispatcher: Annotated[ModelsDispatcher, Depends(get_models_dispatcher)],
):
    return {"models": models_dispatcher.list_models()}
