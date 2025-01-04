from pydantic import BaseModel
from typing import List, Union, Dict, Literal, Optional


class StatusResponse(BaseModel):
    status: str


class FitUploadCSVResponse(BaseModel):
    operation_id: str


class BaselineHyperparameters(BaseModel):
    penalty: List[Literal["l1", "l2"]]
    dual: List[bool]
    C: List[float]
    class_weight: List[Dict[int, int]]


FitHyperparameters = Union[BaselineHyperparameters]


class FitRequest(BaseModel):
    id: str
    operation_id: str
    hyperparameters: Optional[FitHyperparameters] = None


class FitResponse(BaseModel):
    message: str


class SetRequest(BaseModel):
    id: str


class SetResponse(BaseModel):
    message: str


class ModelDetails(BaseModel):
    id: str
    status: str


class ModelGetResponse(BaseModel):
    models: List[ModelDetails]


class PredictRequest(BaseModel):
    X: List[str]


class PredictResponse(BaseModel):
    texts: List[str]
    predictions: List[int]
    probabilities: List[List[float]]


class ValidationError(BaseModel):
    loc: List[Union[str, int]]
    msg: str
    type: str


class HTTPValidationError(BaseModel):
    detail: List[ValidationError]
