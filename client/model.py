from dataclasses import dataclass, field
from enum import Enum
from typing import Dict


class PenaltyType(Enum):
    L1 = "l1"
    L2 = "l2"


@dataclass
class Model:
    name: str

    def get_hyperparameters(self):
        return None

    def get_id(self):
        return None


@dataclass
class BaselineModel(Model):
    name: str
    penalty: PenaltyType = PenaltyType.L2
    dual: bool = False
    C: float = 1.0
    class_weight: Dict[float, float] = field(default_factory=lambda: {1: 1})

    def get_hyperparameters(self) -> dict:
        return {'penalty': [self.penalty.value],
                'dual': [self.dual],
                'C': [self.C],
                'class_weight': [self.class_weight]}

    def get_id(self):
        return "baseline"


class ModelType(Enum):
    BASELINE = BaselineModel('Baseline')


model_types: Dict[str, Model] = {}
for model in ModelType:
    model_types[model.name.lower()] = model.value
