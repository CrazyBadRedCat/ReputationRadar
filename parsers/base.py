from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from pandas import DataFrame

@dataclass
class BaseParser(ABC):    
    @abstractmethod
    def get_request(self) -> DataFrame:
        raise NotImplementedError