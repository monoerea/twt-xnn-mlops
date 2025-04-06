from abc import ABC
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Type, TypeVar

import pandas as pd


Step = TypeVar('Step', bound='PipelineStep')

@dataclass
class StepConfig:
    step: Type[Step]
    name: Optional[str]
    params: Dict[str, Any]

class PipelineComponent(ABC):
    """Base class for all pipeline components"""
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f'pipeline.{self.name}')
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

class PipelineStep(PipelineComponent):
    """Base class for all pipeline steps"""
    def __init__(self, name: Optional[str] = None):
        super().__init__(name or self.__class__.__name__)
        self.config = None

    def set_config(self, config: dict) -> 'PipelineStep':
        """Set the configuration for this step"""
        self.config = config

    def fit(self, data: pd.DataFrame, config: dict = None) -> None:
        """Fit the step to the data"""
        pass
    def transform(self, data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Transform the data"""
        pass
    def fit_transform(self, data: pd.DataFrame, config: dict = None) -> pd.DataFrame:
        """Fit and transform the data"""
        self.fit(data, config)
        return self.transform(data, config)


        return result

class Cleaner(PipelineStep):
    """Base class for all cleaners in the pipeline."""

    def __init__(self, name: str = None):
        super().__init__(name or self.__class__.__name__)