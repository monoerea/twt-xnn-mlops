from abc import ABC
from dataclasses import dataclass
from typing import List, Dict, Optional, Type, TypeVar
import pandas as pd
from Pipeline.DataAnalysis import CorrelationHeatmap, DataProfiler
from Pipeline.transformers import CategoricalEncoder, DataImputer, DataScaler, MissingValueRemover
from base import PipelineComponent, PipelineStep, StepConfig

Component = TypeVar('PipelineComponent', bound=[PipelineComponent, PipelineStep])

@dataclass
class ProcessConfig:
    class_type: Type[Component]
    name: Optional[str]
    params: Dict[str, any]

class Pipeline(ABC):
    """"
    Implementation of the pipeline class for data processing.
    This class is responsible for executing a series of steps on the data.
    Can store the steps in a list and execute them sequentially.
    """
    def __init__(self):
        self.processes: List[ProcessConfig] = []
        self.logger = None
        self.results = None
        self.processed = None

    def get_result(self, step_name: str) -> pd.DataFrame:
        """
        Get the result of a specific step in the pipeline.
        """
        if step_name in self.processed:
            return self.processed[step_name]['result']
        else:
            raise ValueError(f"Step {step_name} not found in processed steps.")
    def create_processes(self, config: List[Dict] = None) -> List[StepConfig]:
        return [ProcessConfig(class_type=step.get('class_name'), name=step_args.pop('name', None), params=step_args) for step in config for step_name, step_args in step.items() if self._get_step_class(step_name)]
    def _get_step_class(self, class_name: str) -> Type[PipelineStep]:
        """
        Resolve step class by name - only called during execution.
        """
        classes = {
            "MissingValueRemover": MissingValueRemover,
            "DataImputer": DataImputer,
            "DataScaler": DataScaler,
            "CorrelationHeatmap": CorrelationHeatmap,
            "DataProfiler": DataProfiler,
            'CategoricalEncoder': CategoricalEncoder,
        }
        target_class = classes.get(class_name)
        self.logger.info(f"Step class: { target_class}")
        return target_class

    def fit_transform(self, data: pd.DataFrame, step_configs: List[Dict] = None) -> pd.DataFrame:
        """
        Run all steps on the data.
        Returns the transformed data.
        Stores the results via stroring in the class object.
        """
        self.processes = self.create_processes(step_configs) if step_configs else self.processes
        if not self.processes:
            self.logger.warning("No steps configured in pipeline")
            return result
        for process in self.processes:
            try:
                self.logger.info(f"Running step: {process.name}")
                process_class = self._get_step_class(process.class_type)
                process = process_class(**process.params)
                result = process.fit_transform(data)
                self.results = {'class': process.__class__.__name__, 'result': result}
                self.processed[process.get_name()] = {'name': process.get_name(), 'process': process, 'config':process.get_config() ,'result': result}
            except Exception as e:
                self.logger.error(f"Error in step {process.name}: {e}")