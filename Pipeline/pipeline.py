from typing import Dict, List, Optional, Type

import pandas as pd
from Pipeline.DataAnalysis import DataProfiler, CorrelationHeatmap
from base import PipelineComponent, PipelineStep, StepConfig
from Pipeline.transformers import CategoricalEncoder, DataImputer, DataScaler, MissingValueRemover


class Pipeline(PipelineComponent):
    """Base class for all pipelines

    Args:
        PipelineComponent (ABC): Base class for all pipeline components.
    """
    def __init__(self, name: Optional[str] = None, steps: Optional[List[StepConfig]] = None):
        """Initialize the pipeline with a list of steps."""
        super().__init__(name or self.__class__.__name__)
        self.steps = self._create_steps_(steps) if steps else []

    def _create_steps_(self, step_config: List[Dict] = None) -> List[StepConfig]:
        """Create steps based on the pipeline configuration."""
        steps = []
        for config in step_config:
            print(config)
            for step_name, step_args in config.items():
                steps.append(
                    StepConfig(
                        step=step_name,
                        name = step_args.pop('name', None),
                        params=step_args)
                    )
        return steps
    def _get_step_class(self, class_name: str) -> Type[PipelineStep]:
        """Resolve step class by name - only called during execution."""
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
        """Run all steps on the data"""
        result = data.copy()
        print(self.steps)
        steps_to_execute = (self._create_steps(step_configs)
                if step_configs is not None
                else self.steps)
        if not steps_to_execute:
            self.logger.warning("No steps configured in pipeline")
            return result
        self.logger.info(f"Starting pipeline execution with {len(steps_to_execute)} steps, steps: {steps_to_execute}")
        for step in steps_to_execute:
            try:
                self.logger.info(f"Running step: {step.name}")
                config = step.params
                self.logger.info(f"Step {step.name} params: {config['strategy'] if 'strategy' in config else config}")
                step_instance = self._get_step_class(step.step)(name=step.name)
                self.logger.info(f"Step {step_instance} instance created")
                step_instance.set_config(config=config)
                result = step_instance.fit_transform(result, config)
                self.logger.info(f"Step {step.name} completed successfully")
                self.logger.info(f"Result shape after step {step.name}: {result.shape if isinstance(result, pd.DataFrame) else result}")
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                raise
        return result