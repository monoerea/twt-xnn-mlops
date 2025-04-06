from typing import Dict, List, Optional, Type

import pandas as pd
from base import PipelineComponent, PipelineStep, StepConfig
from cleaners import MissingValueHandler


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
    def _get_step_class(self, step_name: str) -> Type[PipelineStep]:
        """Resolve step class by name - only called during execution."""
        return {
            "MissingValueHandler": MissingValueHandler,
            # Add other step mappings here
        }.get(step_name)
    def fit_transform(self, data: pd.DataFrame, step_configs: List[Dict] = None) -> pd.DataFrame:
        """Run all steps on the data"""
        result = data.copy()
        print(self.steps)
        steps_to_execute = self.steps
        # (self._create_steps(step_configs)
        #             if step_configs is not None
        #             else self.steps)
        if not steps_to_execute:
            self.logger.warning("No steps configured in pipeline")
            return result
        self.logger.info(f"Starting pipeline execution with {len(steps_to_execute)} steps")
        for step in steps_to_execute:
            try:
                self.logger.info(f"Running step: {step.name}")
                config = step.params
                self.logger.info(f"Step {step.name} params: {config['strategy']}")
                step_instance = self._get_step_class(step.step)(name=step.name)
                step_instance.set_config(config=config)
                result = step_instance.fit_transform(result, config)
                self.logger.info(f"Step {step.name} completed successfully")
                self.logger.info(f"Result shape after step {step.name}: {result.shape}")
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                raise
        return result