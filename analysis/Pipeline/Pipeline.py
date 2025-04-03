import pandas as pd
import logging
from typing import List, Dict, Any, Callable

class PipelineStep:
    """Base class for all pipeline steps"""
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f'pipeline.{self.name}')
    
    def process(self, data: pd.DataFrame, config: Dict = None) -> pd.DataFrame:
        """Process the data"""
        return data

class Pipeline:
    """Simplified pipeline to process data through multiple steps"""
    
    def __init__(self, name: str = "SimplePipeline"):
        self.name = name
        self.steps: List[PipelineStep] = []
        self.logger = logging.getLogger(f'pipeline.{name}')
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """Add a step to the pipeline"""
        self.steps.append(step)
        return self
    
    def run(self, data: pd.DataFrame, step_configs: Dict[str, Dict] = None) -> pd.DataFrame:
        """Run all steps on the data"""
        result = data.copy()
        step_configs = step_configs or {}
        
        for step in self.steps:
            try:
                self.logger.info(f"Running step: {step.name}")
                config = step_configs.get(step.name, {})
                result = step.process(result, config)
            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                raise
        
        return result