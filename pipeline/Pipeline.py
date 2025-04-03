import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Optional, Callable, Tuple, Type

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_pipeline')


class Step(ABC):
    """Base abstract class for all pipeline steps"""
    
    def __init__(self, name: str = None, **config):
        self.name = name or self.__class__.__name__
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        self.logger = logging.getLogger(f'data_pipeline.{self.name}')
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Process the dataframe"""
        pass
    
    def __str__(self):
        return f"{self.name} ({self.__class__.__name__})"


class Pipeline:
    """Main pipeline class to orchestrate multiple steps"""
    
    def __init__(self, steps: List[Step] = None, name: str = "DefaultPipeline"):
        self.steps = steps or []
        self.name = name
        self.logger = logging.getLogger(f'data_pipeline.{name}')
        self.results = {}
        self.step_registry = {}
    
    def register_step(self, step_class: Type[Step], name: str = None) -> None:
        """Register a step class in the registry"""
        name = name or step_class.__name__
        self.step_registry[name] = step_class
        return self
        
    def add_step(self, step: Union[Step, str], **config) -> 'Pipeline':
        """
        Add a step to the pipeline
        
        Args:
            step: Either a Step instance or a string name of a registered step class
            config: Configuration to pass to the step constructor if step is a string
        """
        if isinstance(step, str):
            if step not in self.step_registry:
                raise ValueError(f"Step '{step}' not found in registry. Available steps: {list(self.step_registry.keys())}")
            # Create step instance from registered class
            step_class = self.step_registry[step]
            step = step_class(**config)
            
        self.steps.append(step)
        return self
    
    def run(self, data: pd.DataFrame, step_configs: Dict[str, Dict] = None) -> pd.DataFrame:
        """Run the entire pipeline with the provided data"""
        result = data.copy()
        step_configs = step_configs or {}
        self.results = {"input": {"shape": data.shape}}
        
        self.logger.info(f"Starting pipeline '{self.name}' with {len(self.steps)} steps")
        
        for i, step in enumerate(self.steps):
            step_name = step.name
            step_num = i + 1
            
            try:
                self.logger.info(f"Step {step_num}/{len(self.steps)}: Running {step_name}")
                
                # Get step-specific configs and merge with kwargs
                step_specific_config = step_configs.get(step_name, {})
                
                # Track metrics before transformation
                before_shape = result.shape
                before_nulls = result.isnull().sum().sum()
                
                # Run the step
                result = step.process(result, **step_specific_config)
                
                # Track metrics after transformation
                after_shape = result.shape
                after_nulls = result.isnull().sum().sum()
                
                # Store results for reporting
                self.results[step_name] = {
                    "step": step_num,
                    "before_shape": before_shape,
                    "after_shape": after_shape,  
                    "columns_added": after_shape[1] - before_shape[1],
                    "nulls_before": before_nulls,
                    "nulls_after": after_nulls,
                    "nulls_change": before_nulls - after_nulls
                }
                
                self.logger.info(
                    f"Step {step_num}: {step_name} completed - "
                    f"Shape changed from {before_shape} to {after_shape}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in step {step_num} '{step_name}': {type(e).__name__} - {str(e)}")
                self.results[step_name] = {
                    "step": step_num, 
                    "error": f"{type(e).__name__}: {str(e)}"
                }
                raise
        
        self.logger.info(f"Pipeline '{self.name}' completed successfully")
        return result
    
    def get_report(self) -> pd.DataFrame:
        """Get a DataFrame with the pipeline results"""
        report_data = []
        
        for step_name, metrics in self.results.items():
            if step_name != "input":  # Skip input metrics
                row = {"step_name": step_name, **metrics}
                report_data.append(row)
        
        return pd.DataFrame(report_data)


class Strategy(ABC):
    """Base strategy class for different algorithm implementations"""
    
    def __init__(self, name: str = None, **config):
        self.name = name or self.__class__.__name__
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        self.logger = logging.getLogger(f'data_pipeline.{self.name}')
    
    @abstractmethod
    def execute(self, data: pd.DataFrame, **kwargs) -> Any:
        """Execute the strategy"""
        pass


class StrategyRegistry:
    """Registry for strategies that can be used across different components"""
    
    def __init__(self):
        self.strategies = {}
        
    def register(self, strategy_class: Type[Strategy], name: str = None) -> None:
        """Register a strategy class"""
        name = name or strategy_class.__name__
        self.strategies[name] = strategy_class
        
    def get(self, name: str, **config) -> Strategy:
        """Get a strategy instance by name"""
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found in registry")
        return self.strategies[name](**config)
        
    def list_strategies(self) -> List[str]:
        """List all registered strategies"""
        return list(self.strategies.keys())


class ConfigManager:
    """Helper class to manage pipeline configurations"""
    
    def __init__(self, base_config: Dict[str, Any] = None):
        self.base_config = base_config or {}
    
    def create_step_config(self, step_name: str, **kwargs) -> Dict[str, Any]:
        """Create configuration for a specific step"""
        # Start with base config for this step if it exists
        config = self.base_config.get(step_name, {}).copy()
        # Update with provided kwargs
        config.update(kwargs)
        return config
    
    def create_pipeline_config(self, **step_configs) -> Dict[str, Dict[str, Any]]:
        """Create a full pipeline configuration dictionary"""
        pipeline_config = {}
        
        # Add each step config
        for step_name, step_config in step_configs.items():
            pipeline_config[step_name] = self.create_step_config(step_name, **step_config)
        
        return pipeline_config
        
    def load_from_json(self, json_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        import json
        with open(json_path, 'r') as f:
            config = json.load(f)
        return config
        
    def save_to_json(self, config: Dict[str, Any], json_path: str) -> None:
        """Save configuration to JSON file"""
        import json
        with open(json_path, 'w') as f:
            json.dump(config, f, indent=2)


# Global strategy registry
strategy_registry = StrategyRegistry()