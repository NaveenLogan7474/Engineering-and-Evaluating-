from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self, model_name=None) -> None:
        """Initialize the base model.
        
        Args:
            model_name: Optional name for the model
        """
        self.model_name = model_name or self.__class__.__name__


    @abstractmethod
    def train(self, data) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        
        Args:
            data: Data object containing training data
            
        Returns:
            self for method chaining
        """
        return self

    @abstractmethod
    def predict(self, X_test) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Predictions as numpy array
        """
        pass

    @abstractmethod
    def print_results(self, data) -> None:
        """
        Print evaluation results.
        
        Args:
            data: Data object containing test data and labels
        """
        pass

    def data_transform(self) -> None:
        """
        Transform data if needed.
        """
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
