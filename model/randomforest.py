import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


class RandomForest(BaseModel):
    """Random Forest classifier model."""
    
    def __init__(self, model_name="RandomForest", embeddings=None) -> None:
        """Initialize the Random Forest model.
        
        Args:
            model_name: Name of the model
            embeddings: Feature embeddings
        """
        super().__init__(model_name)
        self.embeddings = embeddings
        self.classifier = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.predictions = None
        self.data_transform()

    def train(self, data) -> BaseModel:
        """Train the model.
        
        Args:
            data: Data object containing training data
            
        Returns:
            self for method chaining
        """
        X_train = data.get_X_train()
        y_train = data.get_type_y_train()
        self.classifier = self.classifier.fit(X_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        """Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predictions as numpy array
        """
        self.predictions = self.classifier.predict(X_test)
        return self.predictions

    def print_results(self, data) -> None:
        """Print evaluation results.
        
        Args:
            data: Data object containing test data and labels
        """
        print(f"\nResults for {self.model_name}:")
        print(classification_report(data.get_type_y_test(), self.predictions))

    def data_transform(self) -> None:
        """Transform data if needed."""
        pass

