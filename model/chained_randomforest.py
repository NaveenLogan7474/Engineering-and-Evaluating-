import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import random

seed = 0
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class ChainedRandomForest(BaseModel):
    """Chained Random Forest classifier for multi-label classification."""
    
    def __init__(self, model_name="ChainedRandomForest", embeddings=None) -> None:
        """Initialize the Chained Random Forest model.
        
        Args:
            model_name: Name of the model
            embeddings: Feature embeddings
        """
        super().__init__(model_name)
        self.embeddings = embeddings
        
        # Initialize three separate models for different label combinations
        self.model_y2 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.model_y2_y3 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        self.model_y2_y3_y4 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        
        self.y2_predictions = None
        self.y2_y3_predictions = None
        self.y2_y3_y4_predictions = None
        self.y3_predictions = None
        self.y4_predictions = None
        
        self.data_transform()

    def train(self, data) -> BaseModel:
        """Train the model.
        
        Args:
            data: ChainedData object containing training data
            
        Returns:
            self for method chaining
        """
        # Train the model for Type 2
        self.model_y2 = self.model_y2.fit(data.get_X_train(), data.get_y2_train())
        
        # Train the model for Type 2 + Type 3
        self.model_y2_y3 = self.model_y2_y3.fit(data.get_X_train(), data.get_y2_y3_train())
        
        # Train the model for Type 2 + Type 3 + Type 4
        self.model_y2_y3_y4 = self.model_y2_y3_y4.fit(data.get_X_train(), data.get_y2_y3_y4_train())
        
        return self

    def predict(self, X_test) -> np.ndarray:
        """Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Primary prediction array (Type 2)
        """
        # Predict Type 2
        self.y2_predictions = self.model_y2.predict(X_test)
        
        # Predict Type 2 + Type 3
        self.y2_y3_predictions = self.model_y2_y3.predict(X_test)
        
        # Predict Type 2 + Type 3 + Type 4
        self.y2_y3_y4_predictions = self.model_y2_y3_y4.predict(X_test)
        
        # Split the combined predictions to extract individual types
        self.y3_predictions = np.array([pred.split('+')[1] for pred in self.y2_y3_predictions])
        
        temp = np.array([pred.split('+') for pred in self.y2_y3_y4_predictions])
        self.y4_predictions = temp[:, 2]
        
        return self.y2_predictions

    def print_results(self, data):
        """Print model evaluation results."""
        # Type 2 Results
        print("\n--- Type 2 Classification Results ---")
        print(classification_report(data.get_y2_test(), self.y2_predictions, zero_division=0))
        
        # Combined Type 2+3 Results
        print("\n--- Type 2+3 Combined Classification Results ---")
        print(classification_report(data.get_y2_y3_test(), self.y2_y3_predictions, zero_division=0))
        
        # Combined Type 2+3+4 Results
        print("\n--- Type 2+3+4 Combined Classification Results ---")
        print(classification_report(data.get_y2_y3_y4_test(), self.y2_y3_y4_predictions, zero_division=0))
        
        # Extract Type 3 results from combined predictions
        print("\n--- Type 3 Individual Classification Results (Extracted from Combined) ---")
        print(classification_report(data.get_y3_test(), self.y3_predictions, zero_division=0))
        
        # Extract Type 4 results from combined predictions
        print("\n--- Type 4 Individual Classification Results (Extracted from Combined) ---")
        print(classification_report(data.get_y4_test(), self.y4_predictions, zero_division=0))
        
        return True

    def data_transform(self) -> None:
        """Transform data if needed."""
        pass 