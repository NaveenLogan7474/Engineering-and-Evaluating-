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


class HierarchicalRandomForest(BaseModel):
    """Hierarchical Random Forest classifier for multi-label classification."""
    
    def __init__(self, model_name="HierarchicalRandomForest", embeddings=None) -> None:
        """Initialize the Hierarchical Random Forest model.
        
        Args:
            model_name: Name of the model
            embeddings: Feature embeddings
        """
        super().__init__(model_name)
        self.embeddings = embeddings
        
        # Initialize the main Type 2 model
        self.model_y2 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
        
        # Dictionaries to store Type 3 and Type 4 models
        self.models_y3 = {}  # Key: Type 2 class, Value: RF model for Type 3
        self.models_y4 = {}  # Key: (Type 2 class, Type 3 class), Value: RF model for Type 4
        
        # Store predictions
        self.y2_predictions = None
        self.y3_predictions = {}  # Key: Type 2 class, Value: predictions for Type 3
        self.y4_predictions = {}  # Key: (Type 2 class, Type 3 class), Value: predictions for Type 4
        
        # Store true test values for evaluation
        self.y3_test_values = {}
        self.y4_test_values = {}
        
        self.data_transform()

    def train(self, data) -> BaseModel:
        """Train the model.
        
        Args:
            data: HierarchicalData object containing training data
            
        Returns:
            self for method chaining
        """
        # Train the model for Type 2
        X_train, y2_train = data.get_type2_train_data()
        self.model_y2 = self.model_y2.fit(X_train, y2_train)
        
        # For each Type 2 class, train a model for Type 3
        for type2_class in data.get_type2_classes():
            # Get data for this Type 2 class
            X3_train, X3_test, y3_train, y3_test, _ = data.get_type3_data_for_type2_class(type2_class)
            
            # Skip if not enough data
            if X3_train is None:
                continue
                
            # Create and train Type 3 model for this Type 2 class
            model_y3 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
            model_y3 = model_y3.fit(X3_train, y3_train)
            self.models_y3[type2_class] = model_y3
            self.y3_test_values[type2_class] = (X3_test, y3_test)
            
            # Get unique Type 3 classes for this Type 2 class
            unique_y3_values = np.unique(y3_train)
            
            # For each Type 3 class in this Type 2 class, train a model for Type 4
            for type3_class in unique_y3_values:
                # Get data for this Type 2, Type 3 combination
                X4_train, X4_test, y4_train, y4_test, _ = data.get_type4_data_for_type2_type3_class(type2_class, type3_class)
                
                # Skip if not enough data
                if X4_train is None:
                    continue
                    
                # Create and train Type 4 model for this Type 2, Type 3 combination
                model_y4 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
                model_y4 = model_y4.fit(X4_train, y4_train)
                self.models_y4[(type2_class, type3_class)] = model_y4
                self.y4_test_values[(type2_class, type3_class)] = (X4_test, y4_test)
        
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
        
        # For each Type 2 class, predict Type 3
        for type2_class, model_y3 in self.models_y3.items():
            if type2_class in self.y3_test_values:
                X3_test, _ = self.y3_test_values[type2_class]
                self.y3_predictions[type2_class] = model_y3.predict(X3_test)
        
        # For each Type 2, Type 3 combination, predict Type 4
        for (type2_class, type3_class), model_y4 in self.models_y4.items():
            if (type2_class, type3_class) in self.y4_test_values:
                X4_test, _ = self.y4_test_values[(type2_class, type3_class)]
                self.y4_predictions[(type2_class, type3_class)] = model_y4.predict(X4_test)
        
        return self.y2_predictions

    def print_results(self, data):
        """Print model evaluation results."""
        # Type 2 (top-level) results
        print("\n--- Type 2 Classification Results ---")
        X_test, y2_test = data.get_type2_test_data()
        print(classification_report(y2_test, self.y2_predictions, zero_division=0))
        
        # Type 3 results for each Type 2 class
        print("\n--- Type 3 Classification Results ---")
        for type2_class, predictions in self.y3_predictions.items():
            _, y3_test = self.y3_test_values[type2_class]
            print(f"\nType 2 Class: {type2_class}")
            print(classification_report(y3_test, predictions, zero_division=0))
        
        # Type 4 results for each Type 2 + Type 3 combination
        print("\n--- Type 4 Classification Results ---")
        for (type2_class, type3_class), predictions in self.y4_predictions.items():
            _, y4_test = self.y4_test_values[(type2_class, type3_class)]
            print(f"\nType 2 Class: {type2_class}, Type 3 Class: {type3_class}")
            print(classification_report(y4_test, predictions, zero_division=0))
        
        return True

    def data_transform(self) -> None:
        """Transform data if needed."""
        pass 