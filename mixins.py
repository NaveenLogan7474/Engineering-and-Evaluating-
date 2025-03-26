import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
import re

class LoggingMixin:
    """Adds logging capabilities to models."""
    
    def __init__(self, *args, **kwargs):
        self.log_entries = []
        super().__init__(*args, **kwargs)
    
    def log(self, message):
        """Record a log message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {message}"
        self.log_entries.append(entry)
        print(entry)
    
    def train(self, data):
        self.log(f"Training started with model: {getattr(self, 'model_name', 'Unknown')}")
        start_time = time.time()
        result = super().train(data)
        duration = time.time() - start_time
        self.log(f"Training completed in {duration:.2f} seconds")
        return result
    
    def predict(self, X_test):
        self.log(f"Prediction started for {X_test.shape[0]} samples")
        start_time = time.time()
        result = super().predict(X_test)
        duration = time.time() - start_time
        self.log(f"Prediction completed in {duration:.2f} seconds")
        return result
    
    def print_results(self, data):
        self.log("Printing model evaluation results")
        return super().print_results(data)


class ValidationMixin:
    """Adds input validation to models."""
    
    def validate_data(self, data):
        """Validate input data object."""
        if data is None:
            raise ValueError("Data object cannot be None")
        
        # Check if data object has the necessary methods
        # Support both hierarchical and chained data models
        required_methods_groups = [
            # For standard data model
            ['get_X_train', 'get_X_test', 'get_embeddings'],
            # For hierarchical data model
            ['get_type2_train_data', 'get_type2_test_data', 'get_embeddings']
        ]
        
        # Check if at least one group of methods is fully supported
        for required_methods in required_methods_groups:
            all_methods_present = True
            for method in required_methods:
                if not hasattr(data, method):
                    all_methods_present = False
                    break
            
            if all_methods_present:
                return  # At least one group is fully supported
                
        # If we get here, no group was fully supported
        raise ValueError("Data object missing required methods for any supported data model")
    
    def validate_prediction_inputs(self, X):
        """Validate inputs for prediction."""
        if X is None:
            raise ValueError("X cannot be None")
        
        if isinstance(X, np.ndarray) and X.size == 0:
            raise ValueError("X cannot be empty")
    
    def train(self, data):
        self.validate_data(data)
        return super().train(data)
    
    def predict(self, X_test):
        self.validate_prediction_inputs(X_test)
        return super().predict(X_test)


class MetricsMixin:
    """Adds performance metrics tracking to models."""
    
    def __init__(self, *args, **kwargs):
        self.metrics = {}
        super().__init__(*args, **kwargs)
    
    def train(self, data):
        start_time = time.time()
        result = super().train(data)
        self.metrics["training_time"] = time.time() - start_time
        
        # Handle both data model types
        if hasattr(data, 'get_X_train') and data.get_X_train() is not None:
            self.metrics["training_samples"] = data.get_X_train().shape[0]
        elif hasattr(data, 'get_type2_train_data'):
            X_train, _ = data.get_type2_train_data()
            if X_train is not None:
                self.metrics["training_samples"] = X_train.shape[0]
            else:
                self.metrics["training_samples"] = 0
        else:
            self.metrics["training_samples"] = 0
            
        return result
    
    def calculate_classification_metrics(self, y_true, y_pred):
        """Calculate standard classification metrics."""
        if len(y_true) == 0 or len(y_pred) == 0:
            return {"error": "Empty data for metrics calculation"}
            
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        
        # F1 score (macro)
        try:
            metrics["f1_macro"] = f1_score(y_true, y_pred, average='macro')
        except:
            metrics["f1_macro"] = 0
            
        # Number of classes
        metrics["num_classes"] = len(np.unique(y_true))
        
        return metrics
    
    def evaluate(self, X_test, y_true, y_pred):
        """Evaluate model performance."""
        start_time = time.time()
        inference_time = time.time() - start_time
        
        # Calculate metrics
        performance_metrics = self.calculate_classification_metrics(y_true, y_pred)
        
        # Store metrics
        self.metrics["inference_time"] = inference_time
        self.metrics["inference_samples"] = X_test.shape[0]
        self.metrics["inference_time_per_sample"] = inference_time / X_test.shape[0]
        self.metrics.update(performance_metrics)
        
        return self.metrics


class SerializationMixin:
    """Adds model saving/loading capabilities."""
    
    def _sanitize_filename(self, filename):
        """Sanitize a string to be used as a filename.
        
        Removes or replaces characters that might cause issues in filenames.
        """
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w\-\.]', '_', filename)
        return sanitized
    
    def save(self, filepath):
        """Save model to disk using a custom serialization approach.
        
        This approach works around limitations with pickling dynamically created classes
        by saving key components separately.
        """
        import pickle
        import json
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Instead of pickling the whole model, we'll save specific parts
        model_data = {
            'model_name': self.model_name,
            'metrics': self.metrics if hasattr(self, 'metrics') else {},
            'log_entries': self.log_entries if hasattr(self, 'log_entries') else []
        }
        
        # For ChainedRandomForest
        if hasattr(self, 'model_y2') and hasattr(self, 'model_y2_y3') and hasattr(self, 'model_y2_y3_y4'):
            # Custom serialization for chained model
            with open(f"{filepath}_model_y2.pkl", 'wb') as f:
                pickle.dump(self.model_y2, f)
            with open(f"{filepath}_model_y2_y3.pkl", 'wb') as f:
                pickle.dump(self.model_y2_y3, f)
            with open(f"{filepath}_model_y2_y3_y4.pkl", 'wb') as f:
                pickle.dump(self.model_y2_y3_y4, f)
            model_data['model_type'] = 'ChainedRandomForest'
        
        # For HierarchicalRandomForest
        elif hasattr(self, 'model_y2') and hasattr(self, 'models_y3') and hasattr(self, 'models_y4'):
            # Custom serialization for hierarchical model
            with open(f"{filepath}_model_y2.pkl", 'wb') as f:
                pickle.dump(self.model_y2, f)
            
            # Save keys for y3 and y4 models with original and sanitized versions
            model_data['models_y3_keys'] = []
            model_data['models_y4_keys'] = []
            
            # Create subdirectories for models
            os.makedirs(f"{filepath}_y3_models", exist_ok=True)
            os.makedirs(f"{filepath}_y4_models", exist_ok=True)
            
            # Save y3 models
            for key in self.models_y3.keys():
                safe_key = self._sanitize_filename(str(key))
                model_data['models_y3_keys'].append({
                    'original': str(key),
                    'safe': safe_key
                })
                
                with open(f"{filepath}_y3_models/{safe_key}.pkl", 'wb') as f:
                    pickle.dump(self.models_y3[key], f)
            
            # Save y4 models
            for (type2, type3) in self.models_y4.keys():
                safe_key = f"{self._sanitize_filename(str(type2))}_{self._sanitize_filename(str(type3))}"
                model_data['models_y4_keys'].append({
                    'original': [str(type2), str(type3)],
                    'safe': safe_key
                })
                
                with open(f"{filepath}_y4_models/{safe_key}.pkl", 'wb') as f:
                    pickle.dump(self.models_y4[(type2, type3)], f)
                    
            model_data['model_type'] = 'HierarchicalRandomForest'
        
        # Standard RandomForest or other models
        elif hasattr(self, 'classifier'):
            with open(f"{filepath}_classifier.pkl", 'wb') as f:
                pickle.dump(self.classifier, f)
            model_data['model_type'] = 'RandomForest'
        
        # Save metadata
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f)
        
        print(f"Model saved using custom serialization to {filepath}")
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """Load model from disk.
        
        This method should recreate the model from the custom serialization format.
        """
        import pickle
        import json
        from model.model_factory import standard_model
        from model.chained_randomforest import ChainedRandomForest
        from model.hierarchical_randomforest import HierarchicalRandomForest
        from model.randomforest import RandomForest
        
        # Load metadata
        try:
            with open(f"{filepath}_metadata.json", 'r') as f:
                model_data = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Could not find model metadata at {filepath}_metadata.json")
        
        model_type = model_data.get('model_type')
        
        # Recreate the appropriate model type
        if model_type == 'ChainedRandomForest':
            # Load the base model
            model = standard_model(ChainedRandomForest, model_data.get('model_name', 'ChainedRandomForest'))
            
            # Load the component models
            with open(f"{filepath}_model_y2.pkl", 'rb') as f:
                model.model_y2 = pickle.load(f)
            with open(f"{filepath}_model_y2_y3.pkl", 'rb') as f:
                model.model_y2_y3 = pickle.load(f)
            with open(f"{filepath}_model_y2_y3_y4.pkl", 'rb') as f:
                model.model_y2_y3_y4 = pickle.load(f)
                
        elif model_type == 'HierarchicalRandomForest':
            # Load the base model
            model = standard_model(HierarchicalRandomForest, model_data.get('model_name', 'HierarchicalRandomForest'))
            
            # Load the main model
            with open(f"{filepath}_model_y2.pkl", 'rb') as f:
                model.model_y2 = pickle.load(f)
            
            # Load y3 models
            for key_data in model_data.get('models_y3_keys', []):
                original_key = key_data['original']
                safe_key = key_data['safe']
                
                with open(f"{filepath}_y3_models/{safe_key}.pkl", 'rb') as f:
                    model.models_y3[original_key] = pickle.load(f)
            
            # Load y4 models
            for key_data in model_data.get('models_y4_keys', []):
                original_key = (key_data['original'][0], key_data['original'][1])
                safe_key = key_data['safe']
                
                with open(f"{filepath}_y4_models/{safe_key}.pkl", 'rb') as f:
                    model.models_y4[original_key] = pickle.load(f)
                    
        elif model_type == 'RandomForest':
            # Load the base model
            model = standard_model(RandomForest, model_data.get('model_name', 'RandomForest'))
            
            # Load classifier
            with open(f"{filepath}_classifier.pkl", 'rb') as f:
                model.classifier = pickle.load(f)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Restore metadata
        if hasattr(model, 'metrics'):
            model.metrics = model_data.get('metrics', {})
        if hasattr(model, 'log_entries'):
            model.log_entries = model_data.get('log_entries', [])
        
        print(f"Model loaded using custom serialization from {filepath}")
        return model 