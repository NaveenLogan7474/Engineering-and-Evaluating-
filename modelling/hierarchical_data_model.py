import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

class HierarchicalData():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Store original data
        self.X = X
        self.df = df
        
        # Get individual target variables
        self.y2 = df.y2.to_numpy()
        self.y3 = df.y3.to_numpy()
        self.y4 = df.y4.to_numpy()
        
        # Create series for filtering
        y2_series = pd.Series(self.y2)
        
        # Filter classes with at least 3 samples
        self.good_y2_values = y2_series.value_counts()[y2_series.value_counts() >= 3].index
        
        if len(self.good_y2_values) < 1:
            print("None of the Type 2 classes have more than 3 records: Skipping...")
            self.X_train = None
            return
        
        # Filter X and y based on good values for Type 2
        mask = y2_series.isin(self.good_y2_values)
        y2_good = self.y2[mask]
        X_good = X[mask]
        
        # Calculate new test size
        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        if new_test_size > 0.5:
            new_test_size = 0.5
        
        # Create indices array for data filtering
        indices = np.arange(len(self.y2))
        filtered_indices = indices[mask]
        
        # Split data for Type 2 classification
        train_idx, test_idx = train_test_split(
            np.arange(len(y2_good)), test_size=new_test_size, random_state=0, stratify=y2_good)
        
        self.X_train = X_good[train_idx]
        self.X_test = X_good[test_idx]
        self.y2_train = y2_good[train_idx]
        self.y2_test = y2_good[test_idx]
        
        # Store original indices to use for filtering y3 and y4
        self.train_indices = filtered_indices[train_idx]
        self.test_indices = filtered_indices[test_idx]
        
        # Store embeddings
        self.embeddings = X
    
    def get_type2_train_data(self):
        """Return training data for Type 2 classification"""
        return self.X_train, self.y2_train
    
    def get_type2_test_data(self):
        """Return test data for Type 2 classification"""
        return self.X_test, self.y2_test
    
    def get_type3_data_for_type2_class(self, type2_class):
        """Filter data for Type 3 based on Type 2 class"""
        # Get indices where y2 equals type2_class
        class_mask = self.y2 == type2_class
        
        # Get X and y3 values that match type2_class
        X_class = self.X[class_mask]
        y3_class = self.y3[class_mask]
        
        # Create series for filtering
        y3_series = pd.Series(y3_class)
        
        # Filter classes with at least 3 samples
        good_y3_values = y3_series.value_counts()[y3_series.value_counts() >= 3].index
        
        if len(good_y3_values) < 1:
            print(f"No Type 3 classes with more than 3 records for Type 2 class '{type2_class}': Skipping...")
            return None, None, None, None, None
        
        # Filter X and y3 based on good values
        y3_mask = y3_series.isin(good_y3_values)
        y3_good = y3_class[y3_mask]
        X_good = X_class[y3_mask]
        
        # Calculate new test size
        new_test_size = 0.2
        if len(y3_good) < 10:
            new_test_size = 0.3
        
        # Split data 
        try:
            X_train, X_test, y3_train, y3_test = train_test_split(
                X_good, y3_good, test_size=new_test_size, random_state=0, stratify=y3_good)
            return X_train, X_test, y3_train, y3_test, good_y3_values
        except ValueError:
            # Handle cases where stratification fails (not enough samples per class)
            X_train, X_test, y3_train, y3_test = train_test_split(
                X_good, y3_good, test_size=new_test_size, random_state=0)
            return X_train, X_test, y3_train, y3_test, good_y3_values
    
    def get_type4_data_for_type2_type3_class(self, type2_class, type3_class):
        """Filter data for Type 4 based on Type 2 and Type 3 class combination"""
        # Get indices where y2 equals type2_class and y3 equals type3_class
        class_mask = (self.y2 == type2_class) & (self.y3 == type3_class)
        
        # Get X and y4 values that match the specified classes
        X_class = self.X[class_mask]
        y4_class = self.y4[class_mask]
        
        # Create series for filtering
        y4_series = pd.Series(y4_class)
        
        # Filter classes with at least 3 samples
        good_y4_values = y4_series.value_counts()[y4_series.value_counts() >= 3].index
        
        if len(good_y4_values) < 1:
            print(f"No Type 4 classes with more than 3 records for Type 2 '{type2_class}' and Type 3 '{type3_class}': Skipping...")
            return None, None, None, None, None
        
        # Filter X and y4 based on good values
        y4_mask = y4_series.isin(good_y4_values)
        y4_good = y4_class[y4_mask]
        X_good = X_class[y4_mask]
        
        # Calculate new test size
        new_test_size = 0.2
        if len(y4_good) < 10:
            new_test_size = 0.3
        
        # Split data
        try:
            X_train, X_test, y4_train, y4_test = train_test_split(
                X_good, y4_good, test_size=new_test_size, random_state=0, stratify=y4_good)
            return X_train, X_test, y4_train, y4_test, good_y4_values
        except ValueError:
            # Handle cases where stratification fails (not enough samples per class)
            X_train, X_test, y4_train, y4_test = train_test_split(
                X_good, y4_good, test_size=new_test_size, random_state=0)
            return X_train, X_test, y4_train, y4_test, good_y4_values
    
    def get_type2_classes(self):
        """Return all valid Type 2 classes"""
        return self.good_y2_values
    
    def get_embeddings(self):
        """Return original embeddings"""
        return self.embeddings 