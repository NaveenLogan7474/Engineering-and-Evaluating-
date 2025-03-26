import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

class ChainedData():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Get individual target variables
        y2 = df.y2.to_numpy()
        y3 = df.y3.to_numpy()
        y4 = df.y4.to_numpy()

        # Create series for filtering values with enough samples
        y2_series = pd.Series(y2)
        y3_series = pd.Series(y3)
        y4_series = pd.Series(y4)

        # Filter classes with at least 3 samples
        good_y2_values = y2_series.value_counts()[y2_series.value_counts() >= 3].index
        good_y3_values = y3_series.value_counts()[y3_series.value_counts() >= 3].index
        good_y4_values = y4_series.value_counts()[y4_series.value_counts() >= 3].index

        if len(good_y2_values) < 1:
            print("None of the Type 2 classes have more than 3 records: Skipping...")
            self.X_train = None
            return

        # Filter X and y based on good values
        mask = y2_series.isin(good_y2_values) & y3_series.isin(good_y3_values) & y4_series.isin(good_y4_values)
        y2_good = y2[mask]
        y3_good = y3[mask]
        y4_good = y4[mask]
        X_good = X[mask]

        # Create combined labels
        y2_y3 = np.array([f"{y2_good[i]}+{y3_good[i]}" for i in range(len(y2_good))])
        y2_y3_y4 = np.array([f"{y2_good[i]}+{y3_good[i]}+{y4_good[i]}" for i in range(len(y2_good))])

        # Calculate new test size
        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]
        if new_test_size > 0.5:
            new_test_size = 0.5

        # Create indices for consistent data splitting
        indices = np.arange(len(y2_good))
        
        # Split indices
        train_indices, test_indices = train_test_split(
            indices, test_size=new_test_size, random_state=0, stratify=y2_good)
        
        # Use indices to split data
        self.X_train = X_good[train_indices]
        self.X_test = X_good[test_indices]
        self.y2_train = y2_good[train_indices]
        self.y2_test = y2_good[test_indices]
        self.y3_train = y3_good[train_indices]
        self.y3_test = y3_good[test_indices]
        self.y4_train = y4_good[train_indices]
        self.y4_test = y4_good[test_indices]
        
        # Split the combined labels
        self.y2_y3_train = y2_y3[train_indices]
        self.y2_y3_test = y2_y3[test_indices]
        self.y2_y3_y4_train = y2_y3_y4[train_indices]
        self.y2_y3_y4_test = y2_y3_y4[test_indices]
        
        # Store all class values and embeddings
        self.y2_classes = good_y2_values
        self.y3_classes = good_y3_values
        self.y4_classes = good_y4_values
        self.y2_y3_classes = np.unique(y2_y3)
        self.y2_y3_y4_classes = np.unique(y2_y3_y4)
        self.embeddings = X

    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_y2_train(self):
        return self.y2_train
    
    def get_y2_test(self):
        return self.y2_test
    
    def get_y3_train(self):
        return self.y3_train
    
    def get_y3_test(self):
        return self.y3_test
    
    def get_y4_train(self):
        return self.y4_train
    
    def get_y4_test(self):
        return self.y4_test
    
    def get_y2_y3_train(self):
        return self.y2_y3_train
    
    def get_y2_y3_test(self):
        return self.y2_y3_test
    
    def get_y2_y3_y4_train(self):
        return self.y2_y3_y4_train
    
    def get_y2_y3_y4_test(self):
        return self.y2_y3_y4_test
    
    def get_embeddings(self):
        return self.embeddings 