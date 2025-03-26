import os
import sys
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Run each approach separately
print("============================================================")
print("ASPECT-ORIENTED MIXINS ARCHITECTURE FOR ML CLASSIFICATION")
print("============================================================")

print("\n1. Running Design Decision 1: Chained Multi-Output Classification")
print("------------------------------------------------------------")
start_time_1 = time.time()
os.system('python chained_main.py')
end_time_1 = time.time()

print("\n\n2. Running Design Decision 2: Hierarchical Modelling")
print("------------------------------------------------------------")
start_time_2 = time.time()
os.system('python hierarchical_main.py')
end_time_2 = time.time() 