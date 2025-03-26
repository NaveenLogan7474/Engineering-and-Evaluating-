# 📧 Multi-Label Email Classification System

## 🔍 Project Overview
This project implements a modular, extensible architecture for multi-label email classification using two different architectural approaches: Chained Multi-Output Classification and Hierarchical Modeling. The system classifies support emails across three dependent variables (Type 2, Type 3, and Type 4) using a structured approach that separates concerns, encapsulates data, and employs aspect-oriented programming techniques.

## 📁 Directory Structure
```
EE_NCI_SET_5/
├── Complete -Solution/
│   └── Actvity 3 Full Solution/
│       ├── data/                      # Data directory
│       │   ├── AppGallery.csv         # AppGallery support tickets
│       │   └── Purchasing.csv         # Purchasing support tickets
│       ├── model/                     # Model implementation
│       │   ├── __init__.py
│       │   ├── base.py                # Abstract base model
│       │   ├── randomforest.py        # Base RandomForest implementation
│       │   ├── chained_randomforest.py # Chained approach
│       │   ├── hierarchical_randomforest.py # Hierarchical approach
│       │   └── model_factory.py       # Factory for creating models with mixins
│       ├── modelling/                 # Data modeling
│       │   ├── data_model.py          # Base data model
│       │   ├── chained_data_model.py  # Data model for chained approach
│       │   ├── hierarchical_data_model.py # Data model for hierarchical approach
│       │   └── modelling.py           # Legacy modeling code
│       ├── models/                    # Directory for saved models
│       ├── Config.py                  # Configuration settings
│       ├── compare_approaches.py      # Script to run and compare both approaches
│       ├── chained_main.py            # Main script for chained approach
│       ├── hierarchical_main.py       # Main script for hierarchical approach
│       ├── embeddings.py              # Text to vector embeddings
│       ├── main.py                    # Legacy main script
│       ├── mixins.py                  # Aspect-oriented mixins
│       ├── preprocess.py              # Data preprocessing functions
│       ├── USAGE_GUIDE.md             # Quick usage guide
│       └── README.md                  # Original readme file
```

## 📊 System Architecture

### 🔄 Data Flow
```
┌────────────┐    ┌─────────────┐    ┌──────────────┐    ┌───────────┐    ┌────────────┐
│ CSV Files  │───►│ Preprocess  │───►│ Embeddings   │───►│ Data Model│───►│ ML Models  │
└────────────┘    └─────────────┘    └──────────────┘    └───────────┘    └────────────┘
                        │                                       ▲                │
                        │                                       │                │
                        │                  ┌─────────────────┐  │                │
                        └─────────────────►│ Config Settings │◄─┘                │
                                           └─────────────────┘                   │
                                                                                 ▼
                                                                          ┌────────────┐
                                                                          │ Evaluation │
                                                                          └────────────┘
```

### 🧩 Class Inheritance Diagram

```
BaseModel (Abstract)
    │
    ├─── RandomForest
    │         
    ├─── ChainedRandomForest
    │     (with mixins: LoggingMixin, ValidationMixin, MetricsMixin, SerializationMixin)
    │    
    └─── HierarchicalRandomForest
          (with mixins: LoggingMixin, ValidationMixin, MetricsMixin, SerializationMixin)
```

## 📄 File Descriptions and Connections

### 🚀 Main Scripts

#### `compare_approaches.py`
Entry point for running both architectural approaches sequentially. It:
- Executes `chained_main.py` for the Chained Multi-Output approach
- Executes `hierarchical_main.py` for the Hierarchical Modeling approach
- Measures and reports execution time for each approach

#### `chained_main.py`
Implements the Chained Multi-Output Classification approach:
- Loads and preprocesses data from CSV files
- Creates TF-IDF embeddings
- Instantiates the `ChainedData` object
- Creates a `ChainedRandomForest` model with mixins
- Trains, predicts, and evaluates the model
- Saves the model to disk

#### `hierarchical_main.py`
Implements the Hierarchical Modeling approach:
- Loads and preprocesses data from CSV files 
- Creates TF-IDF embeddings
- Instantiates the `HierarchicalData` object
- Creates a `HierarchicalRandomForest` model with mixins
- Trains, predicts, and evaluates the model
- Saves the model to disk

### ⚙️ Configuration and Data Processing

#### `Config.py`
Contains configuration constants used throughout the application:
- Input column names
- Type column names
- Class column name
- Grouped column name

```python
class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'
```

#### `preprocess.py`
Contains functions for data preprocessing:
- `get_input_data()`: Loads data from CSV files
- `de_duplication()`: Removes duplicate content
- `noise_remover()`: Cleans text data
- `translate_to_en()`: Translates non-English text to English

Key preprocessing steps:
1. Load data from CSV files
2. Remove duplicates
3. Clean text data (remove noise, standardize format)
4. Optional: Translate non-English text

#### `embeddings.py`
Handles text-to-vector conversion:
- `get_tfidf_embd()`: Generates TF-IDF embeddings from text data
- `combine_embd()`: Combines multiple embeddings

```python
def get_tfidf_embd(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
    X = tfidfconverter.fit_transform(data).toarray()
    return X
```

### 💾 Data Models

#### `modelling/data_model.py`
Base data model that:
- Filters classes with insufficient data
- Splits data into training and testing sets
- Provides accessor methods for data

```python
class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Filtering records with fewer than 3 instances
        y = df.y.to_numpy()
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
        
        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good)
```

#### `modelling/chained_data_model.py`
Specialized data model for the chained approach that:
- Creates combined labels for Type 2+3 and Type 2+3+4
- Ensures consistent train/test splits
- Provides methods for accessing individual and combined labels

Key data transformations:
```python
# Create combined labels
y2_y3 = np.array([f"{y2_good[i]}+{y3_good[i]}" for i in range(len(y2_good))])
y2_y3_y4 = np.array([f"{y2_good[i]}+{y3_good[i]}+{y4_good[i]}" for i in range(len(y2_good))])
```

#### `modelling/hierarchical_data_model.py`
Specialized data model for the hierarchical approach that:
- Filters data for each level of the hierarchy
- Creates specialized datasets for each Type 2 class and Type 2+Type 3 combination
- Provides methods for accessing data at each level of the hierarchy

Key methods:
```python
def get_type3_data_for_type2_class(self, type2_class)
def get_type4_data_for_type2_type3_class(self, type2_class, type3_class)
```

### 🤖 Model Implementations

#### `model/base.py`
Abstract base class that defines the model interface:
- `train()`: Abstract method for training models
- `predict()`: Abstract method for making predictions
- `print_results()`: Abstract method for evaluating and reporting results

```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, data) -> None:
        """Train the model using ML Models for Multi-class and mult-label classification."""
        pass

    @abstractmethod
    def predict(self, X_test) -> np.ndarray:
        """Make predictions on test data."""
        pass

    @abstractmethod
    def print_results(self, data) -> None:
        """Print evaluation results."""
        pass
```

#### `model/randomforest.py`
Base RandomForest implementation that:
- Extends the BaseModel abstract class
- Implements training, prediction, and evaluation for a single target variable

Base classifier settings:
```python
self.classifier = RandomForestClassifier(
    n_estimators=1000, 
    random_state=seed, 
    class_weight='balanced_subsample'
)
```

#### `model/chained_randomforest.py`
Implementation for the Chained Multi-Output approach that:
- Contains three internal RandomForest models
- Handles training for Type 2, Type 2+3, and Type 2+3+4
- Provides comprehensive evaluation metrics for each level

```python
# Train the model for Type 2
self.model_y2 = self.model_y2.fit(data.get_X_train(), data.get_y2_train())

# Train the model for Type 2 + Type 3
self.model_y2_y3 = self.model_y2_y3.fit(data.get_X_train(), data.get_y2_y3_train())

# Train the model for Type 2 + Type 3 + Type 4
self.model_y2_y3_y4 = self.model_y2_y3_y4.fit(data.get_X_train(), data.get_y2_y3_y4_train())
```

#### `model/hierarchical_randomforest.py`
Implementation for the Hierarchical Modeling approach that:
- Contains multiple nested RandomForest models
- Manages a hierarchy of models for Type 2, Type 3 for each Type 2 class, and Type 4 for each Type 2+Type 3 combination
- Provides detailed evaluation metrics for each branch of the hierarchy

```python
# For each Type 2 class, train a model for Type 3
for type2_class in data.get_type2_classes():
    # Get data for this Type 2 class
    X3_train, X3_test, y3_train, y3_test, _ = data.get_type3_data_for_type2_class(type2_class)
    
    # Create and train Type 3 model for this Type 2 class
    model_y3 = RandomForestClassifier(n_estimators=1000, random_state=seed, class_weight='balanced_subsample')
    model_y3 = model_y3.fit(X3_train, y3_train)
    self.models_y3[type2_class] = model_y3
```

#### `model/model_factory.py`
Factory for dynamically creating models with mixins:
- `create_model()`: Creates a model with specified mixins
- `standard_model()`: Creates a model with all standard mixins

Dynamic class creation:
```python
def create_model(base_class, mixins=None, *args, **kwargs):
    if mixins:
        # Create a class name that uniquely identifies this combination
        class_name = f"{''.join(m.__name__.replace('Mixin', '') for m in mixins)}{base_class.__name__}"
        
        # Define the new class with the mixins
        combined_class = type(
            class_name,
            tuple(mixins) + (base_class,),
            {}
        )
```

### 🧵 Aspect-Oriented Programming

#### `mixins.py`
Implements cross-cutting concerns using mixins:
- `LoggingMixin`: Adds logging capabilities to track execution
- `ValidationMixin`: Adds input validation for data integrity
- `MetricsMixin`: Tracks performance metrics
- `SerializationMixin`: Handles model saving and loading with custom serialization

Mixin functionality:
```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│ LoggingMixin  │     │ValidationMixin│     │  MetricsMixin │     │SerializeMixin │
└───────┬───────┘     └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
        │                     │                     │                     │
        │ Logging             │ Validation          │ Metrics             │ Persistence
        │                     │                     │                     │
        ▼                     ▼                     ▼                     ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                  Base Model                                      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Running Guide

### 📋 Prerequisites
- Python 3.8 or higher
- Required packages: scikit-learn, pandas, numpy

### 💻 Installation
```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install required packages
pip install scikit-learn pandas numpy
```

### ⚙️ Running the Code

#### Compare Both Approaches
To run and compare both architectural approaches:
```bash
python compare_approaches.py
```

#### Run Individual Approaches
To run only the Chained Multi-Output approach:
```bash
python chained_main.py
```

To run only the Hierarchical Modeling approach:
```bash
python hierarchical_main.py
```

### 📑 Understanding the Output

#### Chained Multi-Output Classification Results
The output shows:
1. **Type 2 Classification Results**: Precision, recall, f1-score for Type 2 classes
2. **Type 2+3 Combined Classification Results**: Metrics for combined Type 2+3 classes
3. **Type 2+3+4 Combined Classification Results**: Metrics for combined Type 2+3+4 classes
4. **Type 3 & Type 4 Individual Results**: Metrics extracted from combined predictions

#### Hierarchical Modeling Results
The output shows:
1. **Type 2 Classification Results**: Metrics for the top-level classification
2. **Type 3 Classification Results**: Metrics for each Type 2 class
3. **Type 4 Classification Results**: Metrics for each Type 2+Type 3 combination

## 🔧 Implementation Details

### 🔗 Design Decision 1: Chained Multi-Output Classification
In this approach, a single model instance handles multiple classification tasks in a chain:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   model_y2  │────►│ model_y2_y3 │────►│model_y2_y3_y4│
└─────────────┘     └─────────────┘     └─────────────┘
      Type 2           Type 2+3           Type 2+3+4
```

The `ChainedRandomForest` maintains three internal models:
- `model_y2`: For Type 2 classification
- `model_y2_y3`: For Type 2+3 classification
- `model_y2_y3_y4`: For Type 2+3+4 classification

This creates a unified approach where classification spans all three dependent variables.

### 🌳 Design Decision 2: Hierarchical Modeling
In this approach, multiple models are arranged in a hierarchy:

```
                ┌─────────────┐
                │  model_y2   │
                └──────┬──────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼──────┐┌──────▼──────┐┌──────▼──────┐
│models_y3[A]  ││models_y3[B] ││models_y3[C] │
└───────┬──────┘└──────┬──────┘└──────┬──────┘
        │              │              │
  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐
  │models_y4  │  │models_y4  │  │models_y4  │
  │ [A,X]     │  │ [B,Y]     │  │ [C,Z]     │
  └───────────┘  └───────────┘  └───────────┘
```

The `HierarchicalRandomForest` maintains:
- `model_y2`: Top-level model
- `models_y3`: Dictionary of models keyed by Type 2 class
- `models_y4`: Dictionary of models keyed by (Type 2, Type 3) tuples

This creates specialized models for each branch of the classification hierarchy.

### 🧩 Aspect-Oriented Mixins Architecture
The project employs mixins to separate cross-cutting concerns:
1. **LoggingMixin**: Adds execution time tracking
2. **ValidationMixin**: Validates input data
3. **MetricsMixin**: Collects performance metrics
4. **SerializationMixin**: Handles model persistence

These mixins can be combined with any model through the `model_factory.py` module.

## 📊 Results

### 📈 Chained Multi-Output Classification Results

```
Processing group: AppGallery & Games

--- Type 2 Classification Results ---
               precision    recall  f1-score   support

Problem/Fault       0.83      1.00      0.90        19
   Suggestion       1.00      0.33      0.50         6

     accuracy                           0.84        25
    macro avg       0.91      0.67      0.70        25
 weighted avg       0.87      0.84      0.81        25


--- Type 2+3 Combined Classification Results ---
                                          precision    recall  f1-score   support

Problem/Fault+AppGallery-Install/Upgrade       0.71      0.83      0.77         6
            Problem/Fault+AppGallery-Use       1.00      1.00      1.00         2
Problem/Fault+Coupon/Gifts/Points Issues       0.54      1.00      0.70         7
          Problem/Fault+Third Party APPs       1.00      0.25      0.40         4
               Suggestion+AppGallery-Use       0.00      0.00      0.00         3
                      Suggestion+General       0.00      0.00      0.00         0
    Suggestion+VIP / Offers / Promotions       1.00      0.33      0.50         3

                                accuracy                           0.64        25
                               macro avg       0.61      0.49      0.48        25
                            weighted avg       0.68      0.64      0.58        25


--- Type 3 Individual Classification Results (Extracted from Combined) ---
                            precision    recall  f1-score   support   

AppGallery-Install/Upgrade       0.71      0.83      0.77         6   
            AppGallery-Use       1.00      0.40      0.57         5   
Coupon/Gifts/Points Issues       0.54      1.00      0.70         7   
                   General       0.00      0.00      0.00         0   
          Third Party APPs       1.00      0.25      0.40         4   
 VIP / Offers / Promotions       1.00      0.33      0.50         3   

                  accuracy                           0.64        25   
                 macro avg       0.71      0.47      0.49        25   
              weighted avg       0.80      0.64      0.62        25   
```

Processing group: In-App Purchase

```
--- Type 2 Classification Results ---
               precision    recall  f1-score   support

Problem/Fault       1.00      1.00      1.00         2
   Suggestion       1.00      1.00      1.00        15

     accuracy                           1.00        17
    macro avg       1.00      1.00      1.00        17
 weighted avg       1.00      1.00      1.00        17


--- Type 3 Individual Classification Results (Extracted from Combined) ---
               precision    recall  f1-score   support

      Invoice       1.00      1.00      1.00         1
      Payment       1.00      1.00      1.00        14
Payment issue       1.00      1.00      1.00         2

     accuracy                           1.00        17
    macro avg       1.00      1.00      1.00        17
 weighted avg       1.00      1.00      1.00        17
```

### 📊 Hierarchical Modeling Results

Processing group: AppGallery & Games

```
--- Type 2 Classification Results ---
               precision    recall  f1-score   support

       Others       0.83      0.71      0.77         7
Problem/Fault       0.75      0.86      0.80        14
   Suggestion       0.33      0.25      0.29         4

     accuracy                           0.72        25
    macro avg       0.64      0.61      0.62        25
 weighted avg       0.71      0.72      0.71        25


--- Type 3 Classification Results ---

Type 2 Class: Problem/Fault
                            precision    recall  f1-score   support   

AppGallery-Install/Upgrade       0.57      1.00      0.73         4   
            AppGallery-Use       0.00      0.00      0.00         1   
Coupon/Gifts/Points Issues       1.00      0.80      0.89         5   
                   General       0.00      0.00      0.00         1   
          Third Party APPs       1.00      1.00      1.00         2   
 VIP / Offers / Promotions       0.00      0.00      0.00         1   

                  accuracy                           0.71        14   
                 macro avg       0.43      0.47      0.44        14   
              weighted avg       0.66      0.71      0.67        14   
```

Processing group: In-App Purchase

```
--- Type 2 Classification Results ---
               precision    recall  f1-score   support

Problem/Fault       0.00      0.00      0.00         2
   Suggestion       0.88      0.93      0.90        15

     accuracy                           0.82        17
    macro avg       0.44      0.47      0.45        17
 weighted avg       0.77      0.82      0.80        17


--- Type 3 Classification Results ---

Type 2 Class: Suggestion
              precision    recall  f1-score   support

     Invoice       1.00      0.50      0.67         2
       Other       0.00      0.00      0.00         1
     Payment       0.86      1.00      0.92        12

    accuracy                           0.87        15
   macro avg       0.62      0.50      0.53        15
weighted avg       0.82      0.87      0.83        15
```

## 📊 Performance Comparison

### 🔄 Comparison Metrics Table

| Metric | Chained Approach | Hierarchical Approach |
|--------|------------------|------------------------|
| **Training Time** | 3-4 seconds | 5-11 seconds |
| **Type 2 Accuracy** | 0.84-1.00 | 0.72-0.82 |
| **Type 3 Accuracy** | 0.64-1.00 | 0.71-0.87 |
| **Memory Usage** | Lower | Higher |
| **Implementation Complexity** | Lower | Higher |
| **Granularity of Analysis** | Lower | Higher |

### 🚀 Chained Multi-Output Approach
**Strengths:**
- ✅ Simpler implementation (fewer models)
- ✅ Faster training (3-4 seconds vs. 5-11 seconds for hierarchical)
- ✅ Unified approach with a single model instance
- ✅ Higher overall accuracy for Type 2 classification

**Limitations:**
- ❌ Cannot provide specialized metrics for each Type 2 class
- ❌ Extraction of individual Type 3 and Type 4 predictions is required
- ❌ Less interpretability of intermediate results

### 🌳 Hierarchical Modeling Approach
**Strengths:**
- ✅ More detailed metrics for each level of the hierarchy
- ✅ Specialized models for each Type 2 class
- ✅ Better transparency into model performance
- ✅ More interpretable results at each level

**Limitations:**
- ❌ More complex implementation
- ❌ Slower training (requires multiple models)
- ❌ Some Type 2+Type 3 combinations might lack sufficient data
- ❌ Higher memory requirements

## 🔍 Conclusion

This implementation demonstrates two different architectural approaches to multi-label email classification while maintaining a modular, extensible design. Both approaches have their strengths and can be chosen based on specific requirements:

- Use the **Chained approach** for:
  - ⚡ Simplicity and speed
  - 🔧 Easier maintenance
  - 💾 Lower memory requirements
  - 🔄 When overall performance is more important than per-class insights

- Use the **Hierarchical approach** for:
  - 📊 More detailed metrics at each level
  - 🎯 When specialized models for each class are needed
  - 🔍 Greater interpretability of results
  - 🧩 When the hierarchical structure of classes is important

The aspect-oriented mixins architecture provides clean separation of concerns and makes the system extensible for future enhancements. 