from preprocess import *
from embeddings import *
from model.chained_randomforest import ChainedRandomForest
from modelling.chained_data_model import ChainedData
from model.model_factory import standard_model
from mixins import LoggingMixin, ValidationMixin, MetricsMixin, SerializationMixin
import random
import os

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """Load the input data."""
    df = get_input_data()
    return df


def preprocess_data(df):
    """Preprocess the data."""
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    # Uncomment if translation is needed
    # df[Config.TICKET_SUMMARY] = translate_to_en(df[Config.TICKET_SUMMARY].tolist())
    return df


def get_embeddings(df: pd.DataFrame):
    """Get TF-IDF embeddings from the data."""
    X = get_tfidf_embd(df)  # Get tf-idf embeddings
    return X, df


def get_chained_data_object(X: np.ndarray, df: pd.DataFrame):
    """Create a ChainedData object with the given data."""
    return ChainedData(X, df)


def perform_chained_modelling(data: ChainedData, df: pd.DataFrame, name):
    """Perform chained multi-label classification."""
    print(f"\n===== Processing {name} with Chained Classification =====")
    
    # Skip if data processing resulted in insufficient samples
    if data.get_X_train() is None:
        print(f"Skipping {name} due to insufficient data")
        return
    
    # Create model with aspect-oriented mixins
    # Order matters: Serialization, Metrics, Validation, Logging
    model = standard_model(
        ChainedRandomForest,
        "ChainedRandomForest", 
        data.get_embeddings()
    )
    
    # Train, predict, and evaluate the model
    model.train(data)
    model.predict(data.get_X_test())
    model.print_results(data)
    
    # Save the model
    model_path = f"models/{name}_chained_model.pkl"
    try:
        os.makedirs("models", exist_ok=True)
        saved_path = model.save(model_path)
        print(f"Model saved to {saved_path}")
    except Exception as e:
        print(f"Could not save model: {str(e)}")
    
    return model


if __name__ == '__main__':
    print("========== DESIGN DECISION 1: CHAINED MULTI-OUTPUT CLASSIFICATION ==========")
    
    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)
    
    # Convert data types to Unicode
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    
    # Group data by Type 1 (y1) and process each group
    grouped_df = df.groupby(Config.GROUPED)
    for name, group_df in grouped_df:
        print(f"\nProcessing group: {name}")
        
        # Get embeddings and create data object
        X, group_df = get_embeddings(group_df)
        data = get_chained_data_object(X, group_df)
        
        # Perform chained classification
        model = perform_chained_modelling(data, group_df, name) 