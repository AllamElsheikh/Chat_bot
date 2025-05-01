import pandas as pd
import os
import argparse
from model_loader import load_model_tok

# Load the model and tokenizer from our separate module
model, tokenizer = load_model_tok()

def preprocess_function(sample, padding="max_length", tokenizer=tokenizer):
    model_inputs = tokenizer(sample["Human"], max_length=256, padding=padding, truncation=True)
    labels = tokenizer(sample["Assistant"], max_length=256, padding=padding, truncation=True)
    if padding == "max_length":
        labels["input_ids"] = [
            l if l != tokenizer.pad_token_id else -100 for l in labels["input_ids"]
        ]
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def prepare_data(data_path="/Chat-bot/out", output_path="/Chat-bot/out"):
    # Construct file paths for input datasets
    train_path = os.path.join(data_path, "train.parquet")
    test_path = os.path.join(data_path, "test.parquet")
    
    # Read the parquet files into pandas DataFrames
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    # Process the training data
    processed_train = train_df.apply(lambda row: preprocess_function(row, padding="max_length"), axis=1)
    processed_train_df = pd.DataFrame(list(processed_train))
    
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Save processed training data to a file
    train_tokenized_file = os.path.join(output_path, "train_tokenized.parquet")
    processed_train_df.to_parquet(train_tokenized_file, index=False)
    
    # Process the testing data
    processed_test = test_df.apply(lambda row: preprocess_function(row, padding="max_length"), axis=1)
    processed_test_df = pd.DataFrame(list(processed_test))
    
    # Save processed testing data to a file
    test_tokenized_file = os.path.join(output_path, "test_tokenized.parquet")
    processed_test_df.to_parquet(test_tokenized_file, index=False)
    
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and save model inputs.")
    parser.add_argument("data_path", type=str, help="Path to the input dataset")
    parser.add_argument("output_path", type=str, help="Path to save the processed dataset")
    args = parser.parse_args()
    prepare_data(args.data_path, args.output_path)
