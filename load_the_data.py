import pandas as pd
from datasets import load_dataset
import argparse
import os

class LoadData:
    def loader(path="Fredithefish/Instruction-Tuning-with-GPT-4-RedPajama-Chat", output_dir="out"):
        dataset = load_dataset(path)
        df = dataset["train"].to_pandas()
        df = df.sample(10000, random_state=42)

        df["Assistant"] = df["text"].apply(lambda x: x.split("<bot>:")[-1].strip())
        df["Human"] = df["text"].apply(lambda x: "Human: " + x.split("<bot>:")[0].replace("<human>:", "").replace("\n", "").strip() + ". Assistant: ")
        df = df[["Human", "Assistant"]]

        os.makedirs(output_dir, exist_ok=True)
        df[:5000].to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
        df[5000:].to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
        print(f"Datasets saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process instruction-tuning dataset.")
    parser.add_argument("--path", type=str, default="Fredithefish/Instruction-Tuning-with-GPT-4-RedPajama-Chat", help="Dataset path on HuggingFace hub")
    parser.add_argument("--output_dir", type=str, default="out", help="Directory to save the processed parquet files")

    args = parser.parse_args()
    LoadData.loader(path=args.path, output_dir=args.output_dir)
