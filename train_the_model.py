from build_the_model import build_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import pandas as pd 
from datasets import Dataset

model , tokenizer =build_model()


train_df = pd.read_parquet("/content/Chat-bot/out/train_tokenized.parquet")

# Convert to Hugging Face Dataset
train_tokenized_dataset = Dataset.from_pandas(train_df)

 

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)
output_dir="lora-flan-t5-large-chat"
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=4,
    learning_rate=1e-3,
    num_train_epochs=1,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="epoch",
    save_strategy="epoch",
    report_to="tensorboard",
    push_to_hub = True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_tokenized_dataset,
)
model.config.use_cache = False

trainer.train()
peft_save_model_id="lora-flan-t5-large-chat"
trainer.model.save_pretrained(peft_save_model_id, push_to_hub=True)
tokenizer.save_pretrained(peft_save_model_id, push_to_hub=True)
trainer.model.base_model.save_pretrained(peft_save_model_id, push_to_hub=True)