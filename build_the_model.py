from peft import LoraConfig, get_peft_model, TaskType
from model_loader import load_model_tok

model , tokenizer = load_model_tok()

lora_config = LoraConfig(
r=16,
lora_alpha=32,
target_modules=["q", "v"],
lora_dropout=0.1,
bias="none",
task_type=TaskType.SEQ_2_SEQ_LM
)
def build_model(model = model, tokenizer = tokenizer  , lora_config=lora_config):
  model = get_peft_model(model  , lora_config)
  model.print_trainable_parameters()
  return model ,tokenizer

