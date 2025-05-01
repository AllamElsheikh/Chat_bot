from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_tok(model_id="google/flan-t5-large" ):
  model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
  tokenizer =AutoTokenizer.from_pretrained(model_id)
  

  return model , tokenizer 
  