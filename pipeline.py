from transformers import pipeline

def get_pipeline(pipeline_name, tokenizer, model):
    return pipeline(pipeline_name, tokenizer=tokenizer, model=model)
