def generate_seq(model, tokenizer, input):
    generated_ids = model.generate(**input)
    generated_text = tokenizer.decode(generated_ids.squeeze(0), skip_special_tokens=True)
    
    return generated_text

def generate_input_target(model, tokenizer, input, label):
    input_text = tokenizer.decode(input['input_ids'].squeeze(0), skip_special_tokens=True)
    generated_text = generate_seq(model, tokenizer, input)
    target_text = tokenizer.decode(label.squeeze(0), skip_special_tokens=True)
    
    return {
        'input_text': input_text,
        'generated_text': generated_text, 
        'target_text': target_text
    }

def generate_from_data(model, tokenizer, data):
    label = data['labels']
    input_data = dict()
    input_data['input_ids'] = data['input_ids']
    input_data['attention_mask'] = data['attention_mask']

    return generate_input_target(model, tokenizer, input_data, label)

def eval(model, tokenizer, input_seq, label, metric, options = dict()):
    generated_input_target = generate_input_target(model, tokenizer, input_seq, label)
    score = metric.compute(
        generated_input_target['generated_text'], 
        generated_input_target['target_text'],
        **options
    )

    return score
