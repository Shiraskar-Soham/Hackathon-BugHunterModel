from transformers import RobertaTokenizer, RobertaForCausalLM
import torch

model_path = "./finetuned_codebert_java"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForCausalLM.from_pretrained(model_path)
model.eval()

def generate_java_code(prompt, max_length=200, num_return_sequences=1):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            max_length=max_length, 
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=30,
            top_p=0.92,
            temperature=0.6,
            repetition_penalty=1.2,
            length_penalty=1.0,
            bad_words_ids=[[tokenizer.unk_token_id]]
        )
    
    generated_codes = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output]
    return generated_codes

# Example usage
prompt = "public class DateUtils{\n"
generated_codes = generate_java_code(prompt, max_length=200, num_return_sequences=3)

print("Generated Java code samples:")
for i, code in enumerate(generated_codes, 1):
    print(f"\nSample {i}:")
    print(code)