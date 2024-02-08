from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

checkpoint = "Salesforce/codet5p-770m-py"  # codet5p-2b codet5p-6b codet5p-16b codet5p-770m-py
device = "cuda"  # for GPU usage or "cpu" for CPU usage
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                              torch_dtype=torch.float16,
                                              # load_in_8bit=True,
                                              # device_map="auto",
                                              trust_remote_code=True).to(device)
print(model.get_memory_footprint())

# python_function ="Write python code for linear regression" # text-to-code
python_function ="write python code to print first 10 primes"
encoding = tokenizer (python_function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=175)
generated_code = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(generated_code)