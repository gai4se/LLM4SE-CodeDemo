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

java_function ="public boolean isEven(int num){ //Complete"
encoding = tokenizer (java_function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=300)
generated_code = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(generated_code)
#Output - public boolean isEven(int num){ //Complete return num%2==0; }

java_function ="public int sumNumbers(int a, int b){ //Complete"
encoding = tokenizer (java_function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=300)
generated_code = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(generated_code)
#Output - public int sumNumbers(int a, int b){ //Complete    return a + b; //Complete}

java_function ="public int findMaximum(int a, int b){ //Complete"
encoding = tokenizer (java_function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=300)
generated_code = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(generated_code)

''' Output - public int findMaximum(int a, int b){ //Complete    if(a>b){ //If a is greater than b
        return a; //If a is less than b
    }
    return b; //If a is equal to b
}
'''

java_function ="public int factorial(int num){ //Complete"
encoding = tokenizer (java_function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=300)
generated_code = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(generated_code)

#Output - public int factorial(int num){ //Complete return num! * (num! + 1) // 2 }

java_function ="public String reverseString(String input){ //Complete"
encoding = tokenizer (java_function, return_tensors="pt").to(device)
encoding['decoder_input_ids'] = encoding['input_ids'].clone()
outputs = model.generate(**encoding, max_length=300)
generated_code = tokenizer.decode (outputs[0], skip_special_tokens=True)

print(generated_code)

#Output - public String reverseString(String input){ //Complete return input[::-1]; //Complete}
