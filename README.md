# chunk-based-sampling

try

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pre-trained model tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Set padding token
tokenizer.pad_token = tokenizer.eos_token

# Load pre-trained model (weights)
model = AutoModelForCausalLM.from_pretrained("gpt2")

sampled_text = chunk_continuation(model, 
                                  tokenizer, 
                                  "We are entering the winter of 2002. Outside, it is ", 
                                  ["rainy", 
                                   "sunny", 
                                   "snowy", 
                                   "cold",
                                   "hot"],
                                  ".",
                                  sum=False,
                                  verbose=True)

print(sampled_text)
```