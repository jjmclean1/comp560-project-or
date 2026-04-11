import tinker
from tinker import types

# 1. AUTHENTICATION
API_KEY = "tml-HBkFduNflhHh6D6H57Brt5kCAY33vzmhNyClj6MDfrqYUxYmiFeiqnDRZULx5hvBBAAAA"
service_client = tinker.ServiceClient(api_key=API_KEY)

# 2. INITIALIZE CLIENT
training_client = service_client.create_lora_training_client(
    base_model='Qwen/Qwen3-30B-A3B-Base'
)

# 3. SETUP SAMPLER AND TOKENIZER
print("🚀 Launching Tinker Sampling Test...")
sampling_client = training_client.save_weights_and_get_sampling_client(name='baseline-sample-a')

# Grab the model's specific tokenizer
tokenizer = training_client.get_tokenizer()

# 4. FIX: Encode the text into integer tokens, then build the ModelInput
raw_text = "Rank gallery images for query_001"
tokens = tokenizer.encode(raw_text)
prompt_input = types.ModelInput.from_ints(tokens=tokens)

# 5. RUN IT
# 5. RUN IT
for i in range(5):
    future = sampling_client.sample(
        prompt=prompt_input, 
        sampling_params={}, 
        num_samples=1
    )
    
    result = future.result() 
    
    # NEW: Extract the raw numbers from the response object
    output_tokens = result.sequences[0].tokens
    
    # NEW: Translate the numbers back into readable text
    readable_text = tokenizer.decode(output_tokens)
    
    print(f"--- Sample {i} ---")
    print(readable_text)
    print("------------------\n")

print("✅ Success! Check your Tinker Console dashboard for logs.")