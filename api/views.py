#with hugging face

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# # Load model and tokenizer once
# tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model = AutoModelForCausalLM.from_pretrained(
#     "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     torch_dtype=torch.float32
# ).to("cpu")

# class TinyLlamaView(APIView):
#     def post(self, request):
#         user_input = request.data.get("prompt", "")
        
#         # Format the prompt for chat-style conversation
#         chat_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"

#         # Tokenize input properly
#         inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

#         # Generate response
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=200,  # avoid using max_length
#             do_sample=True,
#             top_p=0.9,
#             temperature=0.7,
#             pad_token_id=tokenizer.eos_token_id
#         )

#         # Decode and slice assistant response
#         full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("RAW OUTPUT:", repr(full_output))

#         # Extract only the assistant's reply after "<|assistant|>"
#         response_text = full_output.split("<|assistant|>")[-1].strip()

#         return Response({"response": response_text})




#without hugging face


from rest_framework.views import APIView
from rest_framework.response import Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Local path where model is stored
LOCAL_MODEL_PATH = "./models/TinyLlama"

# Load model and tokenizer once
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float32
).to("cpu")

class TinyLlamaView(APIView):
    def post(self, request):
        user_input = request.data.get("prompt", "")
        
        chat_prompt = f"<|system|>\nYou are a helpful assistant.\n<|user|>\n{user_input}\n<|assistant|>\n"
        
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )

        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("RAW OUTPUT:", repr(full_output))

        response_text = full_output.split("<|assistant|>")[-1].strip()
        
        return Response({"response": response_text})
    



    
