from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

MODEL_REPO = "teamobliq/ObliqGPT"

app = FastAPI()

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model = AutoModelForCausalLM.from_pretrained(MODEL_REPO)
print("Model loaded.")

BLOCKED_PATTERNS = [
    r"\bignore.{0,20}(previous|above|instructions)\b",
    r"\byou are now\b",
    r"\bfuck|shit|bitch|asshole\b",
]

def is_safe(text):
    text_lower = text.lower()
    if len(text.strip()) < 1 or len(text) > 500:
        return False
    if re.search(r"(.)\1{6,}", text):
        return False
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text_lower):
            return False
    return True

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def health_check():
    return {"status": "ObliqGPT API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    prompt_text = request.message

    if not is_safe(prompt_text):
        return ChatResponse(reply="Sorry, I can't respond to that.")

    input_text = f"<|user|> {prompt_text} <|bot|>"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    output_ids = model.generate(
        input_ids,
        max_new_tokens=40,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    try:
        bot_reply = full_text.split("<|bot|>")[1].split("<|endoftext|>")[0].strip()
    except IndexError:
        bot_reply = full_text

    if not is_safe(bot_reply):
        bot_reply = "Sorry, I'm not sure how to respond to that."

    return ChatResponse(reply=bot_reply)
