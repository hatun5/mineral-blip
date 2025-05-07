import os
import io
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from dotenv import load_dotenv
from groq import Groq

app = FastAPI()

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.get("/index.html")
async def serve_index_html():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Missing GROQ_API_KEY in .env")

groq_client = Groq(api_key=GROQ_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

last_caption = ""

@app.post("/caption")
async def get_caption(
    file: UploadFile = File(...),
    prompt: str = Form("")
):
    global last_caption
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    blip_inputs = processor(image, return_tensors="pt")
    blip_output = blip_model.generate(
        **blip_inputs,
        max_new_tokens=100,
        do_sample=False,
        top_p=1.0,
        temperature=0.0
    )
    caption = processor.decode(blip_output[0], skip_special_tokens=True)
    last_caption = caption

    llama_prompt = f"""
The image shows the following content:
\"{caption}\"

First, provide only the mineral label and description (max 20 words).
Then, if a user question is provided, clearly answer it in no more than 40 words.

User's question:
\"{prompt}\"
"""

    conversation_messages = [
        {"role": "system", "content": "You are an expert mineral assistant that generates short, precise descriptions and answers user questions."},
        {"role": "user", "content": llama_prompt.strip()}
    ]

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation_messages,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False
    )

    response_text = completion.choices[0].message.content

    formatted_response = f"label minralblip is: {caption}\n"
    formatted_response += f"lamaa Caption is: {response_text}\n"

    return JSONResponse(content={"formatted_response": formatted_response})


@app.post("/chat")
async def chat_only(prompt: str = Form("")):
    global last_caption
    if not last_caption:
        return JSONResponse(content={"error": "No previous image caption available."}, status_code=400)
    llama_prompt = f"""
Based on this previous mineral description:
\"{last_caption}\"

Answer this user question:
\"{prompt}\"
"""
    conversation_messages = [
        {"role": "system", "content": "You are an expert mineral assistant that answers user questions about minerals."},
        {"role": "user", "content": llama_prompt.strip()}
    ]
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversation_messages,
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False
    )
    response_text = completion.choices[0].message.content
    return JSONResponse(content={"answer": response_text})
