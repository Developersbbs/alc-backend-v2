
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mimetypes, base64, os
import google.genai as genai
from google.genai import types
from typing import List, Optional

MODEL_NAME = "gemini-2.5-flash-image-preview"

app = FastAPI(title="Hair Growth Simulation API")

# Allow all CORS (safe for dev, restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Helpers
# ---------------------------
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=api_key)

def _get_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "image/png"

def _process_api_stream_response(stream) -> bytes | None:
    for chunk in stream:
        if not chunk.candidates:
            continue
        candidate = chunk.candidates[0]
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data
    return None

# ---------------------------
# Routes
# ---------------------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend running"}

@app.post("/generate/batch")
async def generate_batch_images(
    images: List[UploadFile] = File(...),
    timeframe: str = Form(...),
    hair_colors: Optional[str] = Form(None),
    hair_types: Optional[str] = Form(None),
    hair_density: Optional[str] = Form(None),
    hair_line_types: Optional[str] = Form(None)  # <-- NEW
):
    import json
    client = get_client()
    
    # Parse arrays
    colors_list = json.loads(hair_colors) if hair_colors else []
    types_list = json.loads(hair_types) if hair_types else []
    density_list = json.loads(hair_density) if hair_density else []
    line_types_list = json.loads(hair_line_types) if hair_line_types else []

    results = []

    for idx, image in enumerate(images):
        if not image.content_type or not image.content_type.startswith('image/'):
            results.append({"index": idx, "filename": image.filename, "error": "File must be an image"})
            continue

        hair_color = colors_list[idx] if idx < len(colors_list) else "#000000"
        hair_type_value = types_list[idx] if idx < len(types_list) else "Unknown"
        hair_density_value = density_list[idx] if idx < len(density_list) else 0.5
        hair_line_value = line_types_list[idx] if idx < len(line_types_list) else "Hairline"

        data = await image.read()
        months = "3" if timeframe == "3months" else "8"
        color_instruction = f" Use hair color: {hair_color}" if hair_color != "#000000" else ""
        
        # Include hair type, density & line type in prompt
        prompt = (
            f"Generate a realistic image showing natural hair regrowth after {months} months of treatment. "
            f"Hair type: {hair_type_value}, Hair density: {hair_density_value:.2f}, Focus area: {hair_line_value}. "
            f"Show {'visible improvement' if months == '3' else 'significant improvement'} filling in thin areas. "
            f"Keep the person's facial features identical.{color_instruction} "
            f"Make the hair look natural and realistic for a {months}-month regrowth period."
        )

        contents = [
            types.Part(inline_data=types.Blob(data=data, mime_type=image.content_type)),
            types.Part.from_text(text=prompt),
        ]
        config = types.GenerateContentConfig(response_modalities=["IMAGE"])
        stream = client.models.generate_content_stream(model=MODEL_NAME, contents=contents, config=config)
        image_data = _process_api_stream_response(stream)

        if not image_data:
            results.append({"index": idx, "filename": image.filename, "error": "Failed to generate image"})
            continue

        results.append({
            "index": idx,
            "filename": image.filename,
            "image": base64.b64encode(image_data).decode("utf-8"),
            "hair_color": hair_color,
            "hair_type": hair_type_value,
            "hair_density": hair_density_value,
            "hair_line_type": hair_line_value,
            "timeframe": timeframe
        })

    return {
        "results": results,
        "timeframe": timeframe,
        "message": f"Generated {len([r for r in results if 'image' in r])} out of {len(images)} images"
    }