
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mimetypes, base64, os,json
import google.genai as genai
from google.genai import types
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
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
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=api_key)

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
    hair_line_types: Optional[str] = Form(None)
):
    client = get_client()
    
    # Parse arrays from form data
    colors_list = json.loads(hair_colors) if hair_colors else []
    types_list = json.loads(hair_types) if hair_types else []
    density_list = json.loads(hair_density) if hair_density else []
    line_types_list = json.loads(hair_line_types) if hair_line_types else []

    results = []

    for idx, image in enumerate(images):
        if not image.content_type or not image.content_type.startswith('image/'):
            results.append({
                "index": idx,
                "filename": image.filename,
                "error": "File must be an image"
            })
            continue

        # Assign attributes (fallback defaults)
        hair_color = colors_list[idx] if idx < len(colors_list) else "#000000"
        hair_type_value = types_list[idx] if idx < len(types_list) else "Unknown"
        hair_density_value = density_list[idx] if idx < len(density_list) else 0.5
        hair_line_value = line_types_list[idx] if idx < len(line_types_list) else "Hairline"

        data = await image.read()
        months = "3" if timeframe == "3months" else "8"
        color_instruction = f" Use hair color: {hair_color}" if hair_color != "#000000" else ""
        
        # Determine if this is FreeMark or Hairline mode
        is_freemark = hair_line_value == "FreeMark"
        
        # Create different prompts based on the mode
        if is_freemark:
            # FreeMark mode - the image contains drawn markings showing target areas
            base_prompt = (
                f"Analyze the drawn markings/lines visible on this image. These markings indicate the exact areas where hair regrowth should occur. "
                f"Generate a realistic high-quality hair regrowth image by growing hair specifically in the marked/drawn areas shown in the image. "
                f"Use the visual markings as your primary reference for where to place new hair growth. "
                f"Ignore any text-based area specifications and focus entirely on the visual markings present in the image. "
            )
        else:
            # Hairline mode - traditional area-based targeting
            area_instruction = f"Focus area: {hair_line_value}. "
            base_prompt = (
                f"Generate a realistic high-quality hair regrowth image focusing on {hair_line_value.lower()} restoration. "
                f"{area_instruction}"
            )

        # Add time-specific instructions
        if months == "3":
            prompt = (
                base_prompt +
                f"Show natural hair regrowth after 3 months of treatment. "
                f"Hair type: {hair_type_value}, Hair density: {hair_density_value:.2f}. "
                f"Show subtle, early-stage improvement with small visible changes such as fine baby hairs, "
                f"light coverage in previously thin areas, and gentle signs of progress. "
                f"Ensure the person's facial features remain identical without distortion.{color_instruction} "
                f"The regrowth should look modest, natural, and realistic for a 3-month result."
            )
        else:  # 8 months
            prompt = (
                base_prompt +
                f"Show advanced hair regrowth after 8 months of treatment. "
                f"Hair type: {hair_type_value}, Hair density: {hair_density_value:.2f}. "
                f"Show significant visible improvement with thicker, denser, and fuller hair coverage, "
                f"clearly filling in previously bald or thinning regions. "
                f"Ensure the person's facial features remain identical without distortion.{color_instruction} "
                f"The regrowth should look dramatic but still natural, reflecting strong 8-month progress."
            )

        # Log which mode is being used
        print(f"Processing image {idx}: {'FreeMark' if is_freemark else 'Hairline'} mode")
        print(f"Hair line type: {hair_line_value}")

        # Build request to Google GenAI
        contents = [
            types.Part(inline_data=types.Blob(data=data, mime_type=image.content_type)),
            types.Part.from_text(text=prompt),
        ]
        config = types.GenerateContentConfig(response_modalities=["IMAGE"])

        stream = client.models.generate_content_stream(
            model=MODEL_NAME, contents=contents, config=config
        )
        image_data = _process_api_stream_response(stream)

        if not image_data:
            results.append({
                "index": idx,
                "filename": image.filename,
                "error": "Failed to generate image"
            })
            continue

        # Append successful result
        results.append({
            "index": idx,
            "filename": image.filename,
            "image": base64.b64encode(image_data).decode("utf-8"),
            "hair_color": hair_color,
            "hair_type": hair_type_value,
            "hair_density": hair_density_value,
            "hair_line_type": hair_line_value,
            "is_freemark": is_freemark,
            "timeframe": timeframe
        })

    return {
        "results": results,
        "timeframe": timeframe,
        "message": f"Generated {len([r for r in results if 'image' in r])} out of {len(images)} images"
    }