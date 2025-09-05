from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64, os, json
import google.genai as genai
from google.genai import types
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
MODEL_NAME = "gemini-2.5-flash-image-preview"

app = FastAPI(title="Hair Growth Simulation API")

# Allow all CORS (safe for dev, restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Backend running"}

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

def _process_text_response(resp) -> str:
    """Extracts plain text from Gemini analysis response"""
    for chunk in resp:
        if not chunk.candidates:
            continue
        cand = chunk.candidates[0]
        if cand.content and cand.content.parts:
            for part in cand.content.parts:
                if part.text:
                    return part.text
    return ""

# ---------------------------
# Stage 1: Image Analysis
# ---------------------------
async def analyze_image(client, data: bytes, mime_type: str, is_freemark: bool):
    """Ask Gemini to analyze scalp, density, marked regions, etc."""
    analysis_prompt = (
        "Analyze this scalp image. Return a structured description in JSON with fields: "
        "detectedAreas (frontal, crown, temples, etc.), scalpVisibility (low/medium/high), "
        "existingHairDensity (0.0 to 1.0), freemarkDetected (true/false), notes (text). "
    )
    if is_freemark:
        analysis_prompt += (
            "If markings/drawn lines are visible, set freemarkDetected to true and mention in notes "
            "which regions are marked for regrowth."
        )

    print("\n[ANALYZE] Starting scalp analysis...")
    print(f"[ANALYZE] is_freemark={is_freemark}")
    print(f"[ANALYZE] Prompt => {analysis_prompt}")

    contents = [
        types.Part(inline_data=types.Blob(data=data, mime_type=mime_type)),
        types.Part.from_text(text=analysis_prompt),
    ]
    config = types.GenerateContentConfig(response_modalities=["TEXT"])

    resp = client.models.generate_content_stream(
        model=MODEL_NAME, contents=contents, config=config
    )
    raw_text = _process_text_response(resp)

    print(f"[ANALYZE] Raw response => {raw_text}")

    try:
        analysis = json.loads(raw_text)
        print(f"[ANALYZE] Parsed JSON => {analysis}")
    except Exception:
        analysis = {"raw": raw_text}
        print("[ANALYZE] Fallback: Could not parse JSON, storing raw text.")

    return analysis

# ---------------------------
# Stage 2: Hair Growth Generation
# ---------------------------
async def generate_image(client, data: bytes, mime_type: str, analysis: dict,
                         hair_type_value: str, hair_density_value: float,
                         hair_color: str, hair_line_value: str,
                         timeframe: str, is_freemark: bool):

    print("\n[GENERATE] Preparing image generation...")
    print(f"[GENERATE] timeframe={timeframe}, months={'3' if timeframe == '3months' else '8'}")
    print(f"[GENERATE] is_freemark={is_freemark}, hair_line_value={hair_line_value}")
    print(f"[GENERATE] hair_type={hair_type_value}, hair_density={hair_density_value}, hair_color={hair_color}")
    print(f"[GENERATE] Analysis context => {analysis}")

    months = "3" if timeframe == "3months" else "8"
    color_instruction = f" Use hair color: {hair_color}" if hair_color != "#000000" else ""

    # Base prompt
    if is_freemark:
        base_prompt = (
            "Use the drawn markings/lines visible in the uploaded image as the ONLY areas "
            "for new hair regrowth. Ignore text-based areas. Follow exactly where markings "
            "are drawn. "
        )
    else:
        base_prompt = (
            f"Focus hair regrowth on {hair_line_value.lower()} region as indicated. "
        )

    base_prompt += f"Analysis notes: {analysis}. "

    if months == "3":
        prompt = (
            base_prompt +
            (
                f"Show realistic hair regrowth after 3 months of treatment. "
                f"Hair should appear significantly denser, thicker, and fuller compared to the original, "
                f"Some scalp visibility may remain in certain areas, with early regrowth and baby hairs starting to cover. "
                f"Hair type: {hair_type_value}, Hair density target: {hair_density_value:.2f}. "
                "Ensure the face remains identical without distortion."
                f"{color_instruction}"
            )
        )
    else:
        prompt = (
            base_prompt +
        (
            f"Show realistic hair regrowth after 8 months of treatment. "
            f"Hair should appear significantly denser, thicker, and fuller. "
            f"Scalp should be much less visible with natural coverage. "
            f"Hair type: {hair_type_value}, Hair density target: {hair_density_value:.2f}. "
            "Ensure the face remains identical without distortion."
            f"{color_instruction}"
        )
    )


    print("\n========= FINAL PROMPT SENT =========")
    print(prompt)
    print("=====================================\n")

    contents = [
        types.Part(inline_data=types.Blob(data=data, mime_type=mime_type)),
        types.Part.from_text(text=prompt),
    ]
    config = types.GenerateContentConfig(response_modalities=["IMAGE"])

    stream = client.models.generate_content_stream(
        model=MODEL_NAME, contents=contents, config=config
    )

    print("[GENERATE] Streaming image response...")
    image_data = _process_api_stream_response(stream)

    if image_data:
        print("[GENERATE] Image successfully generated (binary data length:", len(image_data), ")")
    else:
        print("[GENERATE] ❌ Failed to generate image.")

    return image_data


# ---------------------------
# Batch Route
# ---------------------------
@app.post("/generate/batch")
async def generate_batch_images(
    images: List[UploadFile] = File(...),
    timeframe: str = Form(...),
    hair_colors: Optional[str] = Form(None),
    hair_types: Optional[str] = Form(None),
    hair_density: Optional[str] = Form(None),
    hair_line_types: Optional[str] = Form(None)
):
    print("\n=== [BATCH] Request received ===")
    print(f"Number of images: {len(images)}, timeframe: {timeframe}")
    print(f"hair_colors={hair_colors}, hair_types={hair_types}, hair_density={hair_density}, hair_line_types={hair_line_types}")

    client = get_client()

    # Parse arrays
    colors_list = json.loads(hair_colors) if hair_colors else []
    types_list = json.loads(hair_types) if hair_types else []
    density_list = json.loads(hair_density) if hair_density else []
    line_types_list = json.loads(hair_line_types) if hair_line_types else []

    results = []

    for idx, image in enumerate(images):
        print(f"\n--- [IMAGE {idx}] Processing file: {image.filename} ---")

        if not image.content_type or not image.content_type.startswith("image/"):
            print(f"[IMAGE {idx}] ❌ Invalid file type: {image.content_type}")
            results.append({
                "index": idx,
                "filename": image.filename,
                "error": "File must be an image"
            })
            continue

        hair_color = colors_list[idx] if idx < len(colors_list) else "#000000"
        hair_type_value = types_list[idx] if idx < len(types_list) else "Unknown"
        hair_density_value = density_list[idx] if idx < len(density_list) else 0.5
        hair_line_value = line_types_list[idx] if idx < len(line_types_list) else "Hairline"
        is_freemark = hair_line_value == "FreeMark"

        print(f"[IMAGE {idx}] Params: color={hair_color}, type={hair_type_value}, density={hair_density_value}, line={hair_line_value}, freemark={is_freemark}")

        data = await image.read()

        # Stage 1: Analyze
        analysis = await analyze_image(client, data, image.content_type, is_freemark)

        # Stage 2: Generate
        image_data = await generate_image(
            client, data, image.content_type, analysis,
            hair_type_value, hair_density_value,
            hair_color, hair_line_value, timeframe, is_freemark
        )

        if not image_data:
            print(f"[IMAGE {idx}] ❌ Generation failed.")
            results.append({
                "index": idx,
                "filename": image.filename,
                "analysis": analysis,
                "error": "Failed to generate image"
            })
            continue

        print(f"[IMAGE {idx}] ✅ Completed successfully.")

        results.append({
            "index": idx,
            "filename": image.filename,
            "analysis": analysis,
            "image": base64.b64encode(image_data).decode("utf-8"),
            "hair_color": hair_color,
            "hair_type": hair_type_value,
            "hair_density": hair_density_value,
            "hair_line_type": hair_line_value,
            "is_freemark": is_freemark,
            "timeframe": timeframe
        })

    print("\n=== [BATCH] Finished processing ===")
    return {
        "results": results,
        "timeframe": timeframe,
        "message": f"Generated {len([r for r in results if 'image' in r])} out of {len(images)} images"
    }
