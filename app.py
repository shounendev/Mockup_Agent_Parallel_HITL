import os
import dotenv

dotenv.load_dotenv()

from kodosumi.core import ServeAPI

app = ServeAPI()

import jinja2
from kodosumi.core import forms as F
from kodosumi.core import Launch, Tracer
from kodosumi.response import Markdown
import fastapi
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from typing import TypedDict, List, Optional
from PIL import Image
import boto3
from io import BytesIO
from datetime import datetime
import uuid
import mimetypes
import base64
import json
from google import genai
from google.genai import types
import ray
import asyncio


# ============================================================================
# SIMPLIFIED FORM DEFINITION (Screenshot only - no parameter fields)
# ============================================================================

input_model = F.Model(
    F.Markdown("""
    # Parallel Mockup Generator with HITL

    Upload a website screenshot to generate 6 professional mockup variations.

    ## How it works:
    1. Upload screenshot
    2. AI analyzes and suggests parameters
    3. Review and edit parameters (Human-in-the-Loop)
    4. Generate 6 variations in parallel
    5. Download all mockups
    """),
    F.InputFiles(
        name="screenshot",
        label="Upload Screenshot",
    ),
    F.Submit("Analyze Screenshot"),
    F.Cancel("Cancel"),
)


# ============================================================================
# STATE SCHEMA
# ============================================================================


class MockupGraphState(TypedDict):
    tracer: Tracer
    screenshot_local_path: Optional[str]
    cropped_screenshot_path: Optional[str]

    # AI-detected parameters (before HITL)
    ai_device: Optional[str]
    ai_interior_style: Optional[str]
    ai_profession: Optional[str]
    ai_mood: Optional[str]
    ai_time_of_day: Optional[str]

    # Final parameters (after HITL)
    device: str
    interior_style: str
    profession: str
    mood: str
    time_of_day: str

    filled_prompt: str

    # Parallel generation results
    mockup_results: List[
        dict
    ]  # List of 6 Ray remote results (now with "file_path" instead of "data")
    uploaded_mockups: List[dict]  # List of uploaded mockup metadata
    upload_timestamp: Optional[str]
    temp_output_dir: Optional[str]  # Temporary directory for generated files

    errors: List[str]


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

prompt_template = jinja2.Template("""
Remove the browser UI and place the pasted website screenshot naturally on the screen of a [{{ device }}].

Use the screenshot to guide:
- Interior Style: [{{ interior_style }}]
- Profession of the Space Owner: [{{ profession }}]
- Mood: [{{ mood }}]
- Time of day: [{{ time_of_day }}]

Camera: high-end lifestyle brand film photography, long focal length, close-up framing with shallow depth of field, soft bokeh, slight chromatic aberration at the edges.

Intent: a cohesive, high-quality but approachable scene where the environment gently mirrors the screenshot's style without feeling luxury or overly curated.
The screenshot should be the brightest element in the scene, with natural lighting that enhances the colors and details of the website design.
""")


# ============================================================================
# RAY REMOTE FUNCTION FOR PARALLEL GENERATION
# ============================================================================


@ray.remote
def generate_single_mockup(
    prompt: str,
    cropped_screenshot_path: str,
    gemini_api_key: str,
    variation_index: int,
    output_dir: str,
) -> dict:
    """
    Ray remote function to generate a single mockup.
    Writes to disk immediately, returns dict with file path (NOT image data).
    """
    from google import genai
    from google.genai import types
    from PIL import Image
    import mimetypes
    import os
    from datetime import datetime
    import uuid
    import dotenv

    dotenv.load_dotenv()

    try:
        client = genai.Client(api_key=gemini_api_key)
        model = "gemini-3-pro-image-preview"

        with Image.open(cropped_screenshot_path) as cropped_screenshot:
            contents = [prompt, cropped_screenshot]

            tools = [types.Tool(googleSearch=types.GoogleSearch())]
            generate_content_config = types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
                image_config=types.ImageConfig(image_size="2K"),
                tools=tools,
            )

            mockup_data = None
            mockup_mime = None

            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates
                    and chunk.candidates[0].content
                    and chunk.candidates[0].content.parts
                    and chunk.candidates[0].content.parts[0].inline_data
                    and chunk.candidates[0].content.parts[0].inline_data.data
                ):
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    mockup_data = inline_data.data
                    mockup_mime = inline_data.mime_type

        if not mockup_data:
            return {
                "variation_index": variation_index,
                "success": False,
                "error": "No image data received from Gemini",
                "file_path": None,
                "mime": None,
            }

        file_extension = mimetypes.guess_extension(mockup_mime)
        mockup_format = file_extension.lstrip(".") if file_extension else "jpeg"

        # Write to disk immediately - do NOT hold in memory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_id = str(uuid.uuid4())[:8]
        filename = (
            f"mockup_var{variation_index}_{timestamp}_{unique_id}.{mockup_format}"
        )
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "wb") as f:
            f.write(mockup_data)

        # Get file size from disk
        file_size = os.path.getsize(file_path)

        return {
            "variation_index": variation_index,
            "success": True,
            "error": None,
            "file_path": file_path,  # Return path, NOT data
            "mime": mockup_mime,
            "format": mockup_format,
            "size_bytes": file_size,
        }

    except Exception as e:
        return {
            "variation_index": variation_index,
            "success": False,
            "error": str(e),
            "file_path": None,
            "mime": None,
        }


# ============================================================================
# PROMPT VARIATIONS FUNCTION
# ============================================================================


def generate_prompt_variations(
    base_params: dict, template: jinja2.Template
) -> List[str]:
    """Generate 6 slightly different prompts based on base parameters"""

    # Variation strategies
    variations = [
        {
            # Base prompt - exactly as specified
            "device": base_params["device"],
            "interior_style": base_params["interior_style"],
            "profession": base_params["profession"],
            "mood": base_params["mood"],
            "time_of_day": base_params["time_of_day"],
        },
        {
            # Slightly warmer mood
            "device": base_params["device"],
            "interior_style": base_params["interior_style"] + ", warm tones",
            "profession": base_params["profession"],
            "mood": base_params["mood"] + ", welcoming",
            "time_of_day": base_params["time_of_day"],
        },
        {
            # Slightly cooler mood
            "device": base_params["device"],
            "interior_style": base_params["interior_style"] + ", cool tones",
            "profession": base_params["profession"],
            "mood": base_params["mood"] + ", professional",
            "time_of_day": base_params["time_of_day"],
        },
        {
            # Different camera angle suggestion
            "device": base_params["device"],
            "interior_style": base_params["interior_style"],
            "profession": base_params["profession"],
            "mood": base_params["mood"] + ", slightly angled perspective",
            "time_of_day": base_params["time_of_day"],
        },
        {
            # Enhanced depth of field
            "device": base_params["device"],
            "interior_style": base_params["interior_style"] + ", enhanced bokeh",
            "profession": base_params["profession"],
            "mood": base_params["mood"] + ", cinematic depth",
            "time_of_day": base_params["time_of_day"],
        },
        {
            # Natural lighting emphasis
            "device": base_params["device"],
            "interior_style": base_params["interior_style"] + ", natural window light",
            "profession": base_params["profession"],
            "mood": base_params["mood"] + ", soft ambient",
            "time_of_day": base_params["time_of_day"],
        },
    ]

    prompts = []
    for variation in variations:
        prompt = template.render(**variation)
        prompts.append(prompt)

    return prompts


# ============================================================================
# ANALYSIS NODES (Before HITL)
# ============================================================================


async def validate_and_load_screenshot(state: MockupGraphState) -> MockupGraphState:
    """Node 1: Validate uploaded file and load it into state"""
    tracer = state["tracer"]

    try:
        async with await tracer.fs() as fs:
            files = await fs.ls()

            if not files:
                error = "No screenshot uploaded"
                await tracer.markdown(f"❌ {error}")
                state["errors"].append(error)
                return state

            file_info = files[0]
            file_path = file_info["path"]
            file_name = file_info.get("name", "screenshot.png")

            await tracer.markdown(f"📁 Processing: `{file_name}`")

            # Download file
            local_path = None
            async for downloaded_path in fs.download(file_path):
                local_path = downloaded_path
                break

            if not local_path:
                error = f"Failed to download {file_name}"
                await tracer.markdown(f"❌ {error}")
                state["errors"].append(error)
                return state

            # Validate image
            with Image.open(local_path) as img:
                img_format = img.format
                width, height = img.size

            await tracer.markdown(
                f"✅ Screenshot uploaded: {width}x{height}px, format: {img_format}"
            )

            state["screenshot_local_path"] = local_path
            return state

    except Exception as e:
        error = f"File validation error: {str(e)}"
        await tracer.markdown(f"❌ {error}")
        state["errors"].append(error)
        return state


async def crop_to_16_9(state: MockupGraphState) -> MockupGraphState:
    """Node 2: Crop screenshot to 16:9 aspect ratio"""
    tracer = state["tracer"]
    screenshot_path = state["screenshot_local_path"]

    if not screenshot_path:
        return state

    try:
        with Image.open(screenshot_path) as img:
            original_width, original_height = img.size

            # Calculate 16:9 aspect ratio
            target_aspect = 16 / 9
            current_aspect = original_width / original_height

            if abs(current_aspect - target_aspect) < 0.01:
                await tracer.markdown(
                    f"✅ Image is already 16:9 ({original_width}x{original_height})"
                )
                state["cropped_screenshot_path"] = screenshot_path
                return state

            await tracer.markdown("🔲 Cropping to 16:9 aspect ratio...")

            # Determine crop dimensions
            if current_aspect > target_aspect:
                new_width = int(original_height * target_aspect)
                new_height = original_height
                left = (original_width - new_width) // 2
                top = 0
            else:
                new_width = original_width
                new_height = int(original_width / target_aspect)
                left = 0
                top = (original_height - new_height) // 2

            right = left + new_width
            bottom = top + new_height

            cropped_img = img.crop((left, top, right, bottom))

            # Save cropped image
            temp_dir = os.path.join("data", "temp")
            os.makedirs(temp_dir, exist_ok=True)
            cropped_path = os.path.join(
                temp_dir, f"cropped_{os.path.basename(screenshot_path)}"
            )
            cropped_img.save(cropped_path)
            cropped_img.close()  # Explicitly close cropped image after saving

        await tracer.markdown(
            f"✅ Cropped from {original_width}x{original_height} to {new_width}x{new_height}"
        )

        state["cropped_screenshot_path"] = cropped_path
        return state

    except Exception as e:
        error = f"Cropping error: {str(e)}"
        await tracer.markdown(f"❌ {error}")
        state["errors"].append(error)
        return state


async def query_openai_for_parameters(state: MockupGraphState) -> MockupGraphState:
    """Node 3: Use GPT-4o Vision to analyze screenshot"""
    tracer = state["tracer"]

    try:
        await tracer.markdown("🔍 Analyzing screenshot with AI vision...")

        llm = ChatOpenAI(
            model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7
        )

        screenshot_path = state["cropped_screenshot_path"]

        # Detect image format
        with Image.open(screenshot_path) as img:
            img_format = img.format  # e.g., 'JPEG', 'PNG', 'WEBP'

        # Map format to MIME type
        format_to_mime = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }
        mime_type = format_to_mime.get(img_format, "image/jpeg")  # Default to jpeg

        await tracer.markdown(f"**Debug - Image format:** {img_format}, MIME type: {mime_type}")

        with open(screenshot_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            image_url = f"data:{mime_type};base64,{image_data}"

        prompt_text = """Analyze this website screenshot and provide parameters for generating a realistic mockup scene.

Based on the screenshot's visual aesthetic, color scheme, design style, and overall vibe, suggest appropriate parameters.

Respond ONLY with valid JSON:
{
  "device": "high-end display device name",
  "interior_style": "room aesthetic description",
  "profession": "likely profession using this",
  "mood": "atmosphere description",
  "time_of_day": "morning/afternoon/evening/night"
}"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )

        response = await llm.ainvoke([message])
        content = response.content

        # Debug: log the raw response
        await tracer.markdown(f"**Debug - Raw API response:** {content[:200]}")

        # Parse JSON
        start = content.find("{")
        end = content.rfind("}") + 1

        if start == -1 or end == 0:
            raise ValueError(f"No JSON found in response. Content: {content}")

        json_str = content[start:end]
        params = json.loads(json_str)

        state["ai_device"] = params.get("device", "Apple Pro Display XDR")
        state["ai_interior_style"] = params.get("interior_style", "modern, minimalist")
        state["ai_profession"] = params.get("profession", "UX designer")
        state["ai_mood"] = params.get("mood", "creative, focused")
        state["ai_time_of_day"] = params.get("time_of_day", "morning")

        await tracer.markdown("✅ Analysis complete")

        return state

    except Exception as e:
        await tracer.markdown(f"⚠️ AI analysis failed: {str(e)}")
        await tracer.markdown("Using default parameters")

        # Fallback defaults
        state["ai_device"] = "Apple Pro Display XDR"
        state["ai_interior_style"] = "modern, minimalist"
        state["ai_profession"] = "software developer"
        state["ai_mood"] = "focused, calm"
        state["ai_time_of_day"] = "morning"

        return state


async def hitl_parameter_validation_node(state: MockupGraphState) -> MockupGraphState:
    """Node 4: HITL Parameter Validation and Confirmation

    Displays AI-detected parameters, pauses for human feedback,
    and merges user input with AI suggestions.
    """
    tracer = state["tracer"]

    # Check if analysis succeeded
    if not state["cropped_screenshot_path"]:
        error = "HITL skipped - screenshot analysis failed"
        await tracer.markdown(f"❌ {error}")
        state["errors"].append(error)
        return state

    # Display AI-detected parameters
    await tracer.markdown("## Phase 2: Human-in-the-Loop Parameter Confirmation")
    await tracer.markdown("📊 **AI-Detected Parameters:**")
    await tracer.markdown(f"- **Device:** {state['ai_device']}")
    await tracer.markdown(f"- **Interior Style:** {state['ai_interior_style']}")
    await tracer.markdown(f"- **Profession:** {state['ai_profession']}")
    await tracer.markdown(f"- **Mood:** {state['ai_mood']}")
    await tracer.markdown(f"- **Time of Day:** {state['ai_time_of_day']}")
    await tracer.markdown("")
    await tracer.markdown("👉 Please review and edit the parameters below...")

    # Pause for user input
    feedback = await tracer.lock(
        "parameter_confirmation",
        {
            "ai_device": state["ai_device"],
            "ai_interior_style": state["ai_interior_style"],
            "ai_profession": state["ai_profession"],
            "ai_mood": state["ai_mood"],
            "ai_time_of_day": state["ai_time_of_day"],
        },
    )

    # Merge feedback with AI suggestions
    state["device"] = feedback.get("device", state["ai_device"])
    state["interior_style"] = feedback.get("interior_style", state["ai_interior_style"])
    state["profession"] = feedback.get("profession", state["ai_profession"])
    state["mood"] = feedback.get("mood", state["ai_mood"])
    state["time_of_day"] = feedback.get("time_of_day", state["ai_time_of_day"])

    await tracer.markdown("## Phase 3: Parallel Generation")
    await tracer.markdown("✅ Parameters confirmed, starting parallel generation...")

    return state


# ============================================================================
# GENERATION NODES (After HITL)
# ============================================================================


async def parallel_mockup_generation_node(state: MockupGraphState) -> MockupGraphState:
    """Generate 6 mockups in parallel using Ray"""
    tracer = state["tracer"]

    await tracer.markdown("## Generating 6 Mockup Variations in Parallel")
    await tracer.markdown("⏳ This may take 30-90 seconds...")

    # Generate 6 prompt variations
    base_params = {
        "device": state["device"],
        "interior_style": state["interior_style"],
        "profession": state["profession"],
        "mood": state["mood"],
        "time_of_day": state["time_of_day"],
    }

    prompts = generate_prompt_variations(base_params, prompt_template)

    # Launch 6 Ray remote tasks
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    cropped_path = state["cropped_screenshot_path"]

    # Create temporary output directory for generated images
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_output_dir = os.path.join("output", f"temp_mockups_{timestamp}")
    os.makedirs(temp_output_dir, exist_ok=True)

    futures = [
        generate_single_mockup.remote(
            prompt=prompts[i],
            cropped_screenshot_path=cropped_path,
            gemini_api_key=gemini_api_key,
            variation_index=i,
            output_dir=temp_output_dir,  # Pass output directory to write files
        )
        for i in range(6)
    ]

    # Wait for results with progress tracking
    unready = futures.copy()
    completed_count = 0
    results = []

    while unready:
        ready, unready = ray.wait(unready, num_returns=1, timeout=1)
        if ready:
            result = ray.get(ready[0])
            results.append(result)
            completed_count += 1

            await tracer.markdown(f"### ✅ Generated mockup {completed_count}/6")

            if not result["success"]:
                await tracer.markdown(
                    f"⚠️ Variation {result['variation_index']} failed: {result['error']}"
                )
            else:
                await tracer.markdown(
                    f"📊 Size: {result['size_bytes'] / 1024 / 1024:.2f} MB"
                )

        await asyncio.sleep(0.1)

    # Sort results by variation_index
    results.sort(key=lambda x: x["variation_index"])

    # Store in state
    state["mockup_results"] = results
    state["temp_output_dir"] = temp_output_dir  # Track temp directory for cleanup

    successful_count = sum(1 for r in results if r["success"])
    await tracer.markdown(f"## 🎉 Generation Complete: {successful_count}/6 successful")

    return state


async def upload_all_to_spaces_node(state: MockupGraphState) -> MockupGraphState:
    """Upload all successful mockups to Digital Ocean Spaces"""
    tracer = state["tracer"]

    results = state.get("mockup_results", [])
    successful_results = [r for r in results if r["success"]]

    if not successful_results:
        error = "No successful mockups to upload"
        await tracer.markdown(f"❌ {error}")
        state["errors"].append(error)
        return state

    await tracer.markdown(
        f"## Uploading {len(successful_results)} Mockups to Digital Ocean Spaces"
    )

    # Validate environment variables
    spaces_endpoint = os.getenv("SPACES_ENDPOINT")
    spaces_key = os.getenv("SPACES_KEY")
    spaces_secret = os.getenv("SPACES_SECRET")
    bucket_name = os.getenv("SPACES_BUCKET")
    spaces_region = os.getenv("SPACES_REGION", "fra1")

    if not all([spaces_endpoint, spaces_key, spaces_secret, bucket_name]):
        missing = []
        if not spaces_endpoint:
            missing.append("SPACES_ENDPOINT")
        if not spaces_key:
            missing.append("SPACES_KEY")
        if not spaces_secret:
            missing.append("SPACES_SECRET")
        if not bucket_name:
            missing.append("SPACES_BUCKET")

        error = f"Missing environment variables: {', '.join(missing)}"
        await tracer.markdown(f"❌ {error}")
        state["errors"].append(error)
        return state

    # Initialize S3 client
    s3_client = boto3.client(
        "s3",
        endpoint_url=spaces_endpoint,
        aws_access_key_id=spaces_key,
        aws_secret_access_key=spaces_secret,
        region_name=spaces_region,
    )

    # Upload each mockup
    uploaded_mockups = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for result in successful_results:
        variation_idx = result["variation_index"]
        file_path = result["file_path"]  # Get file path instead of data
        img_format = result["format"]

        await tracer.markdown(f"☁️ Uploading variation {variation_idx + 1}...")

        try:
            # Generate unique filename
            unique_id = str(uuid.uuid4())[:8]
            extension = f".{img_format}"
            s3_key = f"mockups/batch_{timestamp}/mockup_var{variation_idx}_{unique_id}{extension}"

            # Upload directly from disk file
            with open(file_path, "rb") as f:
                s3_client.upload_fileobj(
                    f,
                    bucket_name,
                    s3_key,
                    ExtraArgs={
                        "ACL": "public-read",
                        "ContentType": f"image/{img_format}",
                        "CacheControl": "max-age=31536000",
                    },
                )

            # Construct public URL
            public_url = f"{spaces_endpoint}/{bucket_name}/{s3_key}"

            uploaded_mockups.append(
                {
                    "variation_index": variation_idx,
                    "public_url": public_url,
                    "s3_key": s3_key,
                    "file_size_bytes": result["size_bytes"],
                    "format": img_format,
                }
            )

            await tracer.markdown(f"✅ Uploaded variation {variation_idx + 1}")

            # Delete temp file after successful upload
            try:
                os.remove(file_path)
            except Exception as del_err:
                await tracer.markdown(
                    f"⚠️ Could not delete temp file {file_path}: {del_err}"
                )

        except Exception as e:
            await tracer.markdown(
                f"⚠️ Upload failed for variation {variation_idx}: {str(e)}"
            )

    state["uploaded_mockups"] = uploaded_mockups
    state["upload_timestamp"] = timestamp

    await tracer.markdown(
        f"## 🎉 Upload Complete: {len(uploaded_mockups)} mockups available"
    )

    return state


async def cleanup_temp_files(state: MockupGraphState) -> MockupGraphState:
    """Clean up temporary generated files after upload"""
    tracer = state["tracer"]

    temp_dir = state.get("temp_output_dir")
    if temp_dir and os.path.exists(temp_dir):
        try:
            import shutil

            shutil.rmtree(temp_dir)
            await tracer.markdown(f"🧹 Cleaned up temporary files from {temp_dir}")
        except Exception as e:
            await tracer.markdown(f"⚠️ Cleanup warning: {str(e)}")

    return state


# ============================================================================
# UNIFIED LANGGRAPH WORKFLOW
# ============================================================================

# Single unified graph (Analysis → HITL → Generation)
workflow = StateGraph(MockupGraphState)

# Add all 7 nodes
workflow.add_node("validate_screenshot", validate_and_load_screenshot)
workflow.add_node("crop_to_16_9", crop_to_16_9)
workflow.add_node("query_openai", query_openai_for_parameters)
workflow.add_node("hitl_validation", hitl_parameter_validation_node)
workflow.add_node("parallel_generation", parallel_mockup_generation_node)
workflow.add_node("upload_all", upload_all_to_spaces_node)
workflow.add_node("cleanup", cleanup_temp_files)

# Define sequential edges
workflow.add_edge(START, "validate_screenshot")
workflow.add_edge("validate_screenshot", "crop_to_16_9")
workflow.add_edge("crop_to_16_9", "query_openai")
workflow.add_edge("query_openai", "hitl_validation")
workflow.add_edge("hitl_validation", "parallel_generation")
workflow.add_edge("parallel_generation", "upload_all")
workflow.add_edge("upload_all", "cleanup")
workflow.add_edge("cleanup", END)

# Compile single graph
unified_graph = workflow.compile()


# ============================================================================
# HITL FEEDBACK FORM
# ============================================================================


@app.lock(name="parameter_confirmation")
async def collect_parameter_feedback(data: dict):
    """HITL form for parameter confirmation/editing

    Receives data from tracer.lock() call with AI-detected parameters.
    """

    # Pre-fill with AI-detected values from data dict (passed from tracer.lock())
    ai_device = data.get("ai_device", "Apple Pro Display XDR")
    ai_interior_style = data.get("ai_interior_style", "modern, minimalist")
    ai_profession = data.get("ai_profession", "UX designer")
    ai_mood = data.get("ai_mood", "focused, creative")
    ai_time_of_day = data.get("ai_time_of_day", "morning")

    feedback_form = F.Model(
        F.Markdown("""
        ## Review AI-Detected Parameters

        The AI has analyzed your screenshot. Review and edit the parameters below, then submit to generate 6 mockup variations.
        """),
        F.InputText(
            name="device",
            label="Display Device",
            value=ai_device,
            placeholder="e.g., Apple Pro Display XDR",
        ),
        F.InputText(
            name="interior_style",
            label="Interior Style",
            value=ai_interior_style,
            placeholder="e.g., minimalist modern, cozy creative",
        ),
        F.InputText(
            name="profession",
            label="Space Owner Profession",
            value=ai_profession,
            placeholder="e.g., UX designer, software developer",
        ),
        F.InputText(
            name="mood",
            label="Atmosphere/Mood",
            value=ai_mood,
            placeholder="e.g., bright and inspiring, focused and calm",
        ),
        F.InputText(
            name="time_of_day",
            label="Time of Day (Base)",
            value=ai_time_of_day,
            placeholder="morning, afternoon, evening, night",
        ),
        F.Markdown(
            "**Note:** 6 variations will be generated with slight modifications to these base parameters."
        ),
        F.Submit("Generate 6 Mockups"),
        F.Cancel("Cancel"),
    )

    return feedback_form


@app.lease(name="parameter_confirmation")
async def process_parameter_feedback(inputs: dict):
    """Process submitted HITL form data and return to tracer.lock() caller

    Receives form data submitted by the user and returns it to the
    hitl_parameter_validation_node which will merge with AI suggestions.
    """
    return {
        "device": inputs.get("device", ""),
        "interior_style": inputs.get("interior_style", ""),
        "profession": inputs.get("profession", ""),
        "mood": inputs.get("mood", ""),
        "time_of_day": inputs.get("time_of_day", ""),
    }


# ============================================================================
# GALLERY MARKDOWN BUILDER
# ============================================================================


def build_gallery_markdown(result: dict) -> Markdown:
    """Build markdown gallery showing all 6 mockups"""

    md = ["## 🎉 6 Mockup Variations Generated!", ""]

    uploaded_mockups = result.get("uploaded_mockups", [])

    if not uploaded_mockups:
        md.append("No mockups were successfully generated.")
        return Markdown("\n".join(md))

    md.append(
        f"**{len(uploaded_mockups)} variations** generated and uploaded successfully."
    )
    md.append("")

    # Parameters used
    md.append("### Parameters Used")
    md.append("")
    md.append("| Parameter | Value |")
    md.append("|---|---|")
    md.append(f"| Device | {result['device']} |")
    md.append(f"| Interior Style | {result['interior_style']} |")
    md.append(f"| Profession | {result['profession']} |")
    md.append(f"| Mood | {result['mood']} |")
    md.append(f"| Time of Day | {result['time_of_day']} |")
    md.append("")

    # Gallery of mockups
    md.append("### Generated Mockups")
    md.append("")

    for mockup in sorted(uploaded_mockups, key=lambda x: x["variation_index"]):
        idx = mockup["variation_index"]
        url = mockup["public_url"]
        size_mb = mockup["file_size_bytes"] / 1024 / 1024

        md.append(f"#### Variation {idx + 1}")
        md.append("")
        md.append(f"![Mockup Variation {idx + 1}]({url})")
        md.append("")
        md.append(f"**Size:** {size_mb:.2f} MB")
        md.append("")
        md.append(f"[Download]({url})")
        md.append("")
        md.append(f"`{url}`")
        md.append("")
        md.append("---")
        md.append("")

    return Markdown("\n".join(md))


# ============================================================================
# RUNNER FUNCTION WITH HITL
# ============================================================================


async def runner(inputs: dict, tracer: Tracer):
    """Main runner with unified graph"""

    await tracer.markdown("# Parallel Mockup Generator with HITL")

    # Initialize state
    initial_state = {
        "tracer": tracer,
        "screenshot_local_path": None,
        "cropped_screenshot_path": None,
        "ai_device": "",
        "ai_interior_style": "",
        "ai_profession": "",
        "ai_mood": "",
        "ai_time_of_day": "",
        "device": "",
        "interior_style": "",
        "profession": "",
        "mood": "",
        "time_of_day": "",
        "filled_prompt": "",
        "mockup_results": [],
        "uploaded_mockups": [],
        "upload_timestamp": None,
        "temp_output_dir": None,
        "errors": [],
    }

    # Phase 1: Analysis
    await tracer.markdown("## Phase 1: Analysis")

    # Single graph execution (handles analysis + HITL + generation)
    result = await unified_graph.ainvoke(initial_state)

    # Check for errors
    if result["errors"]:
        md = ["## ❌ Process Failed", ""]
        for error in result["errors"]:
            md.append(f"- {error}")
        return Markdown("\n".join(md))

    # Build markdown response with all 6 mockups
    return build_gallery_markdown(result)


# ============================================================================
# FASTAPI ENDPOINT
# ============================================================================


@app.enter(
    path="/",
    model=input_model,
    summary="Parallel Mockup Generator with HITL",
    description="Generate 6 website mockup variations with human-in-the-loop parameter confirmation.",
    tags=["Mockup", "AI", "HITL", "Parallel"],
    version="1.0.0",
    author="peter.jaschkowske@nmkr.io",
    organization="NMKR",
)
async def enter(request: fastapi.Request, inputs: dict):
    return Launch(request, "Mockup_Agent_Parallel_HITL.app:runner", inputs=inputs)


# ============================================================================
# RAY SERVE DEPLOYMENT
# ============================================================================

from ray import serve


@serve.deployment
@serve.ingress(app)
class ParallelMockupAgent:
    pass


fast_app = ParallelMockupAgent.bind()


# ============================================================================
# DEV SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8014, reload=True)
