# Parallel Mockup Generator with Human-in-the-Loop

A Kodosumi web application that generates 6 professional mockup variations in parallel using Ray remote functions, with human-in-the-loop parameter confirmation after AI analysis.

## Features

- **Screenshot Upload**: Simple upload interface - just provide your screenshot
- **AI Analysis**: GPT-4o Vision automatically analyzes screenshot aesthetic
- **Human-in-the-Loop**: Review and edit AI-detected parameters before generation
- **Parallel Generation**: Generate 6 mockup variations simultaneously using Ray
- **Prompt Variations**: Automatically creates 6 variations with subtle differences:
  - Base (exact parameters)
  - Warm tones emphasis
  - Cool tones emphasis
  - Angled perspective
  - Enhanced bokeh/depth
  - Natural lighting emphasis
- **Real-Time Progress**: Watch generation progress (1/6, 2/6, 3/6, etc.)
- **Cloud Storage**: All mockups uploaded to Digital Ocean Spaces
- **Gallery View**: See all 6 variations in a responsive grid

## Workflow

1. **Upload** - User uploads website screenshot
2. **Analysis** - Image cropped to 16:9, GPT-4o Vision analyzes aesthetic
3. **HITL** - User reviews and optionally edits AI-detected parameters
4. **Generation** - 6 mockups generated in parallel (30-90 seconds)
5. **Upload** - All mockups uploaded to Digital Ocean Spaces
6. **Gallery** - Display all 6 mockups with download links

## Prerequisites

- Python 3.10+
- OpenAI API key (GPT-4o Vision)
- Google Gemini API key (Gemini 3 Pro)
- Digital Ocean Spaces credentials

## Installation

### 1. Navigate to project directory

```bash
cd Mockup_Agent_Parallel_HITL
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.template` to `.env` and fill in credentials:

```bash
cp .env.template .env
```

Edit `.env`:

```bash
# OpenAI API (GPT-4o Vision)
OPENAI_API_KEY=sk-proj-xxxxx

# Google Gemini API (Gemini 3 Pro)
GEMINI_API_KEY=AIzaSyXxxxx

# Digital Ocean Spaces
SPACES_ENDPOINT=https://fra1.digitaloceanspaces.com
SPACES_REGION=fra1
SPACES_BUCKET=your-bucket-name
SPACES_KEY=YOUR_KEY
SPACES_SECRET=YOUR_SECRET
```

### 5. Create required directories

```bash
mkdir -p data/temp output
```

## Usage

### Development Server

Run the development server:

```bash
python app.py
```

Application available at: **http://localhost:8014**

### Production Deployment

```bash
serve run app:fast_app
```

## How It Works

### Two-Phase Workflow

**Phase 1: Analysis (Before HITL)**
- `validate_screenshot` - Load and validate uploaded file
- `crop_to_16_9` - Crop to 16:9 aspect ratio
- `query_openai` - GPT-4o Vision analyzes aesthetic

**Phase 2: Generation (After HITL)**
- `parallel_generation` - Generate 6 variations using Ray
- `upload_all` - Upload to Digital Ocean Spaces

### Human-in-the-Loop Pattern

The workflow pauses after AI analysis:

```python
# Show AI recommendations
await tracer.markdown("📊 **AI-Detected Parameters:**")
await tracer.markdown(f"- **Device:** {ai_device}")
# ... more parameters

# Pause for human feedback
feedback = await tracer.lock('parameter_confirmation', {
    'ai_device': analysis_result['ai_device'],
    # ... pass AI values to form
})

# Continue with user-confirmed parameters
final_device = feedback.get("device", ai_device)
```

### Ray Parallel Generation

Uses Ray remote functions to generate 6 mockups simultaneously:

```python
@ray.remote
def generate_single_mockup(prompt, screenshot_path, api_key, idx):
    # Generate single mockup
    # Returns dict with image data and metadata
    ...

# Launch 6 parallel tasks
futures = [generate_single_mockup.remote(...) for i in range(6)]

# Track progress
while unready:
    ready, unready = ray.wait(unready, num_returns=1)
    completed_count += 1
    await tracer.markdown(f"✅ Generated mockup {completed_count}/6")
```

### Prompt Variations

6 variations are automatically generated from base parameters:

1. **Base** - Exact user parameters
2. **Warm** - Adds warm tones, welcoming mood
3. **Cool** - Adds cool tones, professional mood
4. **Angled** - Slightly angled perspective
5. **Bokeh** - Enhanced depth of field, cinematic
6. **Natural** - Natural window light, soft ambient

## Architecture

### Key Differences from Base Project

| Aspect | Base Project | Parallel HITL Project |
|--------|-------------|----------------------|
| **Initial Form** | Screenshot + 5 parameter fields | Screenshot only |
| **Parameter Input** | Upfront (optional) | HITL after AI analysis |
| **HITL** | None | `tracer.lock()` for confirmation |
| **Generation** | Single mockup | 6 mockups in parallel |
| **Output** | Single image | Gallery of 6 images |
| **Workflows** | One LangGraph | Two (analysis + generation) |

### Technology Stack

- **Kodosumi**: Web framework for AI agents
- **LangGraph**: Workflow orchestration (2 separate graphs)
- **OpenAI GPT-4o**: Screenshot analysis with vision
- **Google Gemini 3 Pro**: Mockup image generation
- **Ray**: Distributed computing for parallel generation
- **FastAPI**: Web server
- **Pillow**: Image processing
- **boto3**: S3/Spaces integration

## Project Structure

```
Mockup_Agent_Parallel_HITL/
├── app.py                  # Main application
├── .env                    # Environment variables (not committed)
├── .env.template           # Environment template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   └── temp/              # Temporary cropped screenshots
└── output/                # Local output (optional)
```

## Troubleshooting

### "Missing environment variables"

Ensure all required variables are in `.env`:
- OPENAI_API_KEY
- GEMINI_API_KEY
- SPACES_ENDPOINT
- SPACES_KEY
- SPACES_SECRET
- SPACES_BUCKET

### "No image data received from Gemini"

- Check GEMINI_API_KEY is valid
- Verify Gemini API quota/credits
- Check Gemini API status

### "Upload failed"

- Verify Digital Ocean Spaces credentials
- Check bucket permissions (allow public-read)
- Verify bucket exists and is accessible

### Port 8014 already in use

Change port in `app.py`:

```python
uvicorn.run("app:app", host="0.0.0.0", port=8015, reload=True)
```

### Ray initialization errors

Make sure Ray is installed:

```bash
pip install ray==2.49.2
```

## API Endpoints

### Main Entry Point

- **Path**: `/`
- **Method**: POST
- **Description**: Upload screenshot and start workflow
- **Input**: Screenshot file
- **Response**: Redirects to workflow execution

### HITL Lock Point

- **Name**: `parameter_confirmation`
- **Type**: Human-in-the-Loop form
- **Purpose**: Review and edit AI-detected parameters
- **Pre-filled**: All 5 parameters from AI analysis
- **Editable**: Yes - all fields can be modified

## Development

### Testing Checklist

- [ ] Upload valid screenshot → AI analysis succeeds
- [ ] HITL form shows pre-filled AI parameters
- [ ] Edit parameters in HITL → changes applied
- [ ] Cancel in HITL → workflow stops gracefully
- [ ] 6 mockups generate in parallel
- [ ] Progress shows 1/6, 2/6, ..., 6/6
- [ ] Failed variations handled gracefully (e.g., 4/6 success)
- [ ] All successful mockups uploaded
- [ ] Gallery displays all mockups correctly
- [ ] Download links work
- [ ] Missing API keys → clear errors

### Adding More Variations

To generate more than 6 mockups, modify `generate_prompt_variations()` and update the range in `parallel_mockup_generation_node()`:

```python
# Change from range(6) to range(10) for 10 variations
futures = [
    generate_single_mockup.remote(...)
    for i in range(10)  # Change this number
]
```

## Performance

- **Analysis**: 5-10 seconds (GPT-4o Vision)
- **HITL**: User-dependent (manual review time)
- **Generation**: 30-90 seconds (6 mockups in parallel)
- **Upload**: 10-20 seconds (6 images to Spaces)
- **Total**: ~1-2 minutes (excluding HITL time)

## License

Based on:
- Mockup_Agent_Kodosumi
- Kodo_LangGraph_Template
- Kodosumi HITL examples

## Author

peter.jaschkowske@nmkr.io
NMKR Organization

## Version

1.0.0
