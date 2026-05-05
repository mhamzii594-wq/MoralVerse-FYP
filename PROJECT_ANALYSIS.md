# MoralVerse.AI - Complete Project Analysis

## 📋 Project Overview

**MoralVerse.AI** is an AI-powered interactive storytelling platform designed to create personalized moral education content for children (ages 3-12). It generates complete multimedia stories with:
- 4-scene moral narratives
- Ghibli-style illustrations with character consistency
- Bilingual voice narration (English & Urdu)
- Interactive branching decisions
- Final video assembly with subtitles

**Status:** Final Year Project (FYP) - Deadline: April 25th, 12 AM

---

## 🏗️ Architecture Overview

```
User Input (Web Form)
        ↓
[Django REST API Layer]
        ↓
[AI Modules Pipeline]
        ├─ LLM Engine (Story Generation)
        ├─ Image Engine (Scene Illustrations)
        ├─ Avatar Processor (Character Extraction)
        ├─ TTS Engine (Audio Generation)
        ├─ Decision Engine (Story Branching)
        ├─ Subtitle Engine (SRT Generation)
        └─ Video Engine (Final Assembly)
        ↓
[Database - Django ORM/SQLite]
        ↓
[Media Output]
    ├─ images/
    ├─ audio/
    ├─ videos/
    └─ subtitles/
```

---

## 🔄 Complete Data Flow & Pipeline

### 1️⃣ **Request Creation** (`views.py` - `create_story_request`)
**Input:**
- Child's name (required)
- Child's age (3-12, validated server-side)
- Moral theme (e.g., "Honesty", "Kindness")
- Optional avatar upload (JPG/PNG, max 5MB)
- Provider selections (LLM, Image, TTS)
- Language preference (English or Urdu)

**Processing:**
- Rate limiting: 3 stories/day per user (to protect API credits)
- Avatar handling:
  - If uploaded: Process with Ghibli-style conversion
  - If not: Create default generic avatar
  - Avatar gets described for character consistency
- Creates `StoryRequest` record in DB with status = "pending"

**Output:**
- Redirects to story preview page (`story_preview` view)

---

### 2️⃣ **Story Generation** (`llm_engine.py` - `generate_story`)

**LLM Providers Supported:**
- **Gemini 1.5 Flash** (default, fastest, free-tier friendly)
- **OpenAI** (GPT-4o-mini)
- **Groq** (Mixtral 8x7b-32768, low-cost)
- **Stub** (placeholder for testing)

**Character Consistency Strategy:**
- Uses character anchor string locked into EVERY image prompt
- Format: `"<child_name> ghibli-style <age>-year-old child large round expressive eyes colorful simple outfit"`
- Same exact string ensures Stable Diffusion generates consistent character across all 4 scenes

**Generated Story Structure:**
```json
{
  "title": "Story Title",
  "scenes": [
    {
      "id": 1,
      "text": "Scene narrative text",
      "image_prompt": "Detailed prompt for image generation",
      "decision": {
        "question": "What should character do?",
        "option_A": "Option A text",
        "option_B": "Option B text"
      }
    },
    // ... scenes 2-4
  ]
}
```

**Key Features:**
- Scenes 1-3 include moral choice points
- Scene 4 (finale) shows consequences of accumulated choices
- Age-appropriate complexity and narration speed
- Moral theme enforced throughout

---

### 3️⃣ **Avatar Processing** (`avatar_processor.py`)

**Avatar Handling Options:**
1. **User uploads photo** → Convert to Ghibli-style anime character
2. **Webcam capture** → Process the captured image
3. **No avatar** → Generate default child character

**Image Conversion Methods** (tried in order):
1. **Local Stable Diffusion API** (via Flask + Ngrok)
   - Fastest, no API costs
   - Requires: Local image server running (`imagemodel.ipynb`)
   - Endpoint: `LOCAL_IMAGE_API_URL` (from `.env`)

2. **Stability AI** (Fallback)
   - If local API unavailable
   - Uses `IMAGE_API_KEY`

3. **Default** (Last Resort)
   - Generate generic Ghibli-style child character

**Avatar Description:**
- Extracts visual characteristics from avatar image
- Description cached with avatar path
- Used in character anchor for image consistency

---

### 4️⃣ **Scene Image Generation** (`image_engine.py`)

**Image Providers Supported:**
- **Google Imagen** (via Gemini API) - Free, integrated with Gemini
- **OpenAI DALL-E** - High quality
- **Stability AI** - Cost-effective
- **Local Stable Diffusion** - No API costs, offline
- **Flux** (Future implementation)
- **Stub** (Placeholder)

**Character Consistency Mechanism:**
```
Scene 1 Image Generation:
  Prompt = character_anchor + scene_image_prompt + avatar_description
  img2img_reference = avatar_path
  Output: scene_1.png

Scene 2-4 Image Generation:
  Prompt = character_anchor + scene_image_prompt
  img2img_reference = previous_scene.png (for visual flow continuity)
  Output: scene_N.png
```

**How it maintains consistency:**
- Same character anchor string in every prompt
- Stable Diffusion tends to generate similar-looking character when anchor is identical
- img2img (image-to-image) guidance ensures scene flow visual coherence
- Optional avatar reference for extra consistency (Scene 1)

**Output:**
- Each scene image saved as PNG/JPG
- Path stored in `StoryScene.image_path`
- Reusable if regeneration occurs (speeds up process)

---

### 5️⃣ **Audio Generation** (`tts_engine.py`)

**TTS Providers Supported:**
- **Gemini TTS** - Free with Gemini API
- **OpenAI TTS** - High-quality, multiple voices
- **ElevenLabs** - Most natural-sounding
- **Coqui TTS** - Offline, CPU-based (fallback)
- **Stub** (Testing)

**Bilingual Support:**
- **English:** Standard narration
- **Urdu:** Translates text, generates Urdu speech

**Voice Selection:**
- Child-friendly voice by default
- Provider-specific voice IDs map correctly

**Output:**
- WAV/MP3 audio file per scene
- Duration extracted for video synchronization
- Stored in `media/audio/scene_<timestamp>.wav`

---

### 6️⃣ **Decision Engine** (`decision_engine.py`)

**How Branching Works:**
1. User views story preview, encounters decision point (Scene 1, 2, or 3)
2. Clicks Option A or Option B
3. Decision engine:
   - Finds first unapplied decision point
   - Appends consequence marker to next scene's text
   - Marks decision as applied
   - Records user's choice
4. Next scene regenerates with new context (if needed)
5. Scene 4 shows final consequences of accumulated choices

**Current Implementation (30% scope):**
- Adds choice annotation to next scene
- Simple but extensible for full LLM regeneration later

**Data Tracked:**
- `applied_decisions[]` - List of applied decision indices
- `last_choice` - Most recent choice ("A" or "B")

---

### 7️⃣ **Subtitle Generation** (`subtitle_engine.py`)

**Format:** SRT (SubRip Text)
```
1
00:00:00,000 --> 00:00:05,000
Scene 1 text here...

2
00:00:05,000 --> 00:00:10,000
Scene 2 text here...
```

**Timing Calculation:**
- Based on audio duration from TTS
- Cumulative: Scene 1 (0-5s), Scene 2 (5-10s), etc.
- Character-level timing for precise sync

**Output:** `subtitles/story_<id>.srt`

---

### 8️⃣ **Video Assembly** (`video_engine.py`)

**Final Video Components:**
```
[Scene 1 Image] ─────────────────────┐
[Scene 1 Audio] (+ background music) ├─→ [Video Clip 1]
[Subtitles]                          │
                                     ├─→ Concatenate all clips
[Scene 2 Image] ─────────────────────┤
[Scene 2 Audio] (+ background music) ├─→ [Video Clip 2]
[Subtitles]                          │
                                     └─→ [Final MP4 - 1920x1080]
... (Scenes 3-4)
```

**Dependencies:**
- **MoviePy 2.0+** - Video composition
- **FFmpeg** - Codec support (must be installed separately)
- **ImageMagick** - Text rendering (for subtitles)
- **Pillow** - Image processing

**Video Settings:**
- Resolution: 1920×1080 (Full HD)
- Frame Rate: 30fps
- Codec: libx264 (H.264)
- Audio: AAC

**ImageMagick Setup (Windows):**
- Auto-detects at:
  ```
  C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe
  E:\ImageMagick-7.1.2-Q16-HDRI\magick.exe
  ```
- Or system PATH
- Can be overridden via `IMAGEMAGICK_BINARY` env var

**Output:** `videos/story_<id>.mp4`

---

## 📊 Database Models

### `StoryRequest` (Main Record)
```python
- id: Primary Key
- user: ForeignKey(User, nullable)  # User who created story
- child_name: CharField(100)         # Child's name
- child_age: PositiveIntegerField    # Age 3-12
- moral_theme: CharField(100)        # Theme (Honesty, Kindness, etc.)
- prompt: TextField                  # Custom user prompt
- avatar_path: CharField(500)        # Path to avatar image
- preferred_language: Char(2)        # "en" or "ur"
- llm_provider: CharField(50)        # Provider used (gemini, openai, etc.)
- image_provider: CharField(50)      # Image generation provider
- tts_provider: CharField(50)        # TTS provider
- story_json: JSONField              # Generated story structure
- title: CharField(200)              # Story title
- video_path: CharField(500)         # Final video path
- subtitle_path: CharField(500)      # SRT subtitles path
- status: CharField(20)              # pending/generating/completed/failed
- created_at: DateTimeField          # Creation timestamp
- updated_at: DateTimeField          # Last update
- completed_at: DateTimeField        # Completion timestamp (nullable)
- api_calls_count: IntegerField      # API call tracking
- error_message: TextField           # Error details if failed
```

### `StoryScene` (Individual Scene)
```python
- id: Primary Key
- story_request: ForeignKey(StoryRequest)
- scene_id: IntegerField             # Scene number (1-4)
- text: TextField                    # Scene narrative
- image_prompt: TextField            # Prompt for image generation
- image_path: CharField(500)         # Path to generated image
- audio_path: CharField(500)         # Path to generated audio
- decision: JSONField                # Branching decision {"A": "...", "B": "..."}
- applied_decision: CharField(1)     # "A" or "B" if applied
- created_at: DateTimeField
```

---

## 🛠️ Technology Stack Breakdown

### Backend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Django 4.2+ | Request handling, ORM, admin panel |
| **Database** | SQLite (or PostgreSQL) | Story/scene persistence |
| **Task Queue** | Optional Django-Q | Async story generation (future) |

### AI/ML
| Component | Technology | Options |
|-----------|-----------|---------|
| **Story LLM** | Gemini 1.5 Flash (primary) | OpenAI, Groq alternatives |
| **Image Gen** | Google Imagen (primary) | DALL-E, Stability AI, Local SD |
| **TTS** | Gemini TTS (primary) | OpenAI, ElevenLabs, Coqui alternatives |
| **Avatar Conv.** | Local Stable Diffusion | Flask server via Ngrok bridge |

### Media Processing
| Component | Library | Purpose |
|-----------|---------|---------|
| **Images** | Pillow, OpenCV | Resize, format conversion |
| **Video** | MoviePy 2.0+ | Clip composition, concatenation |
| **Audio** | google-genai, openai, elevenlabs | TTS synthesis |
| **Subtitles** | Manual SRT generation | Timing calculation & formatting |
| **System Tools** | FFmpeg, ImageMagick | Video encoding, text rendering |

### Frontend
| Component | Technology |
|-----------|-----------|
| **Templates** | Django HTML/Jinja2 |
| **Styling** | CSS (professional.css) |
| **Interactivity** | JavaScript (decision handling) |

---

## 🔐 Configuration & Secrets

### Required Environment Variables (`.env`)
```bash
# Django
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# API Keys
GEMINI_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key  (optional)
GROQ_API_KEY=your-groq-key      (optional)
ELEVENLABS_API_KEY=...          (optional)

# Local Image Server
LOCAL_IMAGE_API_URL=https://your-ngrok-url.ngrok-free.dev/generate

# Defaults
LLM_PROVIDER=gemini
IMAGE_PROVIDER=gemini
TTS_PROVIDER=gemini

# Rate Limiting
DAILY_STORY_LIMIT=3

# Video
SUBTITLE_FONT=/path/to/font.ttf   (optional, auto-detected)
IMAGEMAGICK_BINARY=/path/to/magick.exe  (optional, auto-detected)
```

---

## 📂 Project Structure

```
MoralVerse_FYP/
├── manage.py                      # Django entry point
├── db.sqlite3                     # Development database
├── requirements.txt               # Python dependencies
├── README.md                      # User documentation
│
├── moralverse/                    # Django project config
│   ├── settings.py               # Settings, env loading
│   ├── urls.py                   # URL routing
│   ├── wsgi.py                   # WSGI for deployment
│   └── __init__.py
│
├── core/                          # Main Django app
│   ├── views.py                  # Request handlers
│   ├── models.py                 # StoryRequest, StoryScene
│   ├── urls.py                   # Core URL patterns
│   ├── admin.py                  # Django admin config
│   ├── auth_views.py             # Login/signup handlers
│   ├── migrations/               # DB migrations
│   └── services/
│       └── pipeline.py           # Main orchestration
│
├── ai_modules/                    # AI/ML engines
│   ├── llm_engine.py             # Story generation
│   ├── image_engine.py           # Scene illustrations
│   ├── avatar_processor.py       # Avatar handling
│   ├── tts_engine.py             # Voice synthesis
│   ├── decision_engine.py        # Branching logic
│   ├── subtitle_engine.py        # SRT generation
│   ├── video_engine.py           # Video assembly
│   ├── runtime.py                # Render settings
│   └── __init__.py
│
├── templates/                     # HTML views
│   ├── home.html                 # Main page
│   ├── landing.html              # Landing page
│   ├── story_preview.html        # Story viewing
│   ├── video_ready.html          # Video display
│   ├── user_dashboard.html       # User profile
│   ├── admin_dashboard.html      # Admin panel
│   └── auth/                     # Auth templates
│
├── static/
│   └── css/
│       └── professional.css      # Styling
│
├── media/                         # Generated files
│   ├── images/                   # Scene images
│   ├── audio/                    # TTS audio
│   ├── videos/                   # Final videos
│   ├── subtitles/                # SRT files
│   └── avatars/                  # Converted avatars
│
├── logs/                          # Debug logs
├── scripts/                       # Helper scripts
│   ├── list_models.py           # Available LLM models
│   └── verify_keys.py           # API key verification
│
└── tests/                         # Test files
    └── scratch_test_*.py         # Integration tests
```

---

## 🚀 How to Start the Project

### Step 1: Setup Python Environment
```bash
cd MoralVerse_FYP
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # Unix/Mac
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Create `.env` File
Copy from `.env.example` and add:
```
GEMINI_API_KEY=your-key-here
LOCAL_IMAGE_API_URL=https://your-ngrok.ngrok-free.dev/generate
```

### Step 4: Start Local Image Server (CRITICAL!)
- Open `imagemodel.ipynb` (in Image Model folder)
- Run all cells
- Copy Ngrok URL and update `.env`

### Step 5: Run Migrations
```bash
python manage.py migrate
```

### Step 6: Start Django Server
```bash
python manage.py runserver
```

### Step 7: Access Platform
Open `http://127.0.0.1:8000` in browser

---

## 🔄 Typical User Workflow

1. **Land on homepage** → Click "Create Story"
2. **Enter details** → Child name, age, moral theme
3. **Upload avatar** (optional) → Photo converted to Ghibli style
4. **Select providers** → LLM, Image, TTS (defaults: Gemini for all)
5. **Click Generate** → Redirected to preview page
6. **Watch progress** → Story → Images → Audio → Video assembly
7. **View story** → Read, see images, listen to audio
8. **Make decisions** → Click A/B at decision points to branch story
9. **Download video** → Final MP4 with subtitles ready

---

## 🐛 Debugging & Logs

**Log Files:**
- Main logs: `logs/api_debug.log`
- Django logs: Console output
- Database: `db.sqlite3`

**Check API Keys:**
```bash
python scripts/verify_keys.py
```

**List Available LLM Models:**
```bash
python scripts/list_models.py
```

**Test Image API:**
```bash
python scratch/test_image_api.py
```

---

## 🔍 Key Design Patterns

### Provider Pattern
- Multiple providers (LLM, Image, TTS) with consistent interface
- Graceful fallback to "stub" mode for testing
- Env vars for easy switching

### Pipeline Orchestration
- `pipeline.py` coordinates all steps
- Database records tracking at each stage
- Error handling with rollback capability

### Character Consistency
- **Character anchor:** Same string in every image prompt
- **img2img guidance:** Reference images for visual flow
- **Avatar description:** Text-based character info

### Async Processing (Future)
- Django-Q integration ready (optional)
- Current: Synchronous per request
- Can be extended to background tasks

---

## ⚠️ Known Limitations & TODOs

### Current (30% Implementation)
- ✅ 4-scene story generation
- ✅ Basic character consistency via anchors
- ✅ TTS with English/Urdu support
- ✅ Video assembly
- ⚠️ Decision engine: Simple (just annotates next scene)
- ⚠️ Image consistency: Basic (character anchor + previous scene)

### Future Enhancements
- 🔄 Full LLM-driven decision regeneration (regenerate scenes after choices)
- 🔄 Advanced img2img consistency (SDXL, ControlNet)
- 🔄 Background music/sound effects
- 🔄 Async task queue (Django-Q integration)
- 🔄 User authentication/profiles (partially done)
- 🔄 Social sharing/download features
- 🔄 Analytics dashboard

---

## 📈 Performance Considerations

| Stage | Duration | Bottleneck | Optimization |
|-------|----------|-----------|--------------|
| **Story Gen** | 10-20s | LLM API latency | Use Gemini Flash (fast) |
| **Image Gen** | 30-60s per scene | Model inference | Batch requests, local SD |
| **Avatar Conv** | 10-20s | img2img inference | Cached avatars |
| **TTS Gen** | 5-10s per scene | API latency | Parallel requests |
| **Video Assembly** | 20-40s | Video encoding | FFmpeg hardware accel |
| **Total** | 2-3 minutes | Image generation | Pre-cache common prompts |

---

## 🎯 Success Criteria (FYP Evaluation)

The project demonstrates:
✅ Full-stack AI integration (LLM + Image Gen + TTS)
✅ Complex orchestration (7+ AI modules)
✅ Child-focused UX (age-appropriate content)
✅ Technical depth (Stable Diffusion, character consistency)
✅ Bilingual support (English + Urdu)
✅ Production-ready architecture (Django ORM, error handling)
✅ Media output (High-quality MP4 videos)

---

## 📞 Support & References

- **API Docs:** See individual module docstrings
- **Django Docs:** https://docs.djangoproject.com/
- **MoviePy:** https://github.com/Zulko/moviepy
- **Gemini API:** https://ai.google.dev/
- **Stable Diffusion:** https://huggingface.co/stabilityai/stable-diffusion

---

**Project Status:** Ready for FYP submission
**Last Updated:** May 2026
**Version:** 1.0 (Full pipeline working)
