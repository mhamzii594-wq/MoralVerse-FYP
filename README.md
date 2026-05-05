# MoralVerse.AI - Final Year Project (FYP)

MoralVerse.AI is an AI-powered interactive storytelling platform designed for children's moral education. It transforms a simple prompt into a full multimedia experience with consistent characters, Ghibli-style illustrations, voice narration, and branching decision points.

---

## 🌟 Key Features

- **Dynamic Storytelling:** Generates 4-scene moral stories based on user-defined themes.
- **Character Consistency:** Uses advanced instruction engineering to keep the main character visually consistent across all scenes.
- **Ghibli Style Illustrations:** High-quality anime-style visuals generated using a local Stable Diffusion pipeline.
- **Bilingual TTS:** Voice narration in English and Urdu.
- **Interactive Branching:** Users make moral decisions that change the story's outcome in real-time.
- **Full Video Assembly:** Combines images, audio, and subtitles into a final video.

---

## 🛠️ Tech Stack

- **Backend:** Python + Django
- **LLM:** Google Gemini 1.5 Flash (Story Generation)
- **Image Generation:** Local Stable Diffusion (Dreamshaper) via Flask/Ngrok Bridge
- **TTS:** Gemini TTS + Google Translate Fallback
- **Video Processing:** MoviePy + FFmpeg

---

## 🚀 How to Start the Project

### 1. Initial Setup
1. **Navigate to the project folder:**
   ```powershell
   cd MoralVerse_FYP
   ```
2. **Create and Activate Virtual Environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Environment Variables:**
   Create a `.env` file from the `.env.example` and add your `GEMINI_API_KEY` and `LOCAL_IMAGE_API_URL`.

### 2. Start the Local Image Server (CRITICAL)
For Ghibli-style images and avatar conversion, the local model server must be running:
1. Open the `imagemodel.ipynb` notebook (found in the `image Model` folder).
2. Run all cells to start the Flask server.
3. Copy the **Ngrok URL** (e.g., `https://...ngrok-free.dev/generate`) and update it in your `.env` file:
   ```env
   LOCAL_IMAGE_API_URL=https://your-ngrok-link.ngrok-free.dev/generate
   ```

### 3. Run the Application
1. **Run Migrations:**
   ```powershell
   python manage.py migrate
   ```
2. **Start the Django Server:**
   ```powershell
   python manage.py runserver
   ```
3. **Access the Platform:**
   Open `http://127.0.0.1:8000` in your browser.

---

## 📖 Usage Guide

1. **Enter Details:** Input the child's name, age, and a moral theme (e.g., "Honesty").
2. **Upload Avatar:** (Optional) Capture or upload a photo. The system will convert it to Ghibli style.
3. **Generate:** Watch as the AI drafts the story and paints the scenes.
4. **Interact:** Make decisions at the choice points to see how the story changes.
5. **Watch & Download:** Once finished, generate the final video to watch your creation!

---

**Developed for the Final Year Project (FYP) Submission.**
**Deadline:** April 25th, 12 AM.
