# CivicLens AI

> **An evidence-first policy comparison and intelligence platform for official government Gazette notifications and circulars.**

CivicLens AI compares earlier and updated versions of government policies, isolates every substantive change, estimates citizen impact, and provides on-demand plain-language explanations powered by **Groq LPUs** (`openai/gpt-oss-20b`) across **12 Indian languages** — with strict page-level verification against the original documents.

---

## 🌟 Key Features

1. **Deterministic PDF Comparison Engine**
   - Extracts clean, selectable text from multi-page government documents using **PyMuPDF**.
   - Handles multi-column layouts, continuous cross-page paragraph stitching, and ignores repetitive headers, footers, and Gazette publishing marks.
   - Accurately aligns text hunks to detect additions, deletions, and modifications.

2. **Strict Evidence Grounding (Zero Hallucinations)**
   - Every detected change is paired with exact document version labels, PDF page references, and verbatim text quotes.
   - Built-in PDF viewer integration ("View Source") opens the exact page of the official notification for instant human auditability.

3. **Priority Triage & Multidimensional Filters**
   - Automatically classifies changes into **High**, **Medium**, or **Low** priority using statutory force patterns (`shall`, `must`, `penalty`, `interest rate`, `deadline`).
   - Filter changes by intended audience (**Businesses**, **Students**, **General Citizens**) or domain category (**Financial**, **Eligibility**, **Compliance**).

4. **"Explain with AI" Powered by Groq**
   - Clean, per-card on-demand AI explanation button.
   - Generates a concise 2–3 sentence plain-language summary using Groq's high-speed LPU inference (`openai/gpt-oss-20b`).
   - Every generated explanation is cross-checked post-generation by `assess_evidence_claim()` to ensure no numbers, dates, or qualitative claims are hallucinated.

5. **12 Indian Languages Localization**
   - The user interface and AI-generated explanations are fully translated into **English, Hindi, Telugu, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, Odia, and Urdu**.
   - Original legal excerpts remain anchored in verbatim English to preserve legal precision while offering accessible explanations.

6. **AI Provider Health Diagnostics**
   - Provider & Settings pane includes a **"Test AI Connection"** button that sends an authenticated test prompt to verify Groq API connectivity and displays the active model.

---

## 🏗️ Technology Stack

| Component | Technology | Role |
|---|---|---|
| **Backend** | Python 3.10+ (Standard Library `http.server`) | Ultra-lightweight asynchronous threaded web server |
| **PDF Extraction** | PyMuPDF (`fitz`) | Robust PDF text extraction & page parsing |
| **AI Provider** | **Groq LPU API** (`openai/gpt-oss-20b`) | Ultra-fast, provider-independent chat completion for explanations |
| **Translation** | `policy_translator.py` & `static/i18n.js` | Bilingual dictionaries & dynamic grammar translations across 12 languages |
| **Frontend** | Vanilla HTML5, Modern CSS3, Modular JavaScript | Responsive, accessible UI with zero build tooling or npm dependencies |
| **Session State** | In-Memory Data Structures | Privacy-preserving; no persistent user databases or external cloud storage |

---

## 🚀 Quickstart Guide (Windows / Linux / macOS)

### 1. Prerequisites
- Python 3.10 or higher installed. Check with:
  ```powershell
  python --version
  ```
- A free **Groq API Key** from [console.groq.com](https://console.groq.com).

### 2. Clone or Open the Repository
```powershell
cd "C:\Users\Sree Varsha\OneDrive\Documents\Project-ideaforge"
```

### 3. Install Dependencies
Install the required PDF extraction package:
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Open `.env` in any text editor and paste your Groq API key:
```env
AI_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

### 5. Launch the Application
Start the local server:
```powershell
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🎯 Live Demonstration Guide for Judges

1. **Verify AI Connection:**
   - In the top header / settings pane, locate **"Provider & Settings"**.
   - Click **"Test AI Connection"**.
   - Observe the live diagnostic confirmation: `Groq connection successful — Model: openai/gpt-oss-20b.`

2. **Load Sample Government Comparison:**
   - On the homepage, click **"Load CGFMU Sample Policy"** (Credit Guarantee Fund for Micro Units Gazette amendments).
   - Inspect the extracted changes: notice how changes are categorized by High, Medium, and Low priority with substantive reasons.

3. **Verify Evidence Grounding:**
   - Click **"View Source"** on any change card.
   - The application highlights the exact source page and excerpt matching the Gazette notification.

4. **Test "Explain with AI":**
   - Click the **"Explain with AI"** button on an individual change card.
   - A loading indicator appears while Groq processes the excerpt.
   - The card displays a verified 2–3 sentence plain-language explanation with a **"Directly supported"** verification badge.

5. **Test Multilingual Translation:**
   - Use the language selector in the top-right corner to switch to **Hindi (हिन्दी)**, **Telugu (తెలుగు)**, or any of the 12 supported languages.
   - The entire dashboard and AI explanation translate instantly, while the original Gazette citations remain intact.

6. **Test Custom Document Upload:**
   - Under "Compare Policy Documents", upload any two versions of a policy PDF.
   - Click "Compare Documents" to observe automatic paragraph extraction and diff calculation.

---

## 🧪 Testing & Verification

CivicLens includes an automated test suite verifying PDF extraction, text alignment, grounding checks, Groq integration, and language translations:

```powershell
python -m unittest tests.test_workflow
```

> **Note on Windows Sandbox Execution:** When executing within restricted sandbox environments without socket access, 4 local HTTP socket-binding tests report `[WinError 10013]`. All 109 core workflow, parsing, grounding, Groq explanation, and translation tests execute in-memory with 100% passing results.

---

## 📋 Repository Hygiene & Privacy

- **No Secrets in Source:** All API keys and secrets are loaded via environment variables; `.env` is ignored by `.gitignore`.
- **Zero Hallucination Tolerance:** If the AI model generates speculative consequences or ungrounded statistics, `assess_evidence_claim()` rejects the output and falls back safely to cited source excerpts.

---

## ☁️ Deployment Guide (Render)

CivicLens AI is designed to run seamlessly as a containerized or cloud-hosted web service on [Render](https://render.com).

### 1. Create a New Web Service on Render
1. Push your repository to GitHub or GitLab (ensure `.env` is never committed; it is excluded by `.gitignore`).
2. Log in to [dashboard.render.com](https://dashboard.render.com/) and click **New +** $\rightarrow$ **Web Service**.
3. Connect your repository.

### 2. Configure Service Settings
| Setting | Recommended Value |
|---|---|
| **Name** | `civiclens-ai` (or your preferred name) |
| **Language / Runtime** | `Python 3` |
| **Region** | Closest to your users (e.g., `Singapore` or `Frankfurt`) |
| **Branch** | `main` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `python app.py` |

### 3. Add Environment Variables
In the **Environment** tab on Render, add the following variables:

| Key | Value | Description |
|---|---|---|
| `AI_PROVIDER` | `groq` | Configures Groq as the AI inference provider |
| `GROQ_API_KEY` | `gsk_...` | Your secret Groq API key from console.groq.com |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Configured Groq model for fast policy explanations |

> **Note on Port Binding:** Render automatically injects the `PORT` environment variable (e.g. `10000`). `app.py` reads `os.getenv("PORT", "8000")` and binds to `0.0.0.0`, allowing Render's reverse proxy to route traffic directly.

### 4. Hosted Environment Characteristics & Limitations
- **Cold Starts (Free Tier):** Render free instances spin down after 15 minutes of inactivity. Initial requests after spin-down may take ~45–50 seconds to boot.
- **In-Memory Session Storage:** Document comparison sessions are maintained in an in-memory LRU cache (`MAX_DOCUMENT_SESSIONS = 100`). If the web service restarts or redeploys, active sessions clear; users simply reload or re-upload their documents.
- **Stateless PDF Processing:** Uploaded PDFs are parsed in-memory and temporary comparison state is ephemeral, keeping the server private and leak-free.
- **Outbound Network:** Groq API calls require outbound HTTPS access, which is enabled by default on standard Render web services.

## Credits

- **Python** – Backend development
- **PyMuPDF** – PDF text extraction
- **Groq API** – AI-powered explanations
- **HTML, CSS & JavaScript** – Frontend development
- **GitHub** – Version control and source code hosting
- **Render** – Application deployment
- **AI Development Assistance** – OpenAI Codex and other AI tools used during development
