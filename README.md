<p align="center">
  <img src="https://img.shields.io/badge/PolicyAI-Health%20Insurance%20Intelligence-blue?style=for-the-badge&logo=shield" alt="PolicyAI" />
</p>

<h1 align="center">PolicyAI — Health Insurance Intelligence Platform</h1>

<p align="center">
  <em>AI that reads every page of your health insurance policy — including the hidden traps — so you don't have to.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Next.js-14-black?logo=next.js" alt="Next.js 14" />
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/GPT--4o--mini-Structured%20JSON-412991?logo=openai" alt="GPT-4o-mini" />
  <img src="https://img.shields.io/badge/Supabase-pgvector-3ECF8E?logo=supabase" alt="Supabase" />
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python" alt="Python 3.11" />
  <img src="https://img.shields.io/badge/Tailwind-CSS-06B6D4?logo=tailwindcss" alt="Tailwind CSS" />
</p>

---

## 📌 The Problem

**68% of health insurance claims in India get rejected** — not because the treatment isn't covered, but because policyholders don't understand the fine print. Policies are 50–60 page legal documents filled with hidden sub-limits, proportional deductions, narrow definitions, and buried waiting periods.

No existing tool reads the **actual policy PDF** and explains coverage in plain English while detecting hidden traps.

## 💡 The Solution

**PolicyAI** is an end-to-end AI-powered platform covering the **full health insurance lifecycle** — from finding the right policy to filing a successful claim. It uses a **3-layer hybrid RAG pipeline** to cross-reference definitions, exclusions, and conditions, detecting **8 types of hidden policy traps** that cause real-world claim rejections.

---

## ✨ Features

### 1. 🔍 Natural Language Policy Discovery
> _"I need maternity coverage, have diabetes, budget ₹18,000/year, family of 3"_

- GPT-4o-mini extracts structured requirements (needs, budget, conditions, policy type)
- `PolicyRanker` scores and ranks all catalog policies by budget fit, coverage match, PED wait, exclusion risk, network size
- Returns scored policy cards with match percentage and reasons

### 2. ⚖️ Side-by-Side Policy Comparison
- Select 2–3 policies → generates a **19-dimension comparison matrix**
- Covers premiums, waiting periods, co-pay, room rent, maternity, OPD, mental health, AYUSH, dental, NCB, restoration, network hospitals
- AI-generated comparison summary with "best for" recommendations

### 3. 🧠 Hybrid RAG Policy Q&A + Hidden Conditions Detector
> _"Is knee replacement surgery covered?"_

Upload any policy PDF and ask any coverage question. The **3-layer hybrid RAG pipeline** retrieves the most relevant clauses:

| Layer | Method | What It Searches |
|---|---|---|
| **Layer 1** | Semantic (pgvector) + Keyword (tsvector) → RRF Fusion | Direct answer clauses (top 5) |
| **Layer 2** | Section-filtered semantic search | Definitions section (top 3) |
| **Layer 3** | Section-filtered semantic search | Exclusions + Conditions + Limits + Waiting Periods (top 3) |

**Output includes:**
- **Verdict:** COVERED / NOT_COVERED / PARTIALLY_COVERED / AMBIGUOUS
- **Practical Claimability:** 🟢 GREEN / 🟡 AMBER / 🔴 RED
- **Confidence Score:** 0–100%
- **8 Hidden Trap Types** detected with evidence and impact
- **Citations** with exact clause text and page numbers
- **Actionable Recommendation** for the policyholder

### 4. 🏥 Medical Report → Smart Policy Matching
- Input free text or upload a medical report PDF
- AI extracts conditions (name, ICD hint, type, severity)
- Matches against all catalog policies' exclusion lists
- Returns ranked policies flagged with per-condition exclusion risks

### 5. 📋 Claim Eligibility Advisory
- Select policy + enter diagnosis + choose treatment type
- Computes **Claim Feasibility Score (0–100)** with deductions for hidden traps
- Returns required documents checklist by claim type (hospitalization, surgery, maternity, OPD, critical illness)

### 6. 🔎 Coverage Gap Analyzer
- Scans any policy against a standard coverage checklist (7 critical areas + 3 dynamic checks)
- Each gap rated by severity (HIGH / MEDIUM / LOW) with plain-English explanation and recommendation

---

## 🕵️ The 8 Hidden Trap Types

| Type | What It Means |
|---|---|
| `room_rent_trap` | Room rent cap → ALL charges proportionally reduced |
| `pre_auth_required` | Pre-authorization missed → entire claim denied |
| `proportional_deduction` | Sub-limit breach → entire bill cut proportionally |
| `definition_trap` | Key terms defined narrowly (e.g., "Hospitalization" = 24+ hrs only) |
| `waiting_period` | Specific illness or PED waiting period applies |
| `sub_limit` | Cap on specific treatments within broad coverage |
| `documentation` | Non-obvious or time-sensitive document requirements |
| `network_restriction` | Non-network hospital co-pay or full exclusion |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        FRONTEND (Vercel)                         │
│              Next.js 14 + Tailwind CSS + Lucide Icons            │
│                                                                  │
│   ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐     │
│   │  /discover   │  │    /qa       │  │     /claim         │     │
│   │  Discovery & │  │  Policy Q&A  │  │  Claim Advisory +  │     │
│   │  Comparison  │  │  + Upload    │  │  Medical Match +   │     │
│   │              │  │              │  │  Gap Analysis      │     │
│   └──────┬───────┘  └──────┬───────┘  └────────┬───────────┘     │
└──────────┼──────────────────┼───────────────────┼────────────────┘
           │                  │                   │
           ▼                  ▼                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                      BACKEND (Render)                            │
│                    FastAPI · Python 3.11                          │
│                                                                  │
│  ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐  │
│  │ Routers (3)      │  │ Services (7)    │  │ Skills (3)       │  │
│  │  discovery.py    │  │  embedder.py    │  │ HiddenConditions │  │
│  │  qa.py           │  │  llm.py         │  │  Detector        │  │
│  │  claim.py        │  │  pdf_parser.py  │  │ CoverageGap      │  │
│  │                  │  │  vector_store   │  │  Scanner         │  │
│  │ 10 API endpoints │  │  med_extractor  │  │ PolicyRanker     │  │
│  │                  │  │  tools.py       │  │                  │  │
│  └─────────────────┘  └────────┬───────┘  └──────────────────┘  │
└────────────────────────────────┼─────────────────────────────────┘
                                 │
               ┌─────────────────┼──────────────────┐
               ▼                 ▼                   ▼
      ┌────────────────┐ ┌─────────────┐  ┌──────────────────┐
      │   Supabase     │ │   OpenAI    │  │    OpenAI         │
      │  PostgreSQL +  │ │  Embeddings │  │   GPT-4o-mini     │
      │  pgvector      │ │  text-emb-  │  │   Structured JSON │
      │  tsvector      │ │  3-small    │  │   temp=0.1        │
      └────────────────┘ └─────────────┘  └──────────────────┘
```

### RAG Pipeline Flow

```
User Question
      │
      ▼
  Embed Query (text-embedding-3-small → 1536-dim)
      │
      ├──► Layer 1a: Semantic Search (pgvector, top 8)  ──┐
      ├──► Layer 1b: Keyword Search  (tsvector, top 8)  ──┤──► RRF Fusion (k=60) → top 5
      ├──► Layer 2:  Section Search  (definitions, top 3) ─────────────────┐
      └──► Layer 3:  Section Search  (exclusions+conditions+limits, top 3) ┤
                                                                           ▼
                                                        GPT-4o-mini Synthesis
                                                        (Senior Claims Consultant prompt)
                                                                           │
                                                                           ▼
                                                        Structured JSON Verdict
                                                        + Hidden Conditions
                                                        + Citations + Recommendation
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Next.js 14 (App Router) | Server/client components, file-based routing |
| **Styling** | Tailwind CSS + Lucide Icons | Responsive UI with icon system |
| **API Client** | Axios | HTTP client for backend communication |
| **Backend** | FastAPI (Python 3.11) | High-performance async REST API |
| **LLM** | OpenAI GPT-4o-mini | Structured JSON reasoning at temperature 0.1 |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim vectors with retry + batch support |
| **Vector DB** | Supabase pgvector | Cosine similarity ANN search (IVFFlat index) |
| **Keyword Search** | PostgreSQL tsvector | BM25-style full-text search (GIN index) |
| **Search Fusion** | Reciprocal Rank Fusion (RRF) | Merges semantic + keyword results (k=60) |
| **PDF Parsing** | PyMuPDF (fitz) | Section-aware chunking with regex heading detection |
| **Deployment** | Vercel + Render | Frontend CDN + Backend auto-deploy |

---

## 📁 Project Structure

```
PolicyAI/
├── CLAUDE.md                          # Project documentation
├── render.yaml                        # Render deployment config
│
├── backend/
│   ├── main.py                        # FastAPI app + lifespan startup seeder
│   ├── requirements.txt               # Python dependencies
│   ├── data/
│   │   ├── schema.sql                 # Supabase schema (tables + indexes + RPCs)
│   │   └── seed_policies.json         # 10-insurer structured catalog
│   ├── routers/
│   │   ├── discovery.py               # /api/discover, /api/compare
│   │   ├── qa.py                      # /api/upload, /api/policies, /api/ask
│   │   └── claim.py                   # /api/claim-check, /api/extract-*, /api/match-*, /api/gap-*
│   ├── scripts/
│   │   ├── seed_db.py                 # Populate insurance_policies catalog table
│   │   └── startup_seeder.py          # Auto-embed all PDFs from policies/ on boot
│   └── services/
│       ├── embedder.py                # OpenAI embedding wrapper (single + batch + retry)
│       ├── llm.py                     # GPT-4o-mini structured JSON + text helpers
│       ├── pdf_parser.py              # Section-aware PDF chunking (30+ regex patterns)
│       ├── vector_store.py            # Supabase: semantic, keyword, section, RRF, catalog CRUD
│       ├── medical_extractor.py       # Condition extraction from text/PDF + exclusion matching
│       ├── skills.py                  # HiddenConditionsDetector, CoverageGapScanner, PolicyRanker
│       └── tools.py                   # 8 tool implementations + OpenAI function-call schemas
│
├── frontend/
│   ├── package.json
│   ├── next.config.ts
│   ├── tsconfig.json
│   ├── app/
│   │   ├── layout.tsx                 # Root layout
│   │   ├── page.tsx                   # Landing page — 3-mode selector
│   │   ├── globals.css                # Tailwind base styles
│   │   ├── discover/page.tsx          # Feature 1+2: Discovery + Comparison
│   │   ├── qa/page.tsx                # Feature 3: Policy Q&A + Upload
│   │   └── claim/page.tsx             # Feature 4+5+6: Claim + Medical + Gap
│   ├── components/
│   │   ├── AnswerCard.tsx             # Verdict badge, claimability, hidden conditions, citations
│   │   ├── ComparisonTable.tsx        # 19-dimension comparison matrix
│   │   └── PolicyCard.tsx             # Scored policy card with match % and feature pills
│   └── lib/
│       └── api.ts                     # All API client functions (Axios)
│
└── Policies/
    └── tata/                          # 30+ real Tata AIG policy PDFs
        └── *.pdf
```

---

## 🗄️ Database Schema

### Tables

| Table | Purpose |
|---|---|
| `insurance_policies` | Structured catalog — premiums, coverage flags, exclusions, waiting periods for 10 insurers |
| `uploaded_policies` | Tracks embedded PDF documents — filename, insurer, chunk count |
| `policy_chunks` | Text chunks with `embedding VECTOR(1536)` + `content_tsv TSVECTOR` + `section_type` |

### Indexes

| Index | Type | Purpose |
|---|---|---|
| `policy_chunks_embedding_idx` | IVFFlat (lists=100) | Fast ANN cosine similarity on embeddings |
| `policy_chunks_tsv_idx` | GIN | Full-text keyword search on tsvector |
| `policy_chunks_policy_section_idx` | B-tree composite | Fast section-filtered queries |

### Supabase RPC Functions

| Function | Purpose |
|---|---|
| `match_chunks_direct()` | Direct semantic similarity search |
| `match_chunks_by_section()` | Section-filtered semantic search |
| `keyword_search_chunks()` | Full-text keyword search with `ts_rank_cd` ranking |

---

## 🌐 API Endpoints

| Method | Endpoint | Feature | Description |
|---|---|---|---|
| `GET` | `/api/health` | System | Health check |
| `POST` | `/api/discover` | Discovery | NL query → extracted requirements → ranked policies |
| `POST` | `/api/compare` | Comparison | 2–3 policy IDs → 19-dimension comparison matrix + AI summary |
| `GET` | `/api/policies` | Q&A | List all uploaded/embedded policies |
| `POST` | `/api/upload` | Q&A | Upload PDF → section-aware chunk → embed → store |
| `POST` | `/api/ask` | Q&A | Question + policy → 3-layer hybrid RAG → verdict + hidden traps |
| `POST` | `/api/claim-check` | Claim | Diagnosis + policy → feasibility score + document checklist |
| `POST` | `/api/extract-conditions` | Medical | Free text → extracted medical conditions |
| `POST` | `/api/extract-conditions-file` | Medical | PDF upload → extracted medical conditions |
| `POST` | `/api/match-conditions` | Medical | Conditions array → ranked policies with exclusion flags |
| `GET` | `/api/gap-analysis/{id}` | Gap | Policy ID → coverage gaps sorted by severity |

---

## 🚀 Getting Started

### Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.11+
- **Supabase** account (free tier works)
- **OpenAI** API key

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PolicyAI.git
cd PolicyAI
```

### 2. Set Up Supabase

1. Create a new Supabase project
2. Go to **SQL Editor** → paste and run `backend/data/schema.sql`
3. Copy your **Project URL** and **Service Role Key**

### 3. Configure Environment Variables

**Backend** — create `backend/.env`:

```env
OPENAI_API_KEY=sk-...
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
```

**Frontend** — create `frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 4. Start the Backend

```bash
cd backend
pip install -r requirements.txt
python scripts/seed_db.py          # Seed the structured catalog (run once)
uvicorn main:app --reload --port 8000
```

> On startup, the backend automatically scans `Policies/` and embeds any new PDFs.

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) — you're ready to go!

### 6. Add More Policy PDFs (Optional)

```bash
mkdir -p Policies/hdfc
# Drop PDF files into the folder
# Restart the backend — the startup seeder handles the rest
```

**Convention:** `Policies/{insurer_slug}/{policy_filename}.pdf`

---

## 📊 Data

### 10 Insurers in Structured Catalog

| # | Insurer | Plan Name | Type |
|---|---|---|---|
| 1 | Star Health and Allied Insurance | Star Health Assure | Individual |
| 2 | HDFC ERGO General Insurance | HDFC Ergo Optima Secure | Family Floater |
| 3 | Niva Bupa Health Insurance | Niva Bupa ReAssure 2.0 | Family Floater |
| 4 | Care Health Insurance | Care Supreme | Individual |
| 5 | Bajaj Allianz General Insurance | Bajaj Allianz Health Care Supreme | Family Floater |
| 6 | ICICI Lombard General Insurance | ICICI Lombard Complete Health | Family Floater |
| 7 | Aditya Birla Health Insurance | Activ Health Platinum Enhanced | Individual |
| 8 | New India Assurance Company | New India Mediclaim | Individual |
| 9 | Tata AIG General Insurance | Tata AIG Medicare Premier | Family Floater |
| 10 | ManipalCigna Health Insurance | ManipalCigna ProHealth Prime | Individual |

### 30+ Real Policy PDFs
Tata AIG Medicare Premier collection — fully parsed, chunked, and embedded in Supabase pgvector.

---

## ☁️ Deployment

### Frontend → Vercel

```bash
cd frontend
npx vercel --prod
```

Set environment variable: `NEXT_PUBLIC_API_URL=https://your-backend.onrender.com`

### Backend → Render

The included `render.yaml` configures automatic deployment:

```yaml
services:
  - type: web
    name: policyai-backend
    runtime: python
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    rootDir: backend
    envVars:
      - key: OPENAI_API_KEY
      - key: SUPABASE_URL
      - key: SUPABASE_SERVICE_KEY
```

---

## 🧪 Testing

```bash
cd backend
python -m pytest tests/
```

---

## 📐 Technical Decisions

| Decision | Why |
|---|---|
| **Hybrid RAG (semantic + keyword)** | Legal documents have exact terms ("Code-Excl01") that semantic search alone misses |
| **RRF Fusion (k=60)** | Rank-based merge — no score normalization needed across different retrieval methods |
| **Section-aware chunking** | Must cross-reference definitions ↔ exclusions to catch hidden traps |
| **3-layer search** | Single query misses definition and exclusion cross-references |
| **GPT-4o-mini** | Best cost-to-quality ratio for structured JSON output at low temperature |
| **Supabase pgvector** | Single database for vectors + keywords + metadata — no separate vector DB needed |
| **IVFFlat index** | Good accuracy/speed tradeoff for <100K chunks; faster index builds than HNSW |
| **PyMuPDF** | Superior text extraction quality for formatted insurance PDFs vs. alternatives |
| **400-token chunks, 80 overlap** | Balanced granularity — large enough for context, small enough for precision |

---

## 📈 Project Stats

| Metric | Value |
|---|---|
| Python source files | 12 |
| TypeScript source files | 9 |
| API endpoints | 10 |
| AI skills | 3 |
| Tool implementations | 8 |
| Hidden trap types | 8 |
| Insurers in catalog | 10 |
| Real policy PDFs | 30+ |
| Section regex patterns | 30+ |
| Comparison dimensions | 19 |
| Gap checklist items | 10 |
| Database tables | 3 |
| Database indexes | 3 |
| RPC functions | 3 |
| Embedding dimensions | 1,536 |

---

## 🔮 Future Scope

- **Multi-turn Conversational Agent** — follow-up questions with context memory
- **PDF-to-PDF Comparison** — compare two uploaded policy documents directly
- **Multi-language Support** — Hindi, Tamil, Telugu policy Q&A
- **WhatsApp Bot Integration** — policy Q&A for broader accessibility
- **Real-time Premium Quotes** — integrate insurer APIs for live pricing
- **Claim Tracking Dashboard** — real-time claim status monitoring
- **Policy Renewal Advisor** — proactive recommendations at renewal time

---

## 📝 License

This project is built for educational and demonstration purposes.

---

<p align="center">
  <strong>PolicyAI</strong> — <em>Because understanding your health insurance shouldn't require a law degree.</em>
</p>
