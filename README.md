# Autonomous Document Agent

A small autonomous agent that takes a natural-language request, plans its own
task list, executes each task with an LLM, self-checks its own draft, and
returns a polished `.docx` business document (proposal, meeting minutes,
project plan, report, spec, SOP, etc).

```
POST /agent   {"request": "..."}   ->   plan + reflection notes + downloadable .docx
```

## 1. Setup (5 minutes)

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and paste a free key from https://console.groq.com/keys
uvicorn app.main:app --reload
```

Then open **http://127.0.0.1:8000** in a browser — a small UI lets you type a
request, watch the agent work through its pipeline, and download the
resulting `.docx` without touching curl or Swagger. (Swagger/OpenAPI docs are
still available at `/docs` if you prefer the raw API view.)

Groq was chosen as the LLM provider because it's free, fast (sub-second
completions), and OpenAI-compatible, which makes the live demo painless.
Swapping providers only touches `app/llm_client.py::_call_groq` — point the
URL at Ollama's local `/v1/chat/completions` or LM Studio's server and
everything else (planning, execution, reflection, docx generation) is
unchanged, since none of it is provider-specific.

Check it's alive:

```bash
curl http://127.0.0.1:8000/health
```

## 2. Architecture & Agent Workflow

```
Request
   │
   ▼
[1] GUARDRAILS   -- reject empty / too long / disallowed content before
   │                 spending any LLM calls (fast, free rejection)
   ▼
[2] PLAN         -- one LLM call converts free text into a structured plan:
   │                 document_type, title, sections, one task per section,
   │                 and explicit "assumptions" for anything ambiguous or
   │                 missing in the request (the agent never asks a
   │                 clarifying question -- it decides and moves on, which
   │                 is the "autonomous" part of the assignment)
   ▼
[3] EXECUTE      -- one LLM call per task/section, generating real content
   │                 (mock data used freely: names, dates, budgets)
   ▼
[4] REFLECT      -- *** the mandatory engineering improvement ***
   │                 a fresh LLM call re-reads the whole draft against the
   │                 ORIGINAL request and flags generic/missing/contradictory
   │                 sections
   ▼
[4b] REPAIR      -- if reflection found real issues, regenerate only the
   │                 flagged sections (capped at one repair pass -- no
   │                 infinite self-correction loops)
   ▼
[5] RENDER       -- python-docx builds a title page (with the assumptions
   │                 listed transparently) + one heading/body per section
   ▼
Response: JSON (plan, reflection notes, word counts) + downloadable .docx
```

**Files**
| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI routes: `POST /agent`, `GET /files/{name}`, `GET /health` |
| `app/planner.py` | The agent itself — guardrails, plan, execute, reflect, repair |
| `app/llm_client.py` | Groq HTTP wrapper, retry/backoff, defensive JSON parsing |
| `app/doc_generator.py` | Turns plan + section text into a formatted `.docx` |
| `app/models.py` | Pydantic schemas (request/response contracts) |
| `app/config.py` | All environment/config in one place |

**Why FastAPI + a plain function pipeline instead of LangChain/CrewAI?**
The whole agent loop is a small, fixed-shape sequence of LLM calls with
clear inputs/outputs: one to plan, one per section (so this scales with
document length, not a fixed count), one to reflect, plus at most one
repair call per section the self-check flags. A framework would add
abstraction (agents, tools, memory objects) for problems this size doesn't
have yet. Plain Python functions keep every prompt and every decision
inspectable in one file (`planner.py`), which mattered more for a
60-minute build than framework features I wouldn't use.

## 3. The Mandatory Improvement: Reflection / Self-Check

**What I implemented:** after the agent drafts all sections, one more LLM
call acts as an editor — it re-reads the _entire draft_ against the
_original request_ and returns structured JSON: `passed`, a list of
`issues`, and a list of `sections_to_revise`. If it finds real problems
(e.g. a section that's generic filler, or a document that ignores something
the user explicitly asked for), the agent regenerates _only those sections_
and returns the repaired draft. This is capped at exactly one repair pass so
the agent can't loop forever chasing a "perfect" document.

**Why I chose this over the alternatives:** planning and multi-step
execution were already required to satisfy the base assignment, and retry
logic alone doesn't catch _content_ mistakes (a syntactically fine LLM
response can still be off-topic or drop a requirement). Reflection is the
one improvement that actually catches the failure mode most likely to embarrass
an autonomous agent: confidently producing a document that quietly misses
what was asked. It's also the most visibly "autonomous" behavior to show in
a demo — the plan JSON and the reflection JSON are both returned in the API
response, so a reviewer can watch the agent both plan and grade its own work.

**How it improves the agent:** without it, a single bad section (e.g. the
LLM ignores the client's stated budget cap) would silently ship in the final
`.docx`. With it, that section gets caught and rewritten before the document
is ever generated — turning a silent quality problem into a visible,
logged, self-corrected one.

## 4. Test Inputs

Run `uvicorn app.main:app --reload`, then in a second terminal:

```bash
python test_requests.py
```

**Test 1 — standard, well-specified request** (meeting minutes with
attendees, date, and topics already given — planning is straightforward,
reflection should pass clean):

```json
{
  "request": "Create meeting minutes for our weekly engineering sync held on July 3rd, 2026. Attendees: Priya (Eng Lead), Sam (Backend), Jordan (Frontend), Ana (QA). We discussed the API rate-limiting rollout, a bug in checkout flow, and Q3 hiring plan. Action items should be assigned to owners with due dates."
}
```

**Test 2 — complex/ambiguous request** (doesn't say what document type is
needed, has a self-contradictory budget — "flexible" but also capped at
$50k — and vague scope ("enterprise-grade") with a tight, unstated
deadline). This forces the agent to: pick a document type itself, resolve
the budget contradiction with an explicit assumption, and fill in what
"enterprise-grade" should mean for a real buyer:

```json
{
  "request": "We need something to send the client about our new product before Friday, but I'm not sure if it should be a proposal or a spec -- honestly whatever gets us the deal fastest. Budget is 'flexible' but also we were told last month not to go over 50k. The client wants it 'enterprise-grade' but also wants a two-week turnaround. Just make it look good and cover whatever a serious buyer would expect."
}
```

Watch the `plan.assumptions` field in the response for exactly how the agent
resolved the ambiguity — that's the part worth narrating in the demo.

Or with `curl`:

```bash
curl -X POST http://127.0.0.1:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"request": "Create meeting minutes for our weekly engineering sync..."}'
```

## 5. Debugging Insight

**Issue:** with `response_format: json_object` set on Groq, the planning
step still occasionally returned JSON wrapped in a ` ```json ... ``` `
fence (a habit carried over from chat-style training), which made
`json.loads()` throw on maybe 1 in 10 calls — an intermittent failure that
only showed up after several test runs, not on the first try.

**Root cause:** the JSON-mode flag constrains the _shape_ of the model's
output but some providers/models still wrap it in prose or fences on the
open-source model checkpoints Groq serves; I'd assumed `json_object` mode
was a hard guarantee of a bare JSON string and it isn't.

**Fix:** `llm_client.safe_json_parse()` strips leading/trailing code fences
and, if that still doesn't parse, falls back to slicing the substring
between the first `{` and last `}` before giving up. This turned an
intermittent 500 into a non-issue, and it's covered by the same function
everywhere the agent expects JSON back (planning _and_ reflection), rather
than being special-cased once and forgotten in the second call site.

## 6. Tradeoff Discussion

**Autonomous Planning vs. Deterministic Workflows.** The agent asks the LLM
to invent its own section list and task breakdown per request, rather than
picking from a fixed set of hardcoded templates per document type. That's
what makes it "autonomous" instead of a form-filler, and it's why the
complex test case above works at all — a fixed template can't decide on its
own that "enterprise-grade" implies a security/compliance section. The cost
is consistency and predictability: two runs of the exact same request can
produce different section lists, which makes the output harder to unit-test
and harder to guarantee against for a client who expects a specific format
every time. The middle ground I leaned on is the `_fallback_plan()` in
`planner.py` — a deterministic 7-section template that only kicks in if the
LLM is unreachable, so the system degrades to "boring but reliable" instead
of failing outright, while still defaulting to autonomous planning whenever
the LLM is available.

## 7. Deploying to Render

The repo includes a `render.yaml` blueprint, so Render can configure the
service automatically:

1. Push this project to a GitHub repo (Render deploys from git, not a zip).
2. In the [Render dashboard](https://dashboard.render.com), click
   **New → Blueprint**, connect the repo, and Render will read `render.yaml`
   and pre-fill the service (build command, start command, free plan).
3. When prompted, set the one secret it can't infer:
   `GROQ_API_KEY` — paste your key from https://console.groq.com/keys.
   (It's marked `sync: false` in the blueprint so it's never committed to git.)
4. Click **Deploy**. Render builds with `pip install -r requirements.txt`
   and starts with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Once live, your UI and API are at the URL Render gives you, e.g.
   `https://autonomous-document-agent.onrender.com`.

**Note on the free tier:**

- The instance spins down after ~15 minutes of inactivity; the first
  request after that takes 30–60s to wake it back up — worth mentioning if
  a reviewer's first request seems to hang.
- Disk storage is ephemeral: files written to `output/` are wiped on every
  redeploy or restart. Fine for a demo where each request generates its own
  `.docx` on the spot, but not a place to persist documents long-term — see
  the S3/blob storage note below for the real fix.

## 8. Scaling Notes (not required, but worth knowing)

- Section generation currently happens sequentially in a `for` loop; those
  calls are independent and would parallelize cleanly with
  `asyncio.gather` + an async HTTP client for faster response times on
  longer documents.
- The reflection step is currently capped at one repair pass; a stricter
  version could loop until `passed=True` or a max-iteration budget, at the
  cost of latency and API spend.
- Swapping `output/` for S3/blob storage and returning a signed URL instead
  of `FileResponse` would be the natural next step for a multi-instance
  deployment.
