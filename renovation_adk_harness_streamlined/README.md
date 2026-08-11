# Streamlined Renovation ADK Harness

## A100 LoRA configuration in this package

This version is configured for the tested local endpoint:

```text
Qwen-Image-Edit-2511 + step-2880 LoRA
        ↓
FastAPI: http://127.0.0.1:8001/edit
        ↓
ADK renovation agent
```

The initial generation behavior is:

1. **LoRA HTTP result is the default and active image.**
2. **Gemini creates an optional initial comparison image only.**
3. A Gemini comparison failure does not replace or invalidate the LoRA result.
4. Later edit requests continue from the active LoRA image through the existing
   Qwen Replicate iteration path.

Before starting ADK, confirm the LoRA endpoint:

```bash
curl -fsS http://127.0.0.1:8001/health | python -m json.tool
```

From the project root, test the exact client contract:

```bash
python scripts/check_lora_api.py \
  --image images/room_check.png \
  --output images/generated/lora_api_test.png
```

Then start the agent:

```bash
python -m pip install -e .
adk web .
```

Required `.env` values are already present in `.env.example`:

```env
QWEN_LORA_BACKEND=http
QWEN_LORA_API_URL=http://127.0.0.1:8001/edit
QWEN_LORA_HEALTH_URL=http://127.0.0.1:8001/health
QWEN_LORA_HEALTHCHECK_BEFORE_GENERATION=true
QWEN_LORA_SEND_REFERENCE_IMAGE=false
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=4.0
GENERATION_TIMEOUT_SECONDS=1200
GENERATE_INITIAL_GEMINI_COMPARISON=true
ENABLE_INITIAL_GEMINI_FALLBACK=false
```

`QWEN_LORA_SEND_REFERENCE_IMAGE=false` matches the validated LoRA inference
contract: one source image plus the detailed prompt. The selected reference is
still analyzed by Gemini and encoded into that prompt.

Set `REPLICATE_API_TOKEN` privately only when follow-up Replicate iterations are
needed. No API token is included in this archive.

---

This package is the **streamlined version** of your earlier sophisticated ADK project.
It keeps the strengths of the earlier system—stateful orchestration, robust multi-turn handling,
clear tool boundaries, and deterministic state transitions—but removes the product-retrieval stack,
Browserbase flows, reranking, bundle planning, and separate sub-pipelines.

It implements **only** the revised room-renovation workflow shown in your updated diagram:

1. **User uploads a room photo**
2. **Vector index search** using **Gemini Embedding 2** on:
   `gs://adsp-s26-reccys-bucket/living-room-renovation-index/after_orig`
3. **User picks one of the top 3 references**
4. **Gemini 3 reasoning** creates a structured renovation plan
5. **Local Qwen edit + LoRA FastAPI** creates the default initial renovated room
   - Gemini may generate a separate initial comparison image
   - the LoRA output remains the active result
6. **Qwen Replicate API** handles later iterative edits

---

## What changed versus the earlier larger system

### Removed
- Browserbase product retrieval
- Ecommerce candidate retrieval
- Cross-encoder reranking
- Bundle planner / product bundle assembly
- Critic / verification around product cards
- Separate browsing and recommendation pipelines

### Kept in spirit
- ADK-native, stateful orchestration
- Tool-first design with explicit state transitions
- Human-in-the-loop interaction
- Multi-turn handling for complex conversational flows
- Deterministic control of when generation vs iteration is allowed

---

## Project layout

```text
renovation_adk_harness_streamlined/
├── renovation_agent/
│   ├── __init__.py
│   ├── agent.py                    # single conversational controller
│   ├── config.py                   # settings loaded from environment / .env
│   ├── schemas.py                  # Pydantic state objects
│   ├── tools.py                    # workflow tools + session-state transitions
│   └── services/
│       ├── __init__.py
│       ├── gemini_reasoner.py      # Gemini 3.1 plan generation
│       ├── gcs_store.py            # stable image persistence + preview URLs
│       ├── image_io.py             # local / gs:// / https image loading
│       ├── metadata_catalog.py     # Vector Search datapoint -> image metadata mapping
│       ├── qwen_backends.py        # initial Qwen+LoRA and iteration backends
│       └── vector_search.py        # Gemini Embedding 2 + Vertex AI Vector Search
├── scripts/
│   ├── check_lora_api.py           # health check + direct generation smoke test
│   ├── inspect_vector_index.py
│   ├── search_reference_smoke_test.py
│   └── mock_qwen_lora_server.py
├── tests/
├── .env.example
├── pyproject.toml
├── requirements.txt
└── Dockerfile
```

---

## Conversation design

The system is intentionally implemented as a **single stateful controller agent** plus deterministic tools.
This is the right fit for your revised flow because the user journey is linear but still conversationally complex. It also makes it easy to swap the first-result generator between Qwen+LoRA and Gemini Flash Image without changing the conversational state machine.

### The controller handles
- image-path detection
- reference presentation
- flexible selection parsing (`1`, `option 2`, `reference 3`, exact reference ID)
- recap / status requests
- switching references after a prior generation
- user refinements before first render
- iterative edits after generation
- reset / start-over requests

### The controller does **not** do
- shopping
- browsing
- product recommendation
- parallel retrieval pipelines
- separate reranking branches

---

## Core workflow state machine

### Stage 1 — photo_registered
After `start_project(...)`, the source room is persisted to a stable GCS location and the project state is reset.

### Stage 2 — references_ready
After `search_references(...)`, the system stores the top reference candidates from the vector index.

### Stage 3 — reference_selected
After `select_reference(...)`, the chosen candidate becomes the active style reference.

### Stage 4 — plan_ready
After `create_renovation_plan(...)`, Gemini produces a structured design plan and a generation-ready Qwen prompt.

### Stage 5 — generated
After `render_renovation(...)`, the first renovated room is created from the **original source room**.

### Stage 6 — iterating
After `iterate_renovation(...)`, subsequent edits are applied to the **latest generated image**.

---

## Environment setup

```bash
cd renovation_adk_harness_streamlined
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
cd renovation_adk_harness_streamlined
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Authenticate for Google Cloud:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project adsp-s26-reccys
gcloud config set project adsp-s26-reccys
```

---

## Important defaults already wired in

```dotenv
GOOGLE_CLOUD_PROJECT=adsp-s26-reccys
VECTOR_LOCATION=us-central1
VECTOR_DATA_PREFIX=gs://adsp-s26-reccys-bucket/living-room-renovation-index/after_orig
VECTOR_INDEX_DISPLAY_NAME=living-room-renovation-index
OUTPUT_BUCKET=adsp-s26-reccys-bucket
OUTPUT_PREFIX=renovation-agent-outputs
```

If the code cannot auto-discover your deployed Vertex AI Vector Search resources, set:

```dotenv
VECTOR_INDEX_NAME=projects/adsp-s26-reccys/locations/us-central1/indexes/INDEX_ID
VECTOR_INDEX_ENDPOINT_NAME=projects/adsp-s26-reccys/locations/us-central1/indexEndpoints/ENDPOINT_ID
DEPLOYED_INDEX_ID=DEPLOYED_ID
EMBEDDING_DIMENSION=3072
```

---

## Metadata sidecar requirement

The catalog object for this project is:

```text
living-room-renovation-index/catalog.json
```

It is stored at:

```text
gs://adsp-s26-reccys-bucket/living-room-renovation-index/catalog.json
```


Vector Search returns nearest-neighbor IDs and distances, but not necessarily the actual image URIs required for display and selection.
So you should point the harness to a sidecar such as:

```dotenv
CATALOG_OBJECT_NAME=living-room-renovation-index/catalog.json
VECTOR_METADATA_URI=gs://adsp-s26-reccys-bucket/living-room-renovation-index/catalog.json
```

Recommended record format:

```json
{"id":"living_room_001","image_uri":"gs://adsp-s26-reccys-bucket/living-room-renovation-images/living_room_001.jpg","style":"warm minimalist","room_type":"living_room","caption":"Warm oak, cream boucle, low visual clutter"}
```

---

## Initial Qwen Edit + LoRA backend

Set one of the supported Qwen Edit + LoRA backends for the preferred first-result path. In addition, the harness can automatically fall back to Gemini 3.1 Flash Image if the LoRA backend is unavailable during testing.

### HTTP service

```dotenv
QWEN_LORA_BACKEND=http
QWEN_LORA_API_URL=http://127.0.0.1:8001/edit
```

### Vertex AI custom endpoint

```dotenv
QWEN_LORA_BACKEND=vertex_endpoint
QWEN_LORA_VERTEX_ENDPOINT=projects/adsp-s26-reccys/locations/us-central1/endpoints/ENDPOINT_ID
```

### Replicate-hosted model

```dotenv
QWEN_LORA_BACKEND=replicate_model
QWEN_LORA_REPLICATE_MODEL=OWNER/MODEL:VERSION
QWEN_LORA_SOURCE_FIELD=image
QWEN_LORA_REFERENCE_FIELD=reference_image
QWEN_LORA_PROMPT_FIELD=prompt
QWEN_LORA_EXTRA_INPUT_JSON={"lora_weights":"HOSTED_WEIGHTS_URL","lora_scale":1.0}
```

### Optional Gemini 3.1 Flash Image fallback

```dotenv
ENABLE_INITIAL_GEMINI_FALLBACK=true
GEMINI_IMAGE_FALLBACK_MODEL=gemini-3.1-flash-image
```

With this enabled, `render_renovation(...)` first tries the configured Qwen+LoRA backend. If that primary backend is unavailable or errors, the harness automatically retries the **first generated result** with Gemini 3.1 Flash Image. The returned tool payload exposes `fallback_used=true` and the primary error so the controller can report that the fallback path was used.

A mock local HTTP server is included for contract testing:

```bash
uvicorn scripts.mock_qwen_lora_server:app --host 0.0.0.0 --port 8001
```

---

## Iteration backend

Later conversational edits go through the Qwen Replicate API:

```dotenv
REPLICATE_API_TOKEN=r8_...
REPLICATE_ITERATION_MODEL=qwen/qwen-image-edit
REPLICATE_ITERATION_IMAGE_FIELD=image
REPLICATE_ITERATION_PROMPT_FIELD=prompt
REPLICATE_ITERATION_EXTRA_INPUT_JSON={}
```

---

## Run the harness

### ADK Web UI

```bash
adk web .
```

### ADK CLI

```bash
adk run renovation_agent
```

### API server

```bash
adk api_server --host 0.0.0.0 --port 8000 .
```

---

## Example conversation

### Turn 1

```text
image_path="/home/akgupta_uchicago_edu/shopping_agent/data/images/room.jpg"
Make it warm, modern, uncluttered, and slightly Japandi.
```

Agent behavior:
1. `start_project(...)`
2. `search_references(num_neighbors=3)`
3. returns the top 3 references and waits

### Turn 2

```text
Use option 2, but keep the room bright and don't make it too dark in wood tones.
```

Agent behavior:
1. `select_reference("2")`
2. `create_renovation_plan(user_notes="keep the room bright and don't make it too dark in wood tones")`
3. `render_renovation(seed=-1)` using the configured initial backend (Qwen+LoRA by default, or Gemini Flash Image for testing)

### Turn 3

```text
Make the sofa lighter beige, remove the small accessories on the side table, and keep everything else unchanged.
```

Agent behavior:
1. `iterate_renovation(edit_request=...)`

### Turn 4

```text
Use reference 1 instead, but keep the brighter palette.
```

Agent behavior:
1. `select_reference("1")`
2. `create_renovation_plan(user_notes="keep the brighter palette")`
3. `render_renovation(seed=-1)`

---

## Smoke tests

Inspect index discovery:

```bash
python scripts/inspect_vector_index.py
```

Run retrieval on a real room image:

```bash
python scripts/search_reference_smoke_test.py /absolute/path/to/room.jpg --top-k 3
```

---

## Why this design matches your revised system

The earlier project needed multiple pipelines because it was doing retrieval, reranking, bundle planning, and final recommendation generation.
Your revised room-renovation workflow is fundamentally different:

- one user image
- one vector-search retrieval step
- one human selection step
- one reasoning step
- one initial generation step
- optional iterative edits

So the cleanest implementation is a **single ADK conversation controller with strict tool gating**.
That preserves conversational sophistication without reintroducing unnecessary sub-agents or product-retrieval complexity.

---

## Fix for `No API key was provided` while using ADC

The ADK root LLM chooses the Gemini backend while the agent module is imported. The project now runs
`renovation_agent.bootstrap.bootstrap_environment()` before importing `google.adk`, which:

- loads the project `.env`
- forces Vertex AI / Gemini Enterprise Agent Platform mode
- exports the project and location into `os.environ`
- leaves authentication to Application Default Credentials (ADC)

This avoids the Gemini Developer API route that asks for an API key.

Use these `.env` values:

```dotenv
GOOGLE_CLOUD_PROJECT=adsp-s26-reccys
GOOGLE_CLOUD_LOCATION=global
GEMINI_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_GENAI_USE_ENTERPRISE=TRUE
```

On a local machine or a VM where you want user ADC:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project adsp-s26-reccys
```

On a Google Cloud Workbench/Compute Engine VM with an attached service account, ADC can use the
attached service account automatically. The service account still needs the required Vertex AI,
Vector Search, and GCS permissions.

Validate ADC and backend selection before starting ADK:

```bash
python scripts/check_adc_auth.py
```

Then run:

```bash
adk run renovation_agent
```

The deprecation or experimental-feature warning printed before `Running agent ...` is separate from
the authentication failure. The actual crash was caused by ADK selecting the API-key Gemini backend.

### Replicate authentication and Qwen image input

Gemini, Gemini Embedding, Vector Search, and GCS use Google ADC. Only the Replicate-backed Qwen
calls use `REPLICATE_API_TOKEN`.

Do not hardcode the token in source code. Put it in `.env` or inject it from a secret store:

```dotenv
REPLICATE_API_TOKEN=r8_...
REPLICATE_ITERATION_IMAGE_IS_LIST=true
QWEN_LORA_REPLICATE_IMAGE_IS_LIST=true
```

For Colab, load the token from Colab Secrets without writing the value into the notebook:

```python
import os
from google.colab import userdata

token = userdata.get("REPLICATE_API_TOKEN")
if not token:
    raise RuntimeError(
        "Add REPLICATE_API_TOKEN to Colab Secrets and enable notebook access."
    )
os.environ["REPLICATE_API_TOKEN"] = token
```

The Replicate adapter now constructs an authenticated client and invokes the model as:

```python
client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
output = client.run(
    MODEL_NAME,
    input={
        "image": [image_file],
        "prompt": prompt,
    },
)
```

The list wrapping is controlled through `REPLICATE_ITERATION_IMAGE_IS_LIST` and
`QWEN_LORA_REPLICATE_IMAGE_IS_LIST`, because different Replicate model schemas accept either one
file or a list of files.

---

## ADK `State` compatibility note

Recent ADK versions expose `tool_context.state` as a tracked `State` object rather than a normal Python `dict`.
It supports `get`, item assignment, `update`, and `to_dict`, but not `pop`.

The harness resets workflow state with:

```python
state.update({
    "project_id": None,
    "stage": "not_started",
    "source_image_uri": None,
    "reference_candidates": [],
    "selected_reference": None,
    "renovation_plan": None,
    "current_image_uri": None,
    "generation_history": [],
})
```

This avoids:

```text
AttributeError: 'State' object has no attribute 'pop'
```

## Vector index discovery troubleshooting

The configured GCS data prefix is:

```dotenv
VECTOR_DATA_PREFIX=gs://adsp-s26-reccys-bucket/living-room-renovation-index/after_orig
```

This GCS path identifies the data used to build the index; it is not itself the queryable Vertex AI index resource. The harness still resolves a Vertex AI `Index`, its deployed `IndexEndpoint`, and the deployed index ID.

List every visible index and endpoint under the configured project and region:

```bash
python scripts/list_vector_resources.py
```

Then copy the matching resource values into `.env` when auto-discovery is ambiguous:

```dotenv
VECTOR_INDEX_NAME=projects/adsp-s26-reccys/locations/us-central1/indexes/INDEX_ID
VECTOR_INDEX_ENDPOINT_NAME=projects/adsp-s26-reccys/locations/us-central1/indexEndpoints/ENDPOINT_ID
DEPLOYED_INDEX_ID=DEPLOYED_ID
EMBEDDING_DIMENSION=3072
```

The improved resolver matches parent and child GCS prefixes in either direction and automatically selects the index when the project contains exactly one Vector Search index.
