# Install and run on `gpu-a100-80`

Project destination:

```text
/home/akgupta_uchicago_edu/interior_design_recsys/shopping_agent_v2/renovation_adk_harness_streamlined_catalog_id_fixed/renovation_adk_harness_streamlined
```

## 1. Confirm that the already-tested LoRA server is ready

```bash
curl -fsS http://127.0.0.1:8001/health | python -m json.tool
```

Do not start ADK until the response contains `"status": "ready"`.

## 2. Back up the current project

```bash
PROJECT="$HOME/interior_design_recsys/shopping_agent_v2/renovation_adk_harness_streamlined_catalog_id_fixed/renovation_adk_harness_streamlined"
cp -a "$PROJECT" "${PROJECT}_backup_$(date +%Y%m%d_%H%M%S)"
```

## 3. Extract the supplied archive and copy it over the project

Assuming the archive is uploaded to `~/interior_design_recsys/`:

```bash
cd "$HOME/interior_design_recsys"
rm -rf /tmp/shopping_agent_lora_ready
mkdir -p /tmp/shopping_agent_lora_ready
unzip -q shopping_agent_v2_lora_ready.zip -d /tmp/shopping_agent_lora_ready

PATCH="/tmp/shopping_agent_lora_ready/shopping_agent_v2/renovation_adk_harness_streamlined_catalog_id_fixed/renovation_adk_harness_streamlined"
rsync -av --delete \
  --exclude='images/generated/' \
  "$PATCH/" "$PROJECT/"
```

The archive contains a sanitized `.env`; no Replicate token is included.

## 4. Install the shopping-agent package

```bash
cd "$PROJECT"
python -m pip install -e .
```

This installation is for the ADK client. The Qwen base model and LoRA remain in
the separate `/home/jupyter` FastAPI runtime.

## 5. Test the exact HTTP client used by the agent

```bash
python scripts/check_lora_api.py \
  --image images/room_check.png \
  --output images/generated/lora_api_test.png
```

Expected result:

```text
Health: {... 'status': 'ready' ...}
Saved: .../images/generated/lora_api_test.png
```

## 6. Verify loaded settings

```bash
python - <<'PY'
from renovation_agent.config import get_settings
s = get_settings()
print('Backend:', s.qwen_lora_backend)
print('Edit URL:', s.qwen_lora_api_url)
print('Health URL:', s.qwen_lora_health_url)
print('Send reference image:', s.qwen_lora_send_reference_image)
print('Steps:', s.num_inference_steps)
print('CFG:', s.guidance_scale)
print('Gemini comparison:', s.generate_initial_gemini_comparison)
print('Gemini fallback:', s.enable_initial_gemini_fallback)
PY
```

Expected key values:

```text
Backend: http
Edit URL: http://127.0.0.1:8001/edit
Send reference image: False
Steps: 30
CFG: 4.0
Gemini comparison: True
Gemini fallback: False
```

## 7. Start ADK

```bash
adk web .
```

The first render returns:

- LoRA result: default/current image
- Gemini result: initial comparison only

`current_image_uri` always remains the LoRA output.

## 8. Optional follow-up iterations

Follow-up edits use the existing Replicate Qwen backend. Set the token only in
your shell or secret manager:

```bash
export REPLICATE_API_TOKEN='YOUR_NEW_TOKEN'
adk web .
```

Do not write the token into source control. Rotate any token that was previously
included in an uploaded `.env` file.
