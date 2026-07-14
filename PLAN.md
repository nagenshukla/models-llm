# Civil Engineering / BIM Assistant — Local Fine-Tune → Azure AI Foundry

**Status:** Prototype / proof of concept
**Scope:** BIM & digital construction terminology (ISO 19650, IFC, COBie, CDE, LOD/LOI, BEP/EIR, clash detection, 4D/5D BIM, DfMA, digital twin, scan-to-BIM, common authoring/coordination tools)
**Constraint:** No proprietary source documents — synthetic data generated from a hand-curated domain knowledge base, not scraped or hallucinated by an LLM with no grounding.

## Approach

Local QLoRA fine-tune of a small open-weight instruct model, then push the merged model to Azure AI Foundry as a custom deployment. RAG is explicitly out of scope for this phase (no source corpus to retrieve over) but is the natural phase-2 addition once real project documents exist — see "Later" below.

**Base model:** `microsoft/Phi-4-mini-instruct` (3.8B). Rationale: small enough to QLoRA-tune on a single consumer GPU (≥8GB VRAM) or CPU-slow-but-workable, natively listed in the Azure AI Foundry model catalog (so the fine-tuned artifact deploys into a familiar serving stack), and Microsoft's own license/support story fits an MS-Foundry-hosted product better than a Llama/Mistral base. `Phi-3.5-mini-instruct` is the fallback if Phi-4-mini isn't available in your environment yet.

## Pipeline

1. **Knowledge base** (`data/kb_bim.py`) — ~50 hand-written BIM/digital-construction facts: term, acronym, category, definition, real-world example, related terms. This is the ground truth. Everything downstream is derived from it, so factual accuracy is controlled at the source instead of hoping an LLM generator doesn't hallucinate standards content.
2. **Synthetic generation** (`data/generate_synthetic_data.py`) — combinatorial template engine, no external LLM API call required (fully local, free, deterministic). Produces 5 question types per term: definition, persona-framed, comparison/relation, applied/workflow scenario, acronym expansion. ~350-450 examples before split.
3. **Split** — dedup, shuffle with fixed seed, 85/15 train/test, written as chat-format JSONL (`{"messages":[{"role":"system"...},{"role":"user"...},{"role":"assistant"...}]}`) — this format works for both local HF/TRL fine-tuning and Azure OpenAI-style fine-tuning if you ever switch base models.
4. **Local fine-tune** (`train/finetune_qlora.py`) — HF `transformers` + `peft` + `trl` QLoRA SFT script. **Windows note:** `bitsandbytes` 4-bit quantization and flash-attention are unreliable on native Windows — run this under WSL2 or a Docker container with CUDA, not directly in PowerShell.
5. **Eval** (`eval/`) — held-out test set, exact/rubric scoring against the KB (since we authored the ground truth, we can grade objectively instead of needing an LLM judge), plus a manual spot-check against a public glossary (buildingSMART / NIBS) to catch KB errors before they get baked into the model.
6. **Package & deploy** — merge LoRA adapter into base weights, register the merged model in the Azure AI Foundry model catalog/registry, deploy to a Managed Online Endpoint (GPU SKU, e.g. `Standard_NC6s_v3` or serverless if supported for Phi), wire it into a Foundry project.

## Later (not in this phase)

- Swap in a real corpus (project specs, BEPs, IFC exports, company standards) once available, and add a RAG layer in front of the fine-tuned model for factual grounding + citations — this is the recommended production hardening step before this goes past prototype.
- Expand scope beyond BIM into structural, codes/compliance, construction management if the prototype validates.
- Replace the objective KB-based eval with an LLM-judge pipeline once test volume outgrows manual grading.

## What's in this repo

```
models-llm/
  PLAN.md
  data/
    kb_bim.py                    # ground-truth knowledge base
    generate_synthetic_data.py   # generator + train/test split
    train.jsonl / test.jsonl     # generated output (chat format)
  train/
    finetune_qlora.py            # QLoRA SFT script (run under WSL2/Docker)
    requirements.txt
  eval/
    evaluate.py                  # scores fine-tuned model against test.jsonl
  deploy/
    README.md                    # Azure AI Foundry deployment steps
```
