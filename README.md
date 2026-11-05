# medscribe

Clinical speech-to-note on **PriMock57** (57 mock GP consultations, CC BY 4.0), in two parts:

1. **ASR fine-tuning (PyTorch + Hugging Face):** LoRA fine-tuning of `openai/whisper-small` on
   doctor/patient consultation audio. On held-out consultations WER drops from **15.2% to 9.5%**
   (paired 95% CI for the change: −7.0 to −4.4 points).
2. **Multi-agent note writer, built from scratch (no LangGraph/LangChain):** planner → speaker-role
   labelling → entity extractor → SOAP writer → **grounding verifier** → bounded revise loop.
   Typed messages, per-agent traces, timeouts, token/step/time budgets and failure injection.
   It is compared against a single-agent baseline on note quality, hallucination, latency and
   tokens, using free local LLMs only (Ollama).

Every number below comes from a file in [`results/`](results/) produced by the commands shown.

---

## Results

### 1. ASR: whisper-small + LoRA on held-out consultations

**LoRA fine-tuning cut WER from 15.2% to 9.5% (−5.7 points, 95% CI [−7.0, −4.4]; −38% relative)
on the full held-out test split.** About 2 of those points come from the model learning the
transcribers' spelling conventions (see the OK/okay row); the remaining gain is 3.6 points.

**Download the adapter:** [`whisper-small-primock57-lora.zip`](https://github.com/vik-bit/medscribe/releases/tag/v0.1.0)
(28 MB, release v0.1.0, with a model card and SHA-256 checksums). After unzipping, run
`python -m medscribe asr-eval --split test --adapter whisper-small-primock57-lora`.

Test set: all usable utterances of the 12 held-out consultations (`results/splits.json`; 1,157
utterances, 72 min, 12,347 reference words). Utterances marked unintelligible or inaudible by the
transcribers are excluded. Text is normalised with Whisper's `EnglishTextNormalizer`. CIs resample
whole consultations, not utterances; the difference is a paired bootstrap on the same utterances.
The adapter was selected on the **dev** split; test was used once, for the numbers below.

| | Zero-shot whisper-small | + LoRA (this repo) | Change |
|---|---|---|---|
| **WER** [95% CI] | 15.2% [13.5, 16.7] | **9.5%** [8.0, 11.0] | **−5.7 pts** [−7.0, −4.4] |
| **CER** [95% CI] | 9.5% [8.3, 10.7] | **5.5%** [4.6, 6.3] | −4.1 pts |
| WER, doctor channel (594 utts) | 14.1% [11.0, 17.2] | 7.7% [5.6, 10.1] | −6.4 pts |
| WER, patient channel (563 utts) | 17.0% [14.2, 20.5] | 12.6% [10.3, 15.4] | −4.4 pts |
| WER if "OK" ≡ "okay" | 12.9% [11.2, 14.6] | 9.3% [8.0, 10.7] | −3.6 pts |
| Word errors: sub / del / ins | 1,008 / 690 / 171 | 697 / 243 / 226 | |

Sources: `results/asr_test_base.json`, `asr_test_lora.json`, `asr_test_compare.json`,
`asr_test_*_errors.json`, `asr_train_log.json`. Per-utterance predictions are in the `.jsonl` files.

What changed:

* **Transcription conventions:** 264 of the baseline's errors were the transcribers' "OK" written
  as "okay". After fine-tuning only 20 such errors remain, now in the other direction.
* **Fewer dropped words:** deletions fell from 690 to 243. Zero-shot Whisper tends to "clean up"
  disfluent speech (repetitions, false starts); the references are verbatim, and the fine-tuned
  model is closer to verbatim. Insertions rose slightly (171 → 226).
* **Doctors improved more than patients** (−6.4 vs −4.4 points). Doctors repeat across
  consultations (7 clinicians in the whole dataset), patients do not, so part of the doctor gain
  may be speaker adaptation. Expect less on unseen clinicians.
* **Remaining errors** are mostly short function words and near-synonyms (*yep/yeah*,
  *alright/right*). The one recurring medical-term error is *Ventolin → "ventilin"* (5×).

**Training** ([`src/medscribe/asr/train.py`](src/medscribe/asr/train.py)) is a plain PyTorch
loop. It covers PEFT LoRA (r=32, α=64 on q/k/v/out projections: 7.1M trainable params, 2.8% of
the model), on-the-fly log-mel features, label masking, fp16 autocast with grad scaling,
warmup + linear decay, gradient clipping, and best-adapter selection on dev WER. Run: 3 epochs
over 3,547 train utterances (3.9 h of audio), batch 16, lr 1e-3, 663 steps, 27.5 min on a free
Colab GPU via [`notebooks/whisper_lora_kaggle.ipynb`](notebooks/whisper_lora_kaggle.ipynb).
Training loss went from 1.1 to 0.07. Dev WER on 300 dev utterances by step: 18.2% (0) → 11.2%
(200) → 10.3% (400) → **9.6% (600, selected)** → 13.3% (663, the final step). The final-step rise
is why the adapter is chosen on dev rather than taken from the end of training.

I checked reproducibility locally: the adapter, run on CPU on 40 test utterances, produced
word-for-word the same transcripts as the Colab GPU run for 39 of 40. A separate earlier CPU run of
the zero-shot model on a seeded 293-utterance subset gave 15.4% [13.2, 17.6]
(`results/asr_test300_base_cpu.json`), consistent with the full-split 15.2%.

### 2. Notes: single agent vs multi-agent

Setup: the 12 held-out test consultations; every pipeline gets the same diarized transcript
(speakers anonymised to `SPEAKER_A`/`SPEAKER_B`, about 1.5k words); LLM `qwen2.5:3b` (Q4, Ollama,
CPU, temperature 0); prompts were tuned on one **dev** consultation only. `single+verify` is the
*same* single-agent output passed through the verifier, which isolates what the verifier alone
contributes. Means with 95% bootstrap CIs over consultations. Source: `results/notes_test.json`
(`python -m medscribe notes --split test --judge ollama:qwen2.5:3b`).

| Pipeline | Note token-F1 ↑ | Highlight recall ↑ | Ungrounded, lexical ↓ | Unsupported, LLM judge ↓ | Claims / note | Latency (s) | Prompt tokens | Output tokens | LLM calls |
|---|---|---|---|---|---|---|---|---|---|
| single | **0.302** [0.275, 0.330] | **0.274** [0.087, 0.486] | 0.357 [0.258, 0.461] | 0.571 [0.518, 0.625] | 14.3 | **222** | **3,285** | **568** | 1.0 |
| single + verify | 0.262 [0.214, 0.308] | 0.174 [0.024, 0.353] | **0.000** | **0.526** [0.484, 0.568] | 9.2 | 222 | 3,285 | 568 | 1.0 |
| multi-agent | 0.247 [0.197, 0.297] | 0.189 [0.052, 0.346] | **0.000** | 0.547 [0.465, 0.630] | 6.4 | 363 | 16,982 | 1,177 | 5.4 |
| *clinician note (calibration)* | – | – | 0.436 [0.325, 0.546] | 0.613 [0.568, 0.660] | 21.7 | – | – | – | – |

Paired differences vs `single` (same consultations):

| | Δ token-F1 | Δ unsupported (judge) | Δ latency | Δ prompt tokens |
|---|---|---|---|---|
| single + verify | −0.040 [−0.073, −0.010] | −0.045 [−0.115, +0.026] | +0.1 s | 0 |
| multi-agent | −0.055 [−0.108, −0.007] | −0.024 [−0.134, +0.077] | +141 s [+99, +188] | +13,698 (5.2×) |

**Honest reading:**

* **With a 3B model, the multi-agent pipeline is not better than one well-prompted call.** It is
  worse on note F1 (the CI excludes 0), costs 5.2× the prompt tokens and 1.6× the latency, and
  shows no significant improvement in judge-rated support.
* **What does help is the verifier, and it needs no extra LLM calls.** It takes the lexical
  ungrounded-claim rate from 36% to 0% for +0.1 s, at a cost of −0.04 F1 (dropped claims also
  lose recall). That 0% is close to true by construction, because the verifier and this metric use
  the same kind of lexical test. That is why the citation-blind LLM judge is reported next to it.
* **The small model's citations are mostly wrong, but its facts mostly aren't.** Of 172
  single-agent claims, only 26 (15%) cited utterances that support them. 84 (49%) were supported
  elsewhere in the transcript, and the verifier re-cited them. 62 (36%) had no lexical support
  anywhere and were dropped. Without re-citation, a citation-checking verifier would mostly be
  measuring citation accuracy. On the dev consultation it cut F1 from 0.40 to 0.14.
* **Where multi-agent loses claims:** its final notes have 6.4 claims vs 14.3. Its writer's first
  drafts average 460 output tokens (single agent: 568), and the verifier rejected 48 claims on the
  first pass, and after the revise loop (1.7 rounds on average) 16 were still unsupported and
  dropped. Per agent and per run: writer + revisions 173 s / 8.6k prompt tokens, extractor 170 s,
  planner 13 s, speaker 7 s.
* **Speaker roles were correct in 12/12** consultations. The LLM planner chose to skip the
  speaker agent in 2 of 12 runs and the extractor in 1; the question-count fallback still got the
  roles right.
* **Both grounding metrics are strict, and the judge is noisy.** The clinicians' own notes score
  44% "ungrounded" lexically, because of shorthand and inferred impressions, and 61% "unsupported"
  by the qwen2.5:3b judge. That judge also rejected plainly correct sentences such as
  "No blood in stool". Treat the judge column as a noisy relative signal, not a hallucination
  rate; its between-pipeline differences are within noise.
* Note token-F1 against free-text clinician notes is low for every system (about 0.25–0.30),
  because the clinicians write in dense shorthand. It is useful for comparing pipelines, not as
  an absolute quality score.

**Failure injection** (`--fault-rate 0.2`: each LLM call independently has a 20% chance of a
timeout, truncated JSON or an empty reply; first 4 test consultations; `results/notes_test_fault20.json`):

| Pipeline | Runs completed | Faults injected | Token-F1 (clean → faulty, same 4) | Latency s (clean → faulty) |
|---|---|---|---|---|
| single | 3 / 4 | 2 (both in one run, which exhausted its single retry) | 0.287 → 0.225 (failed run scores 0) | 205 → 298 |
| multi-agent | **4 / 4** | 7 (incl. 2 timeouts in one run, where the planner fell back to the default plan) | 0.237 → 0.200 | 355 → 470 |

Degradation behaved as designed: every fault was retried or absorbed by a fallback, and no run
crashed. The multi-agent pipeline's per-step fallbacks kept all 4 runs alive, while one
single-agent run produced no note. But quality still dropped. A retried call sees a modified
prompt and can return a different (here, shorter) note. n = 4, so this shows behaviour under
failure; it is not a precise estimate.


**Limitations:** n = 12 consultations, so the CIs are wide. One small local model (no 8B/70B
comparison was run). The generator and the judge are the same model family. The lexical metrics
miss paraphrase. Results may differ with a stronger model, where decomposition and revision
are more likely to pay off; `--llm groq:...` runs the same experiment on Groq's free tier.


---

## How the multi-agent pipeline works

```
diarized transcript (SPEAKER_A / SPEAKER_B, utterance ids)
        │
     planner ──────────── picks specialists, presenting complaint   (fallback: default plan)
        │
     speaker agent ────── maps clusters → DOCTOR / PATIENT          (fallback: question-count heuristic)
        │
     extractor ────────── symptoms, meds, history, plan items + evidence ids   (fallback: none)
        │
     writer ───────────── SOAP note as claims, each citing utterance ids
        │
     verifier (deterministic) ─ per claim: cited turns support ≥50% of its content words and
        │                      every number it states? If not, does any 4-turn window? (re-cite)
        │                      Otherwise reject.
        ├── rejected claims + reasons → writer revises or drops   (≤2 rounds, inside budget)
        │
     final note: verified claims only
```

* **Typed message passing:** every hop is a pydantic model (`agents/messages.py`). Its JSON
  schema goes to Ollama as a structured-output grammar, with `maxItems` bounds so a small model
  cannot loop inside a list. A malformed reply is a validation error that gets retried, and the
  retry is charged to the budget.
* **Budgets and timeouts:** `Budget(max_tokens, max_steps, max_seconds)` is checked before every
  LLM call, and each call has its own timeout. When the budget runs out, the orchestrator stops
  revising and keeps only verified claims (`status=budget_exhausted`).
* **Traces:** every call records agent, outcome (`ok` / `parse_error` / `timeout` / `error`),
  tokens and latency. The per-agent breakdown is saved with every note in `runs/`.
* **Failure injection:** `FaultInjectingLLM` injects timeouts, truncated JSON and empty replies
  at a seeded rate. Unit tests check that each failure takes the documented fallback.
* **Why the verifier is not an LLM:** the component that guards against hallucination should not
  be able to hallucinate itself. Its "re-cite" step exists because of a measured behaviour (see
  findings).
* **Prompt layout:** all agents share one system prompt and start with the same transcript block,
  so Ollama's KV cache lets later agents skip most of the prefill. Token counts are still the full
  prompt length, because Ollama reports them that way, so the cost comparison is not flattered.

---

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[asr,dev]"
bash scripts/fetch_primock57.sh    # git-lfs optional; verifies sha256; --no-audio for notes only
python -m medscribe splits

# ASR fine-tuning + before/after eval: run notebooks/whisper_lora_kaggle.ipynb on a free
# Colab/Kaggle T4 (~1 h). The same steps by hand, on any CUDA machine:
python -m medscribe asr-eval  --split test --out results/asr_test_base.json --preds results/asr_test_base.jsonl
python -m medscribe asr-train --epochs 3 --batch-size 16 --lr 1e-3 --out checkpoints/whisper-small-lora
python -m medscribe asr-eval  --split test --adapter checkpoints/whisper-small-lora/best \
  --out results/asr_test_lora.json --preds results/asr_test_lora.jsonl
python -m medscribe asr-compare --a results/asr_test_lora.jsonl --b results/asr_test_base.jsonl

# CPU-only: zero-shot baseline on a seeded subset (≈30 min on a laptop)
python -m medscribe asr-eval --split test --max-utts 300 \
  --out results/asr_test300_base_cpu.json --preds results/asr_test300_base_cpu.jsonl

# Notes (needs a local Ollama with qwen2.5:3b; or --llm groq:llama-3.1-8b-instant with GROQ_API_KEY)
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec ollama ollama pull qwen2.5:3b
python -m medscribe notes --split test --judge ollama:qwen2.5:3b \
  --runs runs/notes_test --results results/notes_test.json

pytest -q && ruff check .                             # what CI runs (no models, no network)
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs lint and unit tests on Python
3.10 and 3.12, using a hand-written synthetic consultation and a scripted LLM. No weights, GPU,
dataset or network LLM are needed.

## Repo map

```
src/medscribe/
  data/primock.py        TextGrid parser, consultations, diarized view, seeded split
  asr/                   segments, WER/CER + cluster bootstrap, eval, LoRA training loop
  agents/                messages, LLM backends + fault injection, runtime, specialists, orchestrator
  eval/                  note metrics, LLM judge, experiment runner, Markdown report
  grounding.py text.py stats.py
notebooks/whisper_lora_kaggle.ipynb
results/                 every number in this README; results/runs/ = per-note outputs + agent traces
tests/                   35 unit tests, synthetic fixture
```

## Data and licence

PriMock57: Korfiatis et al., *PriMock57: A Dataset Of Primary Care Mock Consultations*
(ACL 2022), CC BY 4.0, https://github.com/babylonhealth/primock57 (pinned commit in
`scripts/fetch_primock57.sh`). The data is not redistributed here; it is fetched by the
script. The patients are actors working from case cards, so the dataset contains no real
patient data. Code: MIT.
