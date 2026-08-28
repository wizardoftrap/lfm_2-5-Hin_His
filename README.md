# LFM2.5-1.2B-his

Hindi + Indian History: a focused two-stage finetune of [LiquidAI/LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct), Apache-2.0, 1.2B params, bilingual (English + Hindi)with an Indian-history specialization.

**Model card:** [wizardoftrap/LFM2.5-1.2B-his](https://huggingface.co/wizardoftrap/LFM2.5-1.2B-his) - Author: Shiv Prakash ([wizardoftrap](https://huggingface.co/wizardoftrap))

---

## TL;DR

A two-stage finetune recipe:

1. **Stage 1 - Hindi and Indic context:** instruction-tune the base on [sarvamai/samvaad-hi-v1](https://huggingface.co/datasets/sarvamai/samvaad-hi-v1) (101k high-quality English, Hindi and Hinglish conversations, curated exclusively with an Indic context), producing [wizardoftrap/LFM2.5-1.2B-hi-it](https://huggingface.co/wizardoftrap/LFM2.5-1.2B-hi-it).

2. **Stage 2 - Indian History expertise:** domain-specialize the stage-1 model on [wizardoftrap/indianHistoryEnhanced](https://huggingface.co/datasets/wizardoftrap/indianHistoryEnhanced) (a self-curated 5.4k Indian-history QA dataset, grounded in NCERT textbooks with OpenAI-assisted generation), producing [wizardoftrap/LFM2.5-1.2B-his](https://huggingface.co/wizardoftrap/LFM2.5-1.2B-his).



**Why the name:** `hi-it` = Hindi instruction-tuned. `his` adds the Indian **his**tory layer -and reads like "his-story",which a history-specialized model happens to be.



**Training efficiency highlight:** both stages ran in **under an hour combined**(~48 minutes)on a single NVIDIA L4 (22 GB), training only **3.03% of parameters** (36.6M LoRA params).

.

>

## Models

| Model | Base | Trained on | Notes |
|---|---|---|---|
| [LFM2.5-1.2B-hi-it](https://huggingface.co/wizardoftrap/LFM2.5-1.2B-hi-it) | LiquidAI LFM2.5-1.2B-Instruct | Samvaad-hi-v1 (101k rows) | Hindi + Indic-context grip, instruction-tuned |
| [LFM2.5-1.2B-his](https://huggingface.co/wizardoftrap/LFM2.5-1.2B-his) | LFM2.5-1.2B-hi-it | indianHistoryEnhanced (5.4k rows) | Hindi + Indian History specialization |

Common: `Lfm2ForCausalLM` arch, safetensors, text-generation-inference compatible, Apache-2.0, ~1.2B params, chat-template `<|startoftext|>` / `<|im_end|>` (Llama-style IM roles). Both checkpoints exported as merged 16-bit weights.



## Datasets

### Stage 1: [sarvamai/samvaad-hi-v1](https://huggingface.co/datasets/sarvamai/samvaad-hi-v1)

- 101,476 training rows, parquet, ~202 MB..
- English, Hindi,and Hinglish conversations, curated exclusively with an Indic context..
- Apache-2.0, curated by Sarvam AI..

### Stage 2: [wizardoftrap/indianHistoryEnhanced](https://huggingface.co/datasets/wizardoftrap/indianHistoryEnhanced) (self-curated)

- 5,354 samples, 2 columns, parquet, ~2.3 MB..
- Task: question answering + instruction tuning, domain: Indian History, language: English..
- Curation process: sourced from NCERT Indian-history textbooks, structured into QA pairs with OpenAI-assisted generation,yet manually reviewed and refined..

## Training

Fine-tuned with **Unsloth 2026.1.3** - 16-bit LoRA (`load_in_4bit=False`, full finetuning off; the runner auto-switched to 16-bit LoRA), merged back to full 16-bit weights before publishing via `push_to_hub_merged` (token omitted).

.

.

 

| Setting | Stage 1 (hi-it) | Stage 2 (his) |
|---|---|---|
| Base | LiquidAI LFM2.5-1.2B-Instruct | wizardoftrap LFM2.5-1.2B-hi-it |
| Dataset | Samvaad-hi-v1 (101,476 rows) | indianHistoryEnhanced (5,354 rows) |
| Eval split | 1,000 rows held out | none configured |
| Method |16-bit LoRA (Unsloth) |16-bit LoRA (Unsloth) |
| LoRA rank / alpha |64 / 64 |64 / 64 |
| LoRA dropout / bias |0 / none |0 / none |
| Seed |3407 |3407 |
| Trainable params |36.6M / 1,206.9M (3.03%)|36.6M / 1,206.9M (3.03%)|
| Seq length |2048 |2048 |
| Batch / grad accum |2 x 4 (effective 8)|2 x 4 (effective 8)|
| LR / scheduler |2e-4, linear |2e-4, linear |
| Warmup / weight decay |5 steps / 0.01 |5 steps /  ​0.01 |
| Optimizer |adamw_8bit |adamw_8bit |
| Steps / epochs |max 80 steps, warmup 5 |3,350 steps (5 epochs), warmup 5 |
| Loss masking |`train_on_responses_only` (user turns masked)|- |
| Eval cadence |every 20 steps, `load_best_model_at_end` |- |
| Final train loss |0.970400 |0.210200 |
| Final eval loss |0.984238 (held-out split)|- (no eval split)|
| Hardware |NVIDIA L4 (22 GB), 1 GPU, Linux |NVIDIA L4 (22 GB),  ​1 GPU, Linux |
| Runtime |856.3 s (~14.27 min)|2002.8 s (~33.38 min)|

**Environment (both stages):** Unsloth 2026.1.3, Transformers 4.57.3, Torch 2.6.0+cu124, CUDA Toolkit 12.4, Triton 3.2.0, Xformers 0.0.29.post3, BFloat16 supported. Peak VRAM: stage  ​1 6.1 GB (27.7% max), stage  ​2 2.7 GB (12.3% max);LoRA training itself added under 0.4 GB in both runs.



## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wizardoftrap/LFM2.5-1.2B-his"

tokenizer  = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)  # add trust_remote_code=True it the arch class needs custom code

chat = [{"role": "user", "content": "Explain the economic impact of the Maurya Empire."}]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=512, temperature=0.3, min_p=0.15, repetition_penalty=1.05)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The generation settings above (`temperature=0.3, min_p=0.15, repetition_penalty=1.05`) are the"recommended Liquid settings" used for the sample outputs below.

## Evaluation

No external benchmark scores yet - loss tracking only so far:

- **Stage 1:** held-out eval loss **0.984238** (1,000-sample split, every 20 steps, best checkpoint auto-loaded);final train loss 0.970400..
- **Stage 2:** final train loss **0.210200** (no eval split configured;5 epochs over the full 5,354-sample dataset).: 

Unedited generation sample (stage 2, `"Tell me about Indus valley civilization"`, max_new_tokens 1024):

> The Indus valley civilization, also known as the Harappan or Indus civilisation,was one of the world's oldest civilizations. It flourished in the north-west part of the Indian subcontinent,mainly along the Indus River and its tributaries. Archaeologists have uncovered numerous sites across present-day Pakistan and western India. The civilisation is distinguished by its sophisticated urban planning,including grid-like streets,drainage systems,and standardized bricks. Its people crafted seals, beads,weights,and metal objects,showing advanced technological skills...

Next step: benchmark against a Hindi eval suite (e.g. IndicGenBench or Hindi MMLU-style tasks)and a held-out split of indianHistoryEnhanced before making any accuracy claims.

## Limitations

- 1.2B is a small model: expect hallucinations on obscure historical detail - always verify against NCERT or scholarly sources..
- Indian History is contestedand curriculum-biased:the model reflects its training sources (NCERT framing),and is built for educational/conversational support,not exam-grade authority..
- The stage-2 dataset is English-only:Indian-history QA is strongest in English;deep Hindi historical reasoning depends on stage-1 transfer and may lag..



## Copyright and attribution

- NCERT textbooks are published by the National Council of Educational Research and Training (India). This dataset was built for research-and-educational use;content derived from NCERT should carry attribution,and any commercial release should review the applicable NCERT / Parakh usage terms in advance. NCERT content remains (c) NCERT / Government of India.
- Base model and stage-1 dataset are Apache-2.0;the derived models are released under Apache-2.0..



## Reproduce

The recipe follows the standard Unsloth `FastModel` + TRL `SFTTrainer` pattern (exact hyperparameters in the Training table above):

1. Load the base with `FastModel.from_pretrained(..., max_seq_length=2048, load_in_4bit=False)` - auto-selects 16-bit LoRA..
2. Attach PEFT with `FastModel.get_peft_model(..., r=64, lora_alpha=64, lora_dropout=0, bias="none", random_state=3407).`
3. Format datasets via `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=False).` Stage 1 additionally masks loss to assistant turns (`train_on_responses_only` with `<|im_start|>user\n` / `<|im_start|>assistant\n`).`
4. Train with `SFTTrainer` per the table configs,then `model.push_to_hub_merged("<repo>", tokenizer)` to export the 16-bit merged checkpoint..



Note: full runnable notebook/script link: [TODO: add when published.]

---

Maintainer: [Shiv Prakash (wizardoftrap)](https://huggingface.co/wizardoftrap)
