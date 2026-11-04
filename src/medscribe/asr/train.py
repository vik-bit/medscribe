"""LoRA fine-tuning of Whisper on PriMock57 consultation audio (plain PyTorch loop).

Written as an explicit loop rather than ``Seq2SeqTrainer`` so every piece is
visible: on-the-fly log-mel features, label masking, fp16 autocast + grad
scaling on GPU, warmup/linear-decay schedule, gradient clipping, periodic dev
WER with best-adapter checkpointing, and a JSON training log.

Runs on a free Kaggle/Colab T4 (see ``notebooks/whisper_lora_kaggle.ipynb``);
it also runs on CPU, just slowly.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from medscribe.asr.evaluate import normaliser, score_all, subset, transcribe
from medscribe.asr.metrics import corpus_rates
from medscribe.asr.segments import Segment, check_audio, load_audio, split_segments

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


class SegmentDataset:
    def __init__(self, segs: list[Segment], processor, max_label_len: int = 448):
        self.segs, self.processor, self.max_label_len = segs, processor, max_label_len

    def __len__(self) -> int:
        return len(self.segs)

    def __getitem__(self, i: int):
        s = self.segs[i]
        feats = self.processor.feature_extractor(
            load_audio(s), sampling_rate=16_000, return_tensors="np"
        ).input_features[0]
        labels = self.processor.tokenizer(s.text).input_ids[: self.max_label_len]
        return feats, labels


def make_collate(processor, decoder_start_id: int):
    import numpy as np
    import torch

    def collate(batch):
        feats = torch.from_numpy(np.stack([b[0] for b in batch]))
        labels = [b[1] for b in batch]
        # The model prepends decoder_start itself when shifting labels right, so
        # drop it from the labels if the tokenizer already added it.
        labels = [ids[1:] if ids and ids[0] == decoder_start_id else ids for ids in labels]
        width = max(len(x) for x in labels)
        lab = torch.full((len(labels), width), -100, dtype=torch.long)
        for j, ids in enumerate(labels):
            lab[j, : len(ids)] = torch.tensor(ids)
        return feats, lab

    return collate


def main(args) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = split_segments(root, "train")
    dev = subset(split_segments(root, "dev"), args.max_dev_utts)
    check_audio(train + dev)

    processor = WhisperProcessor.from_pretrained(args.model)
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      target_modules=LORA_TARGETS, bias="none")  # fmt: skip
    model = get_peft_model(model, lora).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    print(
        f"train {len(train)} utts ({sum(s.duration for s in train) / 3600:.2f} h), dev {len(dev)}"
    )
    loader = torch.utils.data.DataLoader(
        SegmentDataset(train, processor),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if device == "cuda" else 0,
        collate_fn=make_collate(processor, model.config.decoder_start_token_id),
        drop_last=True,
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = args.max_steps or int(args.epochs * steps_per_epoch)
    warmup = max(1, int(0.05 * total_steps))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            weight_decay=0.01)  # fmt: skip
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, max(0.0, (total_steps - s) / (total_steps - warmup)))
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    norm = normaliser(processor)
    log: dict = {"args": vars(args) | {"func": None}, "train_utts": len(train), "dev_utts": len(dev),
                 "trainable_params": trainable, "steps": [], "evals": []}  # fmt: skip

    def dev_wer() -> float:
        model.eval()
        hyps = transcribe(processor, model, device, dev, batch_size=16, log=lambda *_: None)
        model.train()
        return corpus_rates(score_all(norm, dev, hyps))[0]

    best = dev_wer()
    log["evals"].append({"step": 0, "dev_wer": best})
    print(f"step 0 dev WER {best:.4f}")
    model.save_pretrained(out / "best")

    step, micro, t0 = 0, 0, time.perf_counter()
    model.train()
    while step < total_steps:
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(input_features=feats, labels=labels).loss / args.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            log["steps"].append({"step": step, "loss": loss.item() * args.grad_accum,
                                 "lr": sched.get_last_lr()[0],
                                 "elapsed_s": round(time.perf_counter() - t0, 1)})  # fmt: skip
            if step % 20 == 0:
                print(f"step {step}/{total_steps} loss {log['steps'][-1]['loss']:.4f} "
                      f"{time.perf_counter() - t0:.0f}s")  # fmt: skip
            if step % args.eval_every == 0 or step == total_steps:
                w = dev_wer()
                log["evals"].append({"step": step, "dev_wer": w})
                print(f"step {step} dev WER {w:.4f} (best {best:.4f})")
                if w < best:
                    best = w
                    model.save_pretrained(out / "best")
            if step >= total_steps:
                break
    log["best_dev_wer"] = best
    log["train_seconds"] = round(time.perf_counter() - t0, 1)
    log["device"] = device
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(f"done. best dev WER {best:.4f}; adapter at {out / 'best'}")
