# healing.py
# 드랍 이후 splice 경계 보정을 위한 짧은 LoRA 미세튜닝 (선택)

from typing import List

def try_lora_heal(model, tokenizer, train_texts: List[str], device, r: int = 16, alpha: int = 16, lr: float = 1e-4, steps: int = 100):
    try:
        from peft import LoraConfig, get_peft_model, TaskType
        import torch
        from torch.utils.data import DataLoader, Dataset
        
        class TxtDS(Dataset):
            def __init__(self, texts): self.texts = texts
            def __len__(self): return len(self.texts)
            def __getitem__(self, i):
                enc = tokenizer(self.texts[i], return_tensors="pt", truncation=True, max_length=getattr(model, "seqlen", 2048))
                return {k: v.squeeze(0) for k, v in enc.items()}

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=r, lora_alpha=alpha, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        )
        model = get_peft_model(model, lora_cfg).to(device)
        model.train()

        ds = TxtDS(train_texts)
        dl = DataLoader(ds, batch_size=1, shuffle=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss()

        for step, batch in enumerate(dl):
            if step >= steps: break
            for k in batch: batch[k] = batch[k].to(device)
            out = model(**batch)
            logits = out.logits[:, :-1, :].contiguous()
            labels = batch["input_ids"][:, 1:].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if (step + 1) % 20 == 0:
                print(f"[LoRA heal] step {step+1}/{steps} loss={loss.item():.4f}")

        model.eval()
        return model
    except Exception as e:
        print(f"[LoRA heal] skipped (reason: {e})")
        return model
