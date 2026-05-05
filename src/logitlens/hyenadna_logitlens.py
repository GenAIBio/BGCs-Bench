import argparse
import os
import torch
from pathlib import Path
from typing import Tuple
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_hyenadna(checkpoint: str, offline: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    assert torch.cuda.is_available()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return model, tokenizer


def compute_log_likelihood_per_base(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Compute log probabilities
    logp_next = torch.log_softmax(logits, dim=-1)               # (L, V)
    # Shift so that tokens < n predict n
    shift_logp = logp_next[..., :-1, :].contiguous()            # (L-1, V)
    shift_labels = labels[..., 1:].unsqueeze(-1).contiguous()   # (L-1, 1)
    # Gather log-probability for the actual next token
    LL = torch.gather(shift_logp, -1, shift_labels).squeeze(-1) # (L-1)
    return LL


def run_logitlens(model: AutoModelForCausalLM,
                  tokenizer: AutoTokenizer,
                  in_fasta: str,
                  dtype_out: str,
                  output_dir: Path,
                  reverse_complement: bool = False):
    records = list(SeqIO.parse(in_fasta, "fasta"))
    if len(records) == 0:
        raise RuntimeError("No sequences found in FASTA.")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
    if dtype_out not in dtype_map:
        raise ValueError(f"Unsupported dtype_out: {dtype_out}")
    out_dtype = dtype_map[dtype_out]
    
    # Prepare to retrieve intermediate embeddings
    embeddings = {}
    handles = []
    def hook_fn(layer_name):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            embeddings[layer_name] = output.detach()
        return hook
    for i in range(8):
        handles.append(model.hyena.backbone.layers[i].register_forward_hook(hook_fn(f"hyena.backbone.layers.{i}")))
    
    if not reverse_complement:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.seq)
            input_ids = tokenizer(seq, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            output = model(input_ids)
            ll_scores = {}
            for i in range(8):
                if f"hyena.backbone.layers.{i}" not in embeddings:
                    continue
                t = embeddings[f"hyena.backbone.layers.{i}"].squeeze(0)  # [L,H]
                logits = model.lm_head(t).to(dtype=out_dtype)
                LL = compute_log_likelihood_per_base(logits, input_ids.to(int).squeeze(0))
                ll_scores[f"hyena.backbone.layers.{i}"] = LL.to(device="cpu", dtype=out_dtype)
            torch.save(ll_scores, output_dir / f"{header}.log_likelihood.pt")
    
    else:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.reverse_complement().seq)
            input_ids = tokenizer(seq, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            output = model(input_ids)
            ll_scores = {}
            for i in range(8):
                if f"hyena.backbone.layers.{i}" not in embeddings:
                    continue
                t = embeddings[f"hyena.backbone.layers.{i}"].squeeze(0)  # [L,H]
                logits = model.lm_head(t).to(dtype=out_dtype)
                LL = compute_log_likelihood_per_base(logits, input_ids.to(int).squeeze(0))
                ll_scores[f"hyena.backbone.layers.{i}"] = LL.to(device="cpu", dtype=out_dtype)
            torch.save(ll_scores, output_dir / f"{header}.log_likelihood.pt")
    
    # Remove hooks
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyenaDNA inference")
    parser.add_argument("--input-fasta-list", type=Path, required=True,
                        help="Input FASTA file list (one path per line).")
    parser.add_argument("--output-dir", default=Path("logitlens"), type=Path,
                        help="Output directory to save embeddings.")
    parser.add_argument("--checkpoint", default="LongSafari/hyenadna-large-1m-seqlen-hf",
                        help="Checkpoint name (LongSafari/hyenadna-large-1m-seqlen-hf, LongSafari/hyenadna-medium-160k-seqlen-hf, ...)")
    parser.add_argument("--dtype-out", choices=["f32", "f16", "bf16"], default="bf16",
                        help="Tensor dtype stored in bundle (pt supports bf16).")
    
    args = parser.parse_args()
    
    model, tokenizer = load_hyenadna(args.checkpoint)
    print(model)
    
    in_fasta_paths = args.input_fasta_list.read_text().splitlines()
    output_dir = args.output_dir / "hyenadna"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for in_fasta in in_fasta_paths:
            run_logitlens(model=model,
                          tokenizer=tokenizer,
                          in_fasta=in_fasta,
                          dtype_out=args.dtype_out,
                          output_dir=output_dir)
