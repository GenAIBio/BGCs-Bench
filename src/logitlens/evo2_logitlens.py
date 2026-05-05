import argparse
import os
import torch
from pathlib import Path
from typing import List
from Bio import SeqIO
from evo2 import Evo2  # official wrapper


def _looks_like_local_ckpt_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    if not (p / "config.json").exists():
        return False
    shards = list(p.glob("*.pt.part*")) + list(p.glob("*.pt"))
    return len(shards) > 0


def load_evo2(model_or_path: str, offline: bool = False) -> Evo2:
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    p = Path(model_or_path)
    if _looks_like_local_ckpt_dir(p):
        return Evo2(str(p))  # Vortex resolves local shards
    return Evo2(model_or_path)  # HF name (HF_HOME respected)


def compute_log_likelihood_per_base(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # Compute log probabilities
    logp_next = torch.log_softmax(logits, dim=-1)               # (L, V)
    # Shift so that tokens < n predict n
    shift_logp = logp_next[..., :-1, :].contiguous()            # (L-1, V)
    shift_labels = labels[..., 1:].unsqueeze(-1).contiguous()   # (L-1, 1)
    # Gather log-probability for the actual next token
    LL = torch.gather(shift_logp, -1, shift_labels).squeeze(-1) # (L-1)
    return LL


def run_logitlens(model: Evo2,
                  in_fasta: str,
                  layer_names: List[str],
                  dtype_out: str,
                  output_dir: Path,
                  reverse_complement: bool = False):
    records = list(SeqIO.parse(in_fasta, "fasta"))
    if len(records) == 0:
        raise RuntimeError("No sequences found in FASTA.")
    
    dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
    if dtype_out not in dtype_map:
        raise ValueError(f"Unsupported dtype_out: {dtype_out}")
    out_dtype = dtype_map[dtype_out]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if not reverse_complement:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.seq)
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq), dtype=torch.int
            ).unsqueeze(0).to(device)
            _, embeddings = model(input_ids, return_embeddings=True, layer_names=layer_names)
            ll_scores = {}
            for name in layer_names:
                if name not in embeddings:
                    continue
                t = embeddings[name].detach().to(device=device).squeeze(0)  # [L,H]
                t_norm = model.model.norm(t).detach()
                logits = model.model.unembed(t_norm).to(dtype=out_dtype)
                LL = compute_log_likelihood_per_base(logits, input_ids.to(int).squeeze(0))
                ll_scores[name] = LL
            torch.save(ll_scores, output_dir / f"{header}.log_likelihood.pt")
    
    else:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.reverse_complement().seq)
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq), dtype=torch.int
            ).unsqueeze(0).to(device)
            _, embeddings = model(input_ids, return_embeddings=True, layer_names=layer_names)
            ll_scores = {}
            for name in layer_names:
                if name not in embeddings:
                    continue
                t = embeddings[name].detach().to(device=device).squeeze(0)  # [L,H]
                t_norm = model.model.norm(t).detach()
                logits = model.model.unembed(t_norm).to(dtype=out_dtype)
                LL = compute_log_likelihood_per_base(logits, input_ids.to(int).squeeze(0))
                ll_scores[name] = LL
            torch.save(ll_scores, output_dir / f"{header}.log_likelihood.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evo2 inference")
    parser.add_argument("--input-fasta-list", type=Path, required=True,
                        help="Input FASTA file list (one path per line).")
    parser.add_argument("--output-dir", default=Path("logitlens"), type=Path,
                        help="Output directory to save unembed log-likelihood scores.")
    parser.add_argument("--model", default="evo2_7b",
                        help="Model name (evo2_7b/evo2_40b/...) or LOCAL DIR with config.json & shards")
    parser.add_argument("--layer-names", type=str, nargs="+", required=True,
                        help="Internal module name (repeatable). See `list-layers`.")
    parser.add_argument("--dtype-out", choices=["f32", "f16", "bf16"], default="bf16",
                        help="Tensor dtype stored in bundle (pt supports bf16).")
    
    args = parser.parse_args()
    
    model = load_evo2(args.model)
    
    in_fasta_paths = args.input_fasta_list.read_text().splitlines()
    output_dir = args.output_dir / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for in_fasta in in_fasta_paths:
        run_logitlens(model=model,
                      in_fasta=in_fasta,
                      layer_names=args.layer_names,
                      dtype_out=args.dtype_out,
                      output_dir=output_dir)
