import argparse
import os
import torch
from pathlib import Path
from typing import List
from Bio import SeqIO
from evo2 import Evo2  # official wrapper


def _make_dirs_for_embeddings(output_dir: Path,
                              layer_names: List[str],
                              mean_pooling: bool,
                              last_token: bool,
                              raw_embeddings: bool,
                              reverse_complement: bool):
    suffix = ""
    if reverse_complement:
        suffix += "_rc"
    for name in layer_names:
        if mean_pooling:
            (output_dir / f"mean_pooling{suffix}/{name.replace('.', '')}").mkdir(parents=True, exist_ok=True)
        if last_token:
            (output_dir / f"last_token{suffix}/{name.replace('.', '')}").mkdir(parents=True, exist_ok=True)
        if raw_embeddings:
            (output_dir / f"raw_embeddings{suffix}/{name.replace('.', '')}").mkdir(parents=True, exist_ok=True)


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


def run_embed(model: Evo2,
              in_fasta: str,
              layer_names: List[str],
              dtype_out: str,
              output_dir: Path,
              mean_pooling: bool = True,
              last_token: bool = True,
              raw_embeddings: bool = False,
              reverse_complement: bool = False):
    records = list(SeqIO.parse(in_fasta, "fasta"))
    if len(records) == 0:
        raise RuntimeError("No sequences found in FASTA.")
    
    dtype_map = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}
    if dtype_out not in dtype_map:
        raise ValueError(f"Unsupported dtype_out: {dtype_out}")
    out_dtype = dtype_map[dtype_out]
    
    if not reverse_complement:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.seq)
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq), dtype=torch.int
            ).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else "cpu")
            _, embeddings = model(input_ids, return_embeddings=True, layer_names=layer_names)
            for name in layer_names:
                if name not in embeddings:
                    continue
                t = embeddings[name].detach().to(device="cpu", dtype=out_dtype).squeeze(0)  # [L,H]
                if raw_embeddings:
                    torch.save(t, output_dir / f"raw_embeddings/{name.replace('.', '')}/{header}.pt")
                # save mean-pooled representation and last token representation
                # use contiguous().clone() to prevent pickling the parent tensor’s storage.
                if mean_pooling:
                    mean_rep = t.mean(dim=0).contiguous().clone()  # [H]
                    torch.save(mean_rep, output_dir / f"mean_pooling/{name.replace('.', '')}/{header}.pt")
                if last_token:
                    last_rep = t[-1].contiguous().clone()         # [H]
                    torch.save(last_rep, output_dir / f"last_token/{name.replace('.', '')}/{header}.pt")
    
    else:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.reverse_complement().seq)
            input_ids = torch.tensor(
                model.tokenizer.tokenize(seq), dtype=torch.int
            ).unsqueeze(0).to("cuda:0" if torch.cuda.is_available() else "cpu")
            _, embeddings = model(input_ids, return_embeddings=True, layer_names=layer_names)
            for name in layer_names:
                if name not in embeddings:
                    continue
                t = embeddings[name].detach().to(device="cpu", dtype=out_dtype).squeeze(0)  # [L,H]
                if raw_embeddings:
                    torch.save(t, output_dir / f"raw_embeddings_rc/{name.replace('.', '')}/{header}.pt")
                # save mean-pooled representation and last token representation
                # use contiguous().clone() to prevent pickling the parent tensor’s storage.
                if mean_pooling:
                    mean_rep = t.mean(dim=0).contiguous().clone()  # [H]
                    torch.save(mean_rep, output_dir / f"mean_pooling_rc/{name.replace('.', '')}/{header}.pt")
                if last_token:
                    last_rep = t[-1].contiguous().clone()         # [H]
                    torch.save(last_rep, output_dir / f"last_token_rc/{name.replace('.', '')}/{header}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evo2 inference")
    parser.add_argument("--input-fasta-list", type=Path, required=True,
                        help="Input FASTA file list (one path per line).")
    parser.add_argument("--output-dir", default=Path("embedding"), type=Path,
                        help="Output directory to save embeddings.")
    parser.add_argument("--model", default="evo2_7b",
                        help="Model name (evo2_7b/evo2_40b/...) or LOCAL DIR with config.json & shards")
    parser.add_argument("--layer-names", type=str, nargs="+", required=True,
                        help="Internal module name (repeatable). See `list-layers`.")
    parser.add_argument("--dtype-out", choices=["f32", "f16", "bf16"], default="bf16",
                        help="Tensor dtype stored in bundle (pt supports bf16).")
    parser.add_argument("--mean-pooling", action="store_true",
                        help="If set, compute mean-pooled embeddings for each input sequence.")
    parser.add_argument("--last-token", action="store_true",
                        help="If set, compute last-token embeddings for each input sequence.")
    parser.add_argument("--raw-embeddings", action="store_true",
                        help="If set, save raw token-level embeddings for each input sequence.")
    parser.add_argument("--reverse-complement", action="store_true",
                        help="If set, compute embeddings for the reverse complement of each input sequence.")
    
    args = parser.parse_args()
    
    model = load_evo2(args.model)
    print(model)
    
    in_fasta_paths = args.input_fasta_list.read_text().splitlines()
    output_dir = args.output_dir / args.model
    _make_dirs_for_embeddings(output_dir,
                              args.layer_names,
                              args.mean_pooling,
                              args.last_token,
                              args.raw_embeddings,
                              args.reverse_complement)
    
    for in_fasta in in_fasta_paths:
        run_embed(model=model,
                  in_fasta=in_fasta,
                  layer_names=args.layer_names,
                  dtype_out=args.dtype_out,
                  output_dir=output_dir,
                  mean_pooling=args.mean_pooling,
                  last_token=args.last_token,
                  raw_embeddings=args.raw_embeddings,
                  reverse_complement=args.reverse_complement)
