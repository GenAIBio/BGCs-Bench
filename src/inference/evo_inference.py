import argparse
import os
import torch
from pathlib import Path
from typing import Dict, Tuple
from Bio import SeqIO
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM


def _make_dirs_for_embeddings(output_dir: Path,
                              mean_pooling: bool,
                              last_token: bool,
                              raw_embeddings: bool,
                              reverse_complement: bool):
    suffix = ""
    if reverse_complement:
        suffix += "_rc"
    for i in range(32):
        if mean_pooling:
            (output_dir / f"mean_pooling{suffix}/blocks{i}").mkdir(parents=True, exist_ok=True)
        if last_token:
            (output_dir / f"last_token{suffix}/blocks{i}").mkdir(parents=True, exist_ok=True)
        if raw_embeddings:
            (output_dir / f"raw_embeddings{suffix}/blocks{i}").mkdir(parents=True, exist_ok=True)


def _make_device_map_fot_evo_131k(n_gpus: int) -> Dict[str, torch.device]:
    assert n_gpus >= 1
    devices = [f"cuda:{i}" for i in range(n_gpus)]
    n_blocks = 32
    blocks_per_gpu = (n_blocks + n_gpus - 1) // n_gpus
    device_map = {
        "": devices[0],
        "backbone.embedding_layer": devices[0],
        "backbone.norm": devices[0],
        # "backbone.unembed" is unnecessary due to tied weights of "backbone.embedding_layer"
        **{f"backbone.blocks.{i}": devices[i // blocks_per_gpu] for i in range(n_blocks)}
    }
    return device_map


def load_evo(checkpoint: str, offline: bool = False) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    assert torch.cuda.is_available()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    if "evo-1-131k" in checkpoint:
        device_map = _make_device_map_fot_evo_131k(torch.cuda.device_count())
    else:
        device_map = "auto"
    
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, revision="1.1_fix")
    config.use_cache = False  # disable cache for memory efficiency during inference
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=config, trust_remote_code=True, revision="1.1_fix", device_map=device_map
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision="1.1_fix")
    return model, tokenizer


def run_embed(model: AutoModelForCausalLM,
              tokenizer: AutoTokenizer,
              in_fasta: str,
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
    
    # Prepare to retrieve intermediate embeddings
    embeddings = {}
    handles = []
    def hook_fn(layer_name):
        def hook(_, __, output):
            if isinstance(output, tuple):
                output = output[0]
            embeddings[layer_name] = output.detach().to(device="cpu", dtype=out_dtype)
        return hook
    for i in range(32):
        handles.append(model.backbone.blocks[i].register_forward_hook(hook_fn(f"backbone.blocks.{i}")))
    
    if not reverse_complement:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.seq)
            input_ids = tokenizer(seq, return_tensors="pt").input_ids.to("cuda:0" if torch.cuda.is_available() else "cpu")
            output = model(input_ids)
            for i in range(32):
                if f"backbone.blocks.{i}" not in embeddings:
                    continue
                t = embeddings[f"backbone.blocks.{i}"].squeeze(0)  # [L,H]
                if raw_embeddings:
                    torch.save(t, output_dir / f"raw_embeddings/blocks{i}/{header}.pt")
                # save mean-pooled representation and last token representation
                # use contiguous().clone() to prevent pickling the parent tensor’s storage.
                if mean_pooling:
                    mean_rep = t.mean(dim=0).contiguous().clone()  # [H]
                    torch.save(mean_rep, output_dir / f"mean_pooling/blocks{i}/{header}.pt")
                if last_token:
                    last_rep = t[-1].contiguous().clone()  # [H]
                    torch.save(last_rep, output_dir / f"last_token/blocks{i}/{header}.pt")
    
    else:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.reverse_complement().seq)
            input_ids = tokenizer(seq, return_tensors="pt").input_ids.to("cuda:0" if torch.cuda.is_available() else "cpu")
            output = model(input_ids)
            for i in range(32):
                if f"backbone.blocks.{i}" not in embeddings:
                    continue
                t = embeddings[f"backbone.blocks.{i}"].squeeze(0)  # [L,H]
                if raw_embeddings:
                    torch.save(t, output_dir / f"raw_embeddings_rc/blocks{i}/{header}.pt")
                # save mean-pooled representation and last token representation
                # use contiguous().clone() to prevent pickling the parent tensor’s storage.
                if mean_pooling:
                    mean_rep = t.mean(dim=0).contiguous().clone()  # [H]
                    torch.save(mean_rep, output_dir / f"mean_pooling_rc/blocks{i}/{header}.pt")
                if last_token:
                    last_rep = t[-1].contiguous().clone()  # [H]
                    torch.save(last_rep, output_dir / f"last_token_rc/blocks{i}/{header}.pt")
    
    # Remove hooks
    for handle in handles:
        handle.remove()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evo inference")
    parser.add_argument("--input-fasta-list", type=Path, required=True,
                        help="Input FASTA file list (one path per line).")
    parser.add_argument("--output-dir", default=Path("embedding"), type=Path,
                        help="Output directory to save embeddings.")
    parser.add_argument("--checkpoint", default="togethercomputer/evo-1-131k-base",
                        help="Checkpoint name (togethercomputer/evo-1-8k-base, togethercomputer/evo-1-131k-base, ...)")
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
    
    model, tokenizer = load_evo(args.checkpoint)
    print(model)
    
    in_fasta_paths = args.input_fasta_list.read_text().splitlines()
    output_dir = args.output_dir / "evo"
    _make_dirs_for_embeddings(output_dir,
                              args.mean_pooling,
                              args.last_token,
                              args.raw_embeddings,
                              args.reverse_complement)
    
    with torch.no_grad():
        for in_fasta in in_fasta_paths:
            run_embed(model=model,
                      tokenizer=tokenizer,
                      in_fasta=in_fasta,
                      dtype_out=args.dtype_out,
                      output_dir=output_dir,
                      mean_pooling=args.mean_pooling,
                      last_token=args.last_token,
                      raw_embeddings=args.raw_embeddings,
                      reverse_complement=args.reverse_complement)
