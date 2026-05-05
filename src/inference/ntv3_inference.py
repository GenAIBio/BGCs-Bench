import argparse
import os
import torch
from pathlib import Path
from typing import Tuple
from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM


def _make_dirs_for_embeddings(output_dir: Path, reverse_complement: bool, n_blocks: int):
    suffix = ""
    if reverse_complement: suffix += "_rc"
    for i in range(n_blocks):
        (output_dir / f"mean_pooling{suffix}/blocks{i}").mkdir(parents=True, exist_ok=True)


def load_ntv3(checkpoint: str, offline: bool = False) -> Tuple[AutoModelForMaskedLM, AutoTokenizer]:
    assert torch.cuda.is_available()
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    model = AutoModelForMaskedLM.from_pretrained(checkpoint, trust_remote_code=True, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    return model, tokenizer


def run_embed(model: AutoModelForMaskedLM,
              tokenizer: AutoTokenizer,
              in_fasta: str,
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
    
    if not reverse_complement:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.seq)
            encoding = tokenizer(seq, add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")
            input_ids = encoding.input_ids.to("cuda:0" if torch.cuda.is_available() else "cpu")
            output = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states  # List[torch.Tensor], each is [B,L,H] (B=1 here)
            for i, hidden_state in enumerate(hidden_states):
                h = hidden_state.detach().to(device="cpu", dtype=out_dtype).squeeze(0)  # [L,H]
                # use contiguous().clone() to prevent pickling the parent tensor’s storage.
                mean_rep = h.mean(dim=0).contiguous().clone()
                torch.save(mean_rep, output_dir / f"mean_pooling/blocks{i}/{header}.pt")
    
    elif reverse_complement:
        for idx, record in enumerate(records, 1):
            header = record.id
            seq = str(record.reverse_complement().seq)
            encoding = tokenizer(seq, add_special_tokens=False, padding=True, pad_to_multiple_of=128, return_tensors="pt")
            input_ids = encoding.input_ids.to("cuda:0" if torch.cuda.is_available() else "cpu")
            output = model(input_ids=input_ids, output_hidden_states=True)
            hidden_states = output.hidden_states  # List[torch.Tensor], each is [B,L,H] (B=1 here)
            for i, hidden_state in enumerate(hidden_states):
                h = hidden_state.detach().to(device="cpu", dtype=out_dtype).squeeze(0)  # [L,H]
                # use contiguous().clone() to prevent pickling the parent tensor’s storage.
                mean_rep = h.mean(dim=0).contiguous().clone()
                torch.save(mean_rep, output_dir / f"mean_pooling_rc/blocks{i}/{header}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTv3 inference")
    parser.add_argument("--input-fasta-list", type=Path, required=True,
                        help="Input FASTA file list (one path per line).")
    parser.add_argument("--output-dir", default=Path("embedding"), type=Path,
                        help="Output directory to save embeddings.")
    parser.add_argument("--checkpoint", default="InstaDeepAI/NTv3_650M_pre",
                        help="Checkpoint name (InstaDeepAI/NTv3_100M_pre, InstaDeepAI/NTv3_650M_pre, ...)")
    parser.add_argument("--dtype-out", choices=["f32", "f16", "bf16"], default="bf16",
                        help="Tensor dtype stored in bundle (pt supports bf16).")
    parser.add_argument("--reverse-complement", action="store_true",
                        help="If set, compute embeddings for the reverse complement of each input sequence.")
    
    args = parser.parse_args()
    
    model, tokenizer = load_ntv3(args.checkpoint)
    print(model)
    
    in_fasta_paths = args.input_fasta_list.read_text().splitlines()
    output_dir = args.output_dir / args.checkpoint.split("/")[-1].lower()
    n_blocks = model.config.num_downsamples * 2 + model.config.num_layers
    _make_dirs_for_embeddings(output_dir, args.reverse_complement, n_blocks)
    
    with torch.no_grad():
        for in_fasta in in_fasta_paths:
            run_embed(model=model,
                      tokenizer=tokenizer,
                      in_fasta=in_fasta,
                      dtype_out=args.dtype_out,
                      output_dir=output_dir,
                      reverse_complement=args.reverse_complement)
