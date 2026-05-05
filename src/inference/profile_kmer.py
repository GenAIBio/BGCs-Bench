import argparse
import functools
import numpy as np
from collections import Counter
from pathlib import Path
from typing import List, Dict
from Bio import SeqIO


# K-mer profiling
@functools.lru_cache()
def list_kmers(k: int) -> List[str]:
    bases = ["A", "C", "G", "T"]
    if k == 1:
        return bases
    else:
        smaller_kmers = list_kmers(k - 1)
        kmers = []
        for base in bases:
            for smaller_kmer in smaller_kmers:
                kmers.append(base + smaller_kmer)
        return kmers

def profile_kmer(sequence: str, k: int) -> Dict[str, float]:
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(dict.fromkeys(list_kmers(k), 0))
    kmer_counts.update(kmers)
    total_kmers = sum(kmer_counts.values())
    kmer_freqs = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
    return kmer_freqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="K-mer profiling")
    parser.add_argument("--input-fasta-list", type=Path, required=True,
                        help="Input FASTA file list (one path per line).")
    parser.add_argument("--output-dir", default=Path("profiles"), type=Path,
                        help="Output directory to save profiles.")
    args = parser.parse_args()
    
    in_fasta_paths = args.input_fasta_list.read_text().splitlines()
    
    for k in range(3, 10):
        kmer_profiles = []
        for in_fasta in in_fasta_paths:
            records = list(SeqIO.parse(in_fasta, "fasta"))
            for record in records:
                kmer_profile = profile_kmer(str(record.seq), k)
                kmer_profiles.append([kmer_profile[kmer] for kmer in list_kmers(k)])
        
        kmer_profiles = np.array(kmer_profiles)
        print(kmer_profiles.shape)
        np.save(f"{args.output_dir}/kmer_profile_k{k}.npy", kmer_profiles)
