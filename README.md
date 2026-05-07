# BGCs-Bench

[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](https://github.com/username/repo/blob/main/LICENSE)

BGCs-Bench is a unified benchmark focused on biosynthetic gene clusters (BGCs) for assessing long-sequence modeling and covers three complementary downstream tasks: biosynthetic class prediction, taxonomic classification, and CDS annotation.

Benchmarked models are below:
- HyenaDNA
- Evo
- Evo 2 (7B and 40B)
- NTv3

## Environment
All inference and analysis workflows (e.g., linear probing and logit lens) were executed using Apptainer.
```
# Evo 2
apptainer build containers/evo2_env.sif containers/evo2_env.def

# Other models and analyses
apptainer build containers/hf_env.sif containers/hf_env.def
```