# Pilot Run 001 — RFdiffusion on AWS g6.xlarge

## Run Parameters

| Parameter | Value |
|-----------|-------|
| Date | 2026-06-16 |
| Instance | g6.xlarge (NVIDIA L4, 24GB VRAM) |
| Region | us-east-1 |
| AMI | ami-012ba162b9cd2729c (Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.7, Ubuntu 22.04) |
| Pricing | On-demand $0.805/hr |
| Checkpoint | Base_ckpt.pt (462MB) |
| Target | 1UBQ (ubiquitin, chain A, 76 residues) |
| Contig | [100-150/A1-76] |
| Hotspots | None (Base model doesn't support PPI hotspots) |
| Diffusion steps (T) | 50 |
| Num designs | 50 |

## Timing

| Phase | Duration |
|-------|----------|
| Instance boot | ~1 min |
| SSH available | ~15 sec after boot |
| RFdiffusion clone (full) | ~2 min |
| Conda env setup (Python 3.9 + PyTorch 2.1 + DGL 1.1.3+cu121) | ~8 min |
| Model weights download (Base_ckpt.pt, 462MB) | ~2 min |
| SE3Transformer + RFdiffusion install | ~3 min |
| IGSO3 cache computation (first run) | ~17 sec |
| **Total setup** | **~16 min** |
| Per-design generation (T=50, ~170 residues total) | **51 sec avg** |
| **Total generation (50 designs)** | **~42 min** |
| **Total wall-clock** | **~58 min** |

## Cost

| Item | Cost |
|------|------|
| Instance (58 min × $0.805/hr) | $0.78 |
| EBS (100GB gp3, 1 hr) | ~$0.01 |
| **Total** | **~$0.79** |

## GPU Metrics

| Metric | Value |
|--------|-------|
| GPU utilization | 79–82% |
| VRAM used | 4.3–5.5 GB / 23 GB |
| VRAM headroom | ~18 GB free |

## Results

- **50/50 designs generated successfully**
- PDB files lost (burner account deactivated before scp)
- No filtering analysis completed

## Issues Encountered

1. **Shallow clone (`--depth 1`) missing `igso3.py`** — must use full `git clone`
2. **Complex_base_ckpt.pt 404** from files.ipd.uw.edu — hotspot-guided PPI design unavailable
3. **DGL version hell** — system PyTorch 2.12 incompatible with DGL; required conda env with PyTorch 2.1
4. **numpy 2.x breaks IGSO3 pickle cache** — must pin numpy <2
5. **SE3Transformer install order matters** — must install before `pip install -e RFdiffusion`
6. **Burner account deactivated mid-run** — lost all output data

## Environment Recipe (working)

```bash
# Conda env
conda create -n rfdiff python=3.9 pytorch=2.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
conda activate rfdiff
pip install dgl==1.1.3 -f https://data.dgl.ai/wheels/cu121/repo.html
pip install hydra-core omegaconf e3nn==0.5.1 biopython pyyaml 'numpy<2' scipy pyrsistent

# RFdiffusion (full clone, not --depth 1)
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion/env/SE3Transformer && pip install . && cd ../..
pip install -e .

# Weights
mkdir -p models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
```

## Extrapolations

| Scale | Time | Cost (on-demand) |
|-------|------|-------------------|
| 50 designs | 42 min | $0.56 |
| 500 designs | 7 hr | $5.60 |
| 5,000 designs | 71 hr | $57 |
| 50,000 designs | 29 days | $565 |

## Next Steps

1. Find Complex_base_ckpt.pt mirror (HuggingFace, alternative URL) for hotspot-guided design
2. Update deploy scripts with correct conda recipe
3. Re-run with persistent AWS account, scp results before termination
4. Run contact filter + pipeline stages 3–6 on retrieved PDBs
