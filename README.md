# Repeated mechanical challenge in a vertex model of clustered adhesion-deficient cells

Small simulation study built on the public code of

> N. Briñas-Pascual, V. Manso, P. Guerrero, "Spatial clustering of adhesion-deficient cells controls epithelial
> rigidity transitions", bioRxiv (2026), doi:10.64898/2026.06.08.730803
> code: https://github.com/vmansov/clustering_vmodel (itself based on https://github.com/berberto/vertex_model, GPLv3)

The `vertex_lite/` package and the example notebooks are the original authors' work. My additions are in `scripts/`,
`tests/`, `figures/` and the two `.md` files named below. The licence is GPLv3 (copied from the upstream repository).
The paper repository itself has no licence file; I have not asked the authors about reuse.

## Question
The paper's discussion asks whether the disorder left after a mutant cluster is cleared makes the tissue more
susceptible to later perturbations. I tested one version of this: introduce a second small clone into a tissue that
has already cleared a first clone.

## What I did
1. **Environment fix.** `np.cross` on 2D vectors fails on recent NumPy; replaced by `cross2d` in `vertex_lite/mesh.py`
   (used in `initialisation.py` too). `tests/regression_cross2d.py` checks that results are bitwise identical to the
   original on NumPy 2.0.2 and run on a newer NumPy.
2. **Reproduction (Stage 1).** D = 0.25, f_mut = 0.05 and 0.25, random vs cluster, 10 realisations.
3. **Second challenge (Stage 2).** 20 realisations, design fixed beforehand in `PREREGISTRATION_stage2.md`
   (with one dated amendment).

## Results
Stage 1 (T1_eps = 0.04, the code default):

| | t50 (steps) | t5 (steps) |
|---|---|---|
| random, f=0.05 | 60 | 98 |
| cluster, f=0.05 | 224 ± 8 | 550 ± 63 |
| random, f=0.25 | 80 | 182 ± 9 |
| cluster, f=0.25 | 480 ± 15 | 1456 ± 44 |

The cluster/random ratio of t50 is 3.7 (f=0.05) and 6.0 (f=0.25), so clustered mutants are cleared more slowly,
as in the paper. My clearance times look faster than the paper's figures (compared by eye, not extracted numerically).

Stage 2, second clone of 20 cells (mean ± SEM):

| condition | t50 (steps) | R_glob |
|---|---|---|
| a) primed tissue, clone at first clone's site | 385 ± 28 | 133 ± 12 |
| b) primed tissue, clone far away | 172 ± 3 | 59 ± 6 |
| c) naive tissue | 211 ± 6 | 84 ± 4 |
| d) fresh tissue, D = 0.70 | 178 ± 8 | 35 ± 6 |

Condition (a) differs from (c) and (d) on both primary outcomes after Holm correction; (b) behaves like (d). With a
randomly chosen first-clone location (10 runs) the pattern is the same. So the slowdown is tied to the site of the
earlier clone, not to the tissue-wide shape index. Figures: `figures/`.

## Caveats
- Table 1 of the paper lists T1 threshold 0.01 and an area rule A_crit = 0.01 A0. The code has no area rule: cells are
  removed through the T1 threshold (default 0.04). With 0.01 most clustered mutants are not cleared (`results` of
  Stage 1 variant, `figures/stage1_*_eps0.01.png`). I used 0.04 and do not know which values produced the paper's figures.
- Condition (d) is not an exact match: fresh tissue cannot reach the primed mean shape index for D <= 0.7
  (3.830 vs 3.842), and its topological disorder is much higher (0.69 vs 0.38).
- Primed tissues have about 300 cells and a smaller box; control tissues have 400 cells.
- The mechanism is not tested (residual local structure vs the shape of the second clone are both possible).
- "Local" observables depend on how the local cell set is defined; I report both definitions found in the paper/code.
- One realisation never cleared its first clone and is excluded from (a) and (b), as pre-specified.

## Running
```
pip install -r requirements-study.txt   # Python 3.10-3.12
python tests/regression_cross2d.py . /tmp/out.npz   # or: bash scripts/run_stage2.sh for the full Stage 2 run
cd scripts
python stage1_reproduction.py --eps 0.04 0.01 --nsim 10 --out ../results/stage1
python stage1_analysis.py ../results/stage1
python stage2_calibrate.py
python stage2_core.py --sims 0 20 --dstar 0.7 --out ../results/stage2
python stage2_analysis.py ../results/stage2
```
One full simulation takes roughly 30-40 s on one core; seeds are the realisation indices.

## Tools
I used an AI assistant (Claude) for part of the coding and for checking my reading of the original code. The design,
interpretation and everything stated above are mine to defend. <!-- edit this line to match what is true for you -->
