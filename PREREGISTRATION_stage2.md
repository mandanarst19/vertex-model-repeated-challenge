# Stage 2 pre-registration (written and committed BEFORE any Stage 2 simulation was run)

**Question.** After a clustered mutant clone (D=0.25, f=0.25, seed cell 64) has been fully cleared, the tissue keeps
<p>_global above p* (Stage 1: 3.842 vs 3.766 initially). Does such a primed tissue respond differently to a second,
small clone than (c) a naive tissue and (d) a naive tissue whose <p>_global is matched to the primed one?

**Fixed settings.** eps_T1 = 0.04 (code default; differs from Table 1 of the paper, see README), D0 = 0.25,
first clone f1 = 0.25, second clone n2 = 20 cells (f2 = 0.05), phase-1 length 15000 steps (= paper's evaluation time),
phase-2 length 6000 steps, frames every 20 steps. Realisations: sim_id 0..19 (20).

**Conditions (second clone, always a contiguous cluster of 20 cells):**
- (a) PRIMED-SAME: primed tissue, seed = alive cell nearest to the collapse site of clone 1
  (circular mean of centroids of clone-1's initial neighbour ring).
- (b) PRIMED-FAR: same primed tissue, seed = alive cell with maximal minimum-image distance from that site.
- (c) NAIVE: same-sim_id relaxed tissue (D0) that never had mutants, seed cell 64.
- (d) MATCHED: fresh tissue with D* chosen so that its relaxed <p>_global equals the Stage-1 primed value 3.842
  (calibration on separate sim_ids 1000-1002, linear interpolation over D in {0.25,0.35,0.45,0.55,0.65,0.70}), seed cell 64.
  <Dhex>_global of (d) vs primed (0.377) is reported as the mismatch of the other variable.
Limitation fixed in advance: primed tissues have ~300 alive cells, (c)/(d) have 400 (cell number/box size is not controlled).

**Exclusion rule.** A realisation is excluded from (a),(b) if any phase-1 mutant is still alive at the end of phase 1.
Number excluded is reported.

**Regions / sets.** Disk: WT cells whose centroid (minimum-image) lies within r = sqrt(n2*A0/pi) + sqrt(4*A0/pi)
= 2.52 + 1.13 = 3.65 (units sqrt(A0)) of the second clone's initial centroid; set tracked by cell id. Also reported
(paper definitions): initial neighbour ring (tracked), and the neighbour set frozen at clearance.

**Metrics for the second clone** (computed per realisation):
t50, t5 of clone-2 (frame resolution 20 steps); R_glob = sum_t [<p>_global(t) - <p>_global(0)] * 20;
R_disk (same with disk <p>); T_above = steps with <p>_global > 3.81; peak <p>_global; peak <p>_disk;
final <Dhex> (global, disk, ring, frozen set).
**Primary outcomes: t50 and R_glob.** Everything else is secondary/descriptive.

**Statistics.** Two-sided Mann-Whitney U; effect = difference of means with 95% bootstrap CI (10000 resamples) and
Cliff's delta. Primary contrasts: (a)-(c), (a)-(d) on both primary outcomes = 4 tests, Holm-adjusted. (a)-(b) is secondary.

**Interpretation rules.** "Primed tissue responds differently" only if (a) differs from BOTH (c) and (d) on a primary
outcome after Holm adjustment AND the CI of the (a)-(d) difference excludes 0. If (a) ~ (d): the response is explained
by the <p>_global level, not by history. Null result is reported as such.

**Robustness (after the main analysis).** Repeat (a)-(c) with the first clone seeded at a random cell (as in the paper text),
10 realisations, to check dependence on the single location 64.

---
## Amendment 1 (committed after calibration, before any Stage 2 challenge simulation)
Calibration (3 fresh relaxed tissues per D, eps=0.04) gave <p>_global = 3.767 (D=0.25), 3.807 (0.35), 3.812 (0.45),
3.818 (0.55), 3.824 (0.65), 3.830 (0.70): saturating, and the pre-registered target 3.842 is NOT reachable for any D <= 0.7
(D* = null). The original text had no rule for this case. Decision, taken without having seen any second-clone result:
condition (d) uses D = 0.70 (upper end of the paper's range = closest reachable match). The residual mismatch
(<p>_global 3.830 vs 3.842) and the mismatch of <Dhex>_global (fresh D=0.70: ~0.75 vs primed ~0.38) are reported with the results.
So (d) is "maximally disordered fresh tissue", not an exact match; conclusions about (a) vs (d) must say so.
