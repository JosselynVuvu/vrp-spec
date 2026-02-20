# Milestone 4 — Step 1: EWMA Digital Twin (no gated policy)

## Goal
Implement a Milestone 4 **digital twin multiplier** \(\hat m(b)\) (scalar per time bin \(b\)) with an **online EWMA update**, and use it inside the receding-horizon OR-Tools replanning loop.

This Milestone 4 submission compares the baseline replanning policies:
- **B0_PlanOnce**
- **B2_BlockageReplan**
- **B1_AlwaysReplan**

> We do **NOT** include gated replanning / Option C in this writeup.

---

## Digital Twin (planner prediction)
Planner predicted travel time in bin \(b\):

\[
\widehat{TT}_{ij}(b)=\hat m(b)\cdot TT^{data}_{ij}(b)\cdot \bigl(1+\rho_{TT}(b)\bigr).
\]

- Rain is **observable** to the planner: when rain is on in bin \(b\), apply \(\bigl(1+\rho_{TT}(b)\bigr)\) to TT and \(\bigl(1+\rho_{CO2}(b)\bigr)\) to CO₂.
- Blockage is **not** baked into \(\widehat{TT}\). Blocked arcs are penalized only in the **planning cost** via a BIG\_M patch at `blockage_bin`.

CO₂ prediction is derived from \(\widehat{TT}\) via the emissions proxy:
\[
\widehat{CO2}(b)=\text{MEET}\!\left(dist,\widehat{TT}(b)\right)\cdot \bigl(1+\rho_{CO2}(b)\bigr).
\]

Planning objective:
\[
\widehat{J}(b)=\widehat{CO2}(b)+\lambda\cdot \widehat{TT}(b).
\]
Then \(\widehat{J}\) is scaled into integer arc costs for OR-Tools.

---

## EWMA update rule (frozen \(\alpha=0.2\))
We maintain \(\hat m(b)\) per bin \(b\). Initialize with a TRAIN-only bin-mean prior \(m_{\text{init}}(b)\).
At the start of each episode, set \(\hat m(b)\leftarrow m_{\text{init}}(b)\) for all bins.

When a leg \((i\rightarrow j)\) executes in bin \(b\), compute an observed multiplier estimate:

\[
m_{\text{obs}}=
\frac{TT_{\text{obs}}}
{\max\!\left(TT^{data}_{ij}(b)\cdot (1+\rho_{TT}(b)),\,10^{-9}\right)}.
\]

EWMA update (only for the executed bin \(b\)):
\[
\hat m(b)\leftarrow (1-\alpha)\hat m(b)+\alpha\cdot m_{\text{obs}}.
\]

The updated \(\hat m(b)\) is used for future replans that occur while the vehicle is in bin \(b\).

---

## Accident / blockage execution model (carryover)
If a blocked arc is attempted during `blockage_bin`:
- Vehicle **waits** until the bin ends (engine off)
- Then traverses the arc in the next bin using that bin’s TT/CO₂ truth.

This waiting contributes to `traffic_wait_min` and increases `wall_time_min` unless replanning avoids the blocked arc.

---

## Logged Milestone 4 prediction error metrics
For each executed leg (using the execution bin \(b\)):
- Predicted TT (twin): \(TT_{\text{hat}}=\widehat{TT}_{ij}(b)\)
- Executed truth TT: \(TT_{\text{obs}}\)

Relative error:
\[
e=\frac{|TT_{\text{obs}}-TT_{\text{hat}}|}{\max(TT_{\text{hat}},10^{-9})}.
\]

Aggregates written to the Milestone 4 CSVs:
- `rel_pred_err_mean`
- `rel_pred_err_p95`
- `rel_pred_err_max`
