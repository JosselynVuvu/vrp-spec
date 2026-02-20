5.X Reproducibility Contract

To ensure reproducibility and prevent data leakage, we enforce the following protocol throughout all experiments:

1. Dataset and base instance lock. All experiments use the VRPTDT benchmark dataset and a single fixed base instance, berlin_500.json. No alternative cities or base instances are used in reported results.

2. Canonical episode definition. Each episode contains exactly 20 customers plus one depot (N=21). Customer nodes are sampled deterministically from the base instance using a fixed RNG seed. The sampled node IDs are stored with the episode artifact to guarantee identical reconstruction across runs.

3. Disjoint seed-set splits (no leakage). TRAIN/VAL/TEST are defined by disjoint seed-sets. No seed appears in more than one split. TRAIN is used for parameter fitting (e.g., λ), VAL is used only for model selection when applicable, and TEST is used exclusively for final reporting.

4. Deterministic event generation. For a fixed episode seed, the dynamic events are deterministic. Rain bin selection, rain intensity parameters 
(𝜌𝑇𝑇,𝜌𝐶𝑂2)(ρTT​,ρCO2	​), and the blocked arc (𝑢→𝑣)
(u→v) are generated using a seed-driven procedure. This ensures that repeated runs produce identical disruption realizations.

5. Frozen time discretization. Time dependence is represented using a fixed number of time bins B as provided by the VRPTDT instance configuration. All travel-time and emissions computations are performed within this frozen discretization and are not re-binned in final results.
6. Frozen emissions proxy. Carbon emissions are computed via a speed-dependent MEET/Jabali-style proxy:

CO2_ij (b)=dist_ij⋅e(v_ij (b)),e(v)=αv^2+βv+γ+δ/v,

where speed 𝑣𝑖𝑗(𝑏) is derived from distance and bin-dependent travel time. Parameters and any speed clipping are fixed in code/config and remain unchanged across runs.

7. TRAIN-only objective normalization. The trade-off weight 𝜆
λ used in the composite objective is computed once using TRAIN seeds only:

𝜆 =median(𝐶𝑂2𝑙𝑒𝑔)median(𝑇𝑇𝑙𝑒𝑔). λ =median(TTleg​)/median(CO2leg).

The resulting value is stored in configs/lambda.json and is kept fixed for all VAL/TEST evaluations.


8. Blockage enforcement on planning cost. During the blockage bin, the blocked arc (u→v)is penalized by setting the planning cost to a large constant:

J_cost_int[blockage_bin,u,v]=BIG_M_cost_int,

ensuring the planner avoids the arc during the disruption window. This invariant is verified by automated checks.

9. Hardware-scoped runtime reporting. All latency results (p50/p95) are measured on the same local CPU hardware. GPU/Colab resources are not used for any reported latency metrics.
	
10. Versioning and command reproducibility. Code and configuration files are tracked in git, while raw and processed datasets are excluded via .gitignore. Each figure/table can be reproduced from a documented command sequence and pinned package versions.


