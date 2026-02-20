Z Scenario Parameters and Justification
Rain and disruption effects are modeled using multiplicative scenario parameters to enable controlled sensitivity analysis under limited calibration data. The selected values are not treated as city-specific physical constants; instead, they represent light/moderate/heavy impact regimes that produce measurable differences in routing outcomes while preserving feasibility and interpretability.
Rain impact on travel time. During rain-active bins, travel time is inflated by a factor 1+ρ_TT, where

These levels correspond to approximately +5% (light), +10% (moderate), and +20% (heavy) degradation. As a sanity check, a 10-minute leg becomes 10.5, 11.0, or 12.0 minutes under the three severity levels, respectively.

Rain impact on emissions. Emissions are already affected by rain indirectly because rain inflates travel time, which reduces speed and changes the speed-dependent emissions proxy e(v). To avoid double counting, we apply a smaller additional multiplicative factor to emissions in rain bins:
ρ_CO2∈{0.02,0.05,0.10}.

This term represents residual effects not captured by speed alone (e.g., increased rolling resistance and stop-go behavior).
Blockage modeling and BIG_M. A road closure is represented as a blocked origin–destination arc (uⓜ→v)during a fixed blockage bin. Rather than modifying a full road network graph, feasibility is preserved by enforcing a large penalty directly on the planning cost:
J_cost_int[blockage_bin,u,v]=BIG_M_cost_int.

The constant BIG_M_cost_intis chosen to dominate any feasible arc cost in that bin (e.g., ≥10× the maximum typical arc cost), ensuring the solver avoids the blocked arc while keeping the model simple and reproducible.

Observability assumption. Rain is treated as exogenous and observable to the planner, so the planner’s predicted costs incorporate rain effects. The primary uncertainty studied is the time dependence of congestion and the impact of discrete disruption events on route quality and replanning latency.