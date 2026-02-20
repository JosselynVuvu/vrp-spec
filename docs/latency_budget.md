5.Y Latency Budget and SLO Definition
We define the replanning service-level objective (SLO) as p95 end-to-end replanning latency ≤ 800 ms on local CPU hardware. A replanning call consists of (i) cost construction, (ii) OR-Tools solve under a time cap, and (iii) minimal decoding/glue overhead. 
Formally,
T_replan^p95=T_costbuild^p95+T_solver^p95+T_decode^p95≤800" ms".

We benchmark the cost construction stage independently to quantify fixed overhead. Over repeated runs, cost construction (event generation, rain overlay, CO₂ proxy computation, and integer cost tensor assembly) achieves p50 = 0.078 ms and p95 = 0.156 ms. This overhead is negligible relative to the 800 ms SLO, leaving approximately 800-0.156=799.844ms of p95 budget for the OR-Tools solve and decoding overhead.

Component	What it includes	p50 (ms)	p95 (ms)	Notes
Cost-build	events + rain overlay + CO2 proxy + int costs	0.078	0.156	in-memory (no disk I/O)
OR-Tools solve	routing optimization under time cap	TBD	TBD	caps: 200/500/800ms
Decode + glue	decode solution + overhead	TBD	TBD	keep minimal
Total replan	sum of above	TBD	TBD	target p95 ≤ 800ms

