## context 

- What role exactly does the Deeptica ML model play in this loop? Are you using it to learn collective variables (CVs) _before_ MSM construction, or also using its learned space to drive new simulations?
the deeptica ml model is for training the CV's for the next simulations so we get more data about the FES more quickly. its like the normal CV(RMSD) but found with eigenvalues which helps to make it faster, and i want to train in on the shards like this in the dataset t_1 ) 1 shard t_2) 2 shard(the one that was run by the first model with one shard) t_n) n shards(that which were model was trained on previous one)

- Do you want the experiment to test the full loop where Deeptica-learned CVs are fed back into the simulation to bias new shards (i.e. a closed learning loop)? Or is Deeptica only evaluated post hoc for CV quality?
i want deeptica to be fed back into simulation to bias new shards

- Do you want to test the integration of Deeptica under both E0 and E1/E2 settings, or only after reweighting is successful (E1)?
i want the test to see if the everything works with the worfklow under those different scenarios from different shards. i want deeptica everythwere. i want it to be like that "-Eoue pipeline" which if anything fails we see what's wrong.


- Should the evaluation include whether Deeptica’s CVs improve MSM quality or FES smoothness compared to handpicked features?
so that would be a case later, now we want to see if even it works.

## Goal/Objective
Implement a **repeatable, deterministic** test battery that validates the full PMARLO workflow under three data regimes:

1. Verify end-to-end operation: simulate, ingest, transform/discretize, apply optional reweighting (TRAM/MBAR), build MSM/FES, train **DeepTICA** ML CV model, and feed it back into the simulation loop. This closes the loop (simulation → analysis → learned CV → new simulation) and ensures each stage works in sequence. After each full cycle, produce debug artifacts for analysis.
2. Produce deterministic artifacts and **objective acceptance checks** so regressions are obvious based on the unit tests in the `pmarlo` package. For each experiment, define clear pass/fail criteria (numerical thresholds) on key outputs.
3. Exercise guardrails (SCC coverage, transition counts, reweighter validation, DeepTICA data sufficiency) in both “should pass” and “should fail/flag” settings. The pipeline should **fail fast and loudly** when assumptions are violated (e.g. no overlap between datasets, insufficient data for CV learning), so that issues are caught early.

remember that each matrix experiment should follow a specific path based on the workflow. for example when the temp in the E0 are the same we dont need to do the reweight because the simulations are similar with temp, but when the simulations are with overlapping ladders and disjoin ladders we need to find the temp to make them all equal to to make the analysis.

## Experiment Matrix
- **E0 – Homogeneous (same T)**: 5 shards simulated at the **same temperature ladder** (identical β ladder, or effectively a single temperature). This is a baseline sanity check with no thermodynamic heterogeneity. Expect near-uniform weights at the target T (no reweight needed), and DeepTICA should train on a straightforward dataset. MBAR reduces to identity weighting.
- **E1 – Overlapping ladders**: 5 shards, each simulated from a **slightly shifted temperature ladder** with overlap around a common reference temperature _T<sub>ref</sub>_. This tests the reweighting in a positive scenario: there is sufficient overlap for TRAM/MBAR to combine data. DeepTICA is applied across these shards to learn slow CVs from the aggregated data (which spans slightly different thermodynamic states but shares _T<sub>ref</sub>_). Expect the reweighting to succeed and produce a consistent ensemble at _T<sub>ref</sub>_ with reasonable uncertainty, and the learned CV to capture common slow modes.
- **E2 – Disjoint ladders (negative control)**: _Optional._ 5 shards from temperature ladders with **no overlap** near _T<sub>ref</sub>_. This is a stress test for the reweighting and pipeline guardrails. Expect the reweighter to raise an error or fail-fast (or yield an extremely low effective sample size) due to negligible overlap – the acceptance report should mark **FAIL** with a clear reason. DeepTICA, if attempted, likely cannot learn a meaningful combined CV here (each shard explores a different thermodynamic ensemble); the pipeline should either skip the ML CV step or flag that the data are incompatible. This experiment validates that our failure paths are working (i.e. we don’t silently produce spurious results when data cannot be merged).

## Data Scale
- Aim for **≥ 25k–50k total transitions** per experiment so that guardrail thresholds (e.g. `total_pairs_lt_5000`) are not triggered spuriously. A practical recipe is **5 shards × 10k frames** each (with lag > 1) – or 5 × 4k frames if your discretization yields very dense transitions. This ensures robust statistics for MSM and adequate data for reweighting.
- Fix RNG seeds everywhere (simulation mocker, train/val splits, model inits) to ensure reproducibility. Deterministic outputs will make it easier to detect regressions: any deviation in weights, CVs, or FES beyond tolerance will signal a change in behavior.
- **DeepTICA considerations**: Ensure each shard is long enough to yield time-lagged pairs for the ML CV training. In practice, each shard’s length should exceed the chosen lag time (e.g. lag 5 frames) so that `pairs_total > 0`. Starting with a small lag (e.g. 2–5) is recommended to guarantee some pairs on all shards. If `pairs_total == 0` for a given lag, consider aggregating more frames or using a shorter lag; the pipeline’s fallback ladder (if configured) will try smaller lags automatically. By having at least a few thousand pairs, we improve the stability of DeepTICA training and subsequent FES/MSM analysis.

## structure
```
app_usecase/app/app_inputs/
└─ experiments/
   ├─ same_temp_shards/               # E0 inputs
   │  ├─ shard_000.npz
   │  ├─ ...
   │  └─ manifest.yaml
   ├─ mixed_ladders_shards/           # E1 inputs
   │  ├─ shard_000.npz
   │  ├─ ...
   │  └─ manifest.yaml
   ├─ disjoint_ladders_shards/        # E2 inputs (optional)
   │  ├─ shard_000.npz
   │  ├─ ...
   │  └─ manifest.yaml
   ├─ configs/
   │  ├─ transform_plan.yaml          # includes DeepTICA settings (e.g. lag time, fallback)
   │  ├─ discretize.yaml
   │  ├─ reweighter.yaml              # MBAR/TRAM knobs + β grid (T ladder definition)
   │  └─ msm.yaml
   └─ refs/                           # reference data for truth-checks
      └─ long_run_Tref/summary.npz    # e.g. a long unbiased run at Tref for comparison

app_usecase/app/experiment/
├─ run_same_temp.py                   # script_1 for E0
├─ run_mixed_ladders.py               # script_2 for E1
├─ run_disjoint_ladders.py            # script_3 for E2 (if implemented)
└─ common.py                          # shared helpers (loading data, running analysis steps, etc.)

app_usecase/app/experiment_outputs/
├─ same_temp_output/                  # E0 results
│  ├─ analysis_debug/summary.json     # includes detailed diagnostics (incl. mlcv_deeptica artifact)
│  ├─ fes_Tref.png                    # FES plot at T_ref (for original vs learned CVs)
│  ├─ msm_summary.json                # MSM metrics (e.g. timescales, populations)
│  ├─ weights_summary.json            # Reweighting results (should be trivial in E0)
│  └─ acceptance_report.md            # PASS/FAIL report with criteria checks
├─ mixed_ladders_output/              # E1 results (similar files as above)
│  ├─ analysis_debug/summary.json     # (expect non-uniform weights, TRAM output, etc.)
│  └─ ...
└─ disjoint_ladders_output/           # E2 results (if run; likely a failure report)
   ├─ analysis_debug/summary.json     # might contain partial results or error logs
   └─ ...

```

## what experiment must prove

**E0 (Homogeneous, no reweight needed):** This scenario tests that with identical conditions, the pipeline yields consistent results without relying on reweighting.
- **Weights**: Each shard’s weight factor _w<sub>frame</sub>_ at the target temperature should be ~1.0 (since all data already at _T<sub>ref</sub>_). If normalized, the weight distribution across frames should be nearly uniform (e.g. coefficient of variation < 5%). Essentially, MBAR/TRAM should have nothing to do – we expect the reweighter output to be an identity mapping (all frames equally likely).
- **MSM**: The aggregated trajectory (all shards combined at same T) should produce a well-connected Markov State Model. Check that the largest strongly connected component (SCC) covers ≥ 90% of the data, meaning most frames fall into a single ergodic set. The fraction of “empty” microstates (states with virtually no assignments) should be ≤ 30% (indicating our discretization is not too fine or data-sparse). The post-lag transition matrix should not be overly diagonal – i.e. the sum of self-transition probabilities (diagonal mass) should be ≤ 0.95, ensuring that we captured some dynamics and not just trivial self-transitions.
- **Consistency**: If we build MSMs on each shard separately (at the same T), their stationary distributions should be very similar to one another and to the merged-shard MSM. Quantitatively, the stationary probability distributions for each shard vs. the combined set should have a **JS divergence ≤ 0.02** (or another similarity metric like **EMD ≤ 0.03**) on the discretized state space. This confirms that all shards indeed sampled the same underlying distribution (no drift or run-to-run variation) and that merging them is statistically valid.
- **FES**: The free energy surfaces computed from each shard (or from the merged data) should coincide. Take the FES from the merged model at _T<sub>ref</sub>_ and compare it to the median FES of individual shards. They should agree within **RMSE ≤ 0.2 k<sub>B</sub>T** on all well-sampled bins. In other words, no single shard’s FES deviates significantly from the others – a sign of convergence. Any differences should be within expected statistical fluctuation.
- **DeepTICA**: _Data-driven CV learning should succeed and not alter the physics._ With homogeneous-temperature shards, DeepTICA training is expected to complete (`applied=True`, `reason="ok"`) since there are ample time-lagged pairs available. Verify that `pairs_total` is non-zero (and reasonably large, given the data volume) in the DeepTICA diagnostics. The learned CVs (likely capturing the slowest process in the simulation) should produce results consistent with the original CV space. Concretely, the MSM and FES obtained using the DeepTICA CV should not significantly differ from those using the original CV features. We can check that the stationary distribution from the DeepTICA-based MSM is close to that from the original MSM (e.g. JS divergence ≤ 0.02 as above), and the FES on the DeepTICA CV coordinate (transformed back or projected to original space) does not exceed an RMSE of 0.2–0.3 _k<sub>B</sub>T_ from the baseline. Additionally, monitor MSM metrics on the learned CV: SCC coverage should remain high (≈0.9) and other guardrails stay within limits, indicating that the learned CV hasn’t broken connectivity. _(Rationale: DeepTICA should find a coordinate that preserves the essential slow dynamics. In a well-behaved scenario like E0, it shouldn’t drastically reshape the free energy landscape – any improvement would be subtle, e.g. slightly clearer state separation, but the thermodynamics must remain consistent.)_
**E1 (Overlapping ladders, with TRAM/MBAR reweight):** This experiment must demonstrate that we can combine slightly different simulation conditions into one coherent analysis at _T<sub>ref</sub>_.
- **Reweight feasibility**: There must be sufficient thermodynamic overlap between shards. Check that adjacent temperature ladders share common states or configurations around _T<sub>ref</sub>_. In practice, verify that the reweighting algorithm (TRAM/MBAR) converges without error. A key indicator is the **effective sample size (ESS)**: when reweighting all shards to _T<sub>ref</sub>_, ESS should be ≥ 0.3 × N (30% of total frames) as a rule-of-thumb. This ensures we have enough statistical weight after reweighting (if ESS is too low, the reweighted distribution is dominated by a few frames, implying poor overlap). We expect TRAM to report non-zero exchange probabilities between neighboring temperature windows, confirming overlap.
- **Weights sanity**: Inspect the weights output by TRAM/MBAR for each shard. All weights should be finite and non-negative (no NaNs or negatives). When normalized, the sum of weights originating from each shard should roughly reflect that shard’s intended contribution. For example, if all shards cover _T<sub>ref</sub>_ equally well, each might contribute ~20% of the total weight (5 shards); if some shards are closer to _T<sub>ref</sub>_ in temperature, they may contribute a bit more, but each shard’s weight fraction should be within ~10% of the expected uniform share (unless deliberately different lengths). Large deviations or one shard getting near-zero weight would indicate either overlap issues or uneven sampling – something to flag.
- **Truth check**: We have either a long reference simulation at _T<sub>ref</sub>_ or an analytic expectation for the distribution at _T<sub>ref</sub>_. Compare the **reweighted distribution** (or FES) obtained from the combined shards against this reference. Use similar metrics as in E0: the Jensen-Shannon divergence between the reweighted stationary distribution and the reference distribution should be ≤ 0.05 (a slightly looser threshold since we are combining data). Equivalently, the RMSE of the FES compared to reference should be ≤ 0.4 _k<sub>B</sub>T_ on well-populated regions. This tolerates some uncertainty but ensures the reweighted result is essentially correct. If the combined analysis cannot reproduce the known _T<sub>ref</sub>_ behavior within these bounds, then the reweighting or input data might be flawed.
- **MSM guardrails**: After reweighting and building the MSM at _T<sub>ref</sub>_, the same structural checks from E0 apply. Largest SCC should cover ≥ 90% of the reweighted trajectory frames, indicating good state connectivity when projected to _T<sub>ref</sub>_. The fraction of empty microstates should remain ≤ 30%, and diagonal of the transition matrix ≤ 0.95 after applying the chosen lag. Essentially, even after merging data from slightly different temperatures, the resulting MSM at _T<sub>ref</sub>_ should behave like a proper Markov model of a single long simulation at that temperature.
- **DeepTICA**: _Machine-learned CV integration under varied conditions._ DeepTICA is run on the collection of shards (which span a range of temperatures around _T<sub>ref</sub>_). We expect the training to still complete successfully: each shard can yield time-lagged pairs (since they are sufficiently long and lag is small), though note that **cross-shard pairing** is disabled for different temperatures (the algorithm will not create pairs between frames of different temperature shards). Ensure that **each shard contributes pairs** – i.e., check in the DeepTICA diagnostics that none of the shards reports zero pairs (this would indicate a possible issue like an extremely short shard or mis-set lag). The total pairs count should be reasonable (e.g. on the order of the combined frame count times a factor for lag).

The learned DeepTICA CV in this multi-temperature scenario should ideally capture a slow mode relevant to _T<sub>ref</sub>_. To validate this, we compare the reweighted results using the DeepTICA CV against the reference and against the original CV results:

- The reweighted FES at _T<sub>ref</sub>_ computed on the DeepTICA CV coordinate should still match the reference FES within similar bounds (JS divergence ≤ 0.05, FES RMSE ≤ 0.4 _k<sub>B</sub>T_). Any improvement or change can be noted (e.g. perhaps DeepTICA CV yields slightly lower JS due to focusing on slow variables, which would be a positive sign).
    
- The MSM built on the DeepTICA CV should pass the same guardrails (SCC, etc.), confirming that the learned CV did not break connectivity. If DeepTICA indeed found a better reaction coordinate, we might observe **higher implied timescales** or better state separation in the MSM. This isn’t a formal requirement for the test, but it’s an interesting outcome to monitor (e.g. compare the slowest implied timescale or spectral gap between original CV vs DeepTICA CV MSM).
    
- Also check weight consistency: since DeepTICA doesn’t directly affect weights (that’s purely from TRAM/MBAR), this is more about ensuring that using a different CV for analysis post-reweight doesn’t change the population distribution. The stationary distribution from the DeepTICA-based analysis at _T<sub>ref</sub>_ should be very close to that from the original analysis (they’re both attempting to estimate the same truth, just via different feature transformations).
    

Overall, E1 with DeepTICA proves that even with heterogeneous input data, the pipeline can learn a collective variable and produce a biased analysis that is consistent with known results. If any part of DeepTICA fails here (e.g. if the combined data somehow confuses the trainer), it should be caught: for example, if shards were too short or mis-specified, the `mlcv_deeptica` artifact might show `skipped=True` with a reason like "no_pairs" or an exception. In such a case, the acceptance report should flag it, because we expect DeepTICA to work given the design of E1 (with overlapping ladders and adequate data).


**E2 (Disjoint ladders, negative control):** This scenario intentionally violates the assumptions to ensure the pipeline’s safeguards trigger.
- **Expected failure**: The TRAM/MBAR reweighting should **not** be able to combine shards that have no overlapping thermodynamic states. We anticipate one of two outcomes: (a) the reweighter raises an error or exception (fail-fast) indicating insufficient overlap or singular matrix, or (b) it returns results with an extremely low ESS and very high uncertainty, effectively warning that the outcome is unreliable. In either case, the **acceptance_report.md** for E2 must mark a **FAIL** and clearly state the reason (e.g. _“Reweighting failed: no overlap between input temperature ladders”_ or _“Effective sample size nearly zero – insufficient data merging possible”_). This validates that our guardrails against incompatible data are working. If, unexpectedly, the reweighting does produce an output (perhaps by extrapolation), we need to scrutinize that – but the expectation is that it shouldn’t silently proceed.
- **DeepTICA**: If the ML CV step is attempted in this scenario, it likely will not yield meaningful results either. Since each shard is at a vastly different thermodynamic state, there is no single slow mode that spans all shards. We might even configure this experiment such that DeepTICA is effectively skipped – for example, by using an obviously incompatible lag or by noticing that each shard’s data is unrelated. Ideally, the pipeline should either **skip the DeepTICA training** (setting `skipped=True` with a `reason` like "no_records" or a custom flag because the input shards aren’t mergeable) or run it per shard without combining (each shard would yield its own trivial slow mode, which doesn’t generalize). In any event, no “combined” deep CV should be trusted here. The acceptance criteria for E2 include verifying that **any failure in DeepTICA is also caught or reported**: e.g., if DeepTICA outputs an `exception` or `no_pairs`, that should appear in the debug summary and be noted. However, the primary failure condition remains the reweighting. We consider E2 a pass (of the test) if the pipeline correctly aborts or flags the analysis as invalid – we **want** it to fail in a controlled way.