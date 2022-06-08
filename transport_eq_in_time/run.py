import runner, generic_alp

for n in range(3):
    runner.run_generic_alp(n)
    generic_alp.compute_correct_curves(n + 1)
    runner.run_cw_mR_vs_mphi(n)
    runner.run_cw_Gammainf_vs_mphi(n)
generic_alp.recompute_all_dilutions()
generic_alp.compute_all_example_trajectories()