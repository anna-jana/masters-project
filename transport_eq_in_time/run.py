import runner, generic_alp

runner.run_generic_alp(0)
runner.run_generic_alp(1, m_a_min=1e5)
runner.run_generic_alp(2)
for n in range(3)
    generic_alp.compute_correct_curves(n + 1)
generic_alp.recompute_all_dilutions()
generic_alp.compute_all_example_trajectories()

runner.run_cw_mR_vs_mphi(n)
runner.run_cw_Gammainf_vs_mphi()
