import runner, generic_alp

runner.run_cw_mR_vs_mphi()
runner.run_cw_Gammainf_vs_mphi()
runner.run_generic_alp(0)
runner.run_generic_alp(1)
runner.run_generic_alp(2)
generic_alp.compute_correct_curves(1)
generic_alp.compute_correct_curves(2)
generic_alp.compute_correct_curves(3)
generic_alp.recompute_all_dilutions()
generic_alp.compute_all_example_trajectories()
