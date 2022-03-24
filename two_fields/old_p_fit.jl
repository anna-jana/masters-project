
# FIXME: if p depends on initial_ratio then this makes no sense at all!!!
# function find_p(M, G)
#     @show (G, M)
#     Random.seed!(42)
#     ps = Float64[]
#     p_errs = Float64[]
#     param_guess = [1.0, 1.0]
#     nsamples = 100
#     @time for k = 1:nsamples
#         initial_ratio = 1 + 1e-1*(2*rand() - 1)
#         println("sample: $k, initial_ratio: $initial_ratio")
#         p, p_err, param_guess = _find_p(M, G, initial_ratio, param_guess, false)
#         push!(ps, p)
#         push!(p_errs, p_err)
#     end
#     p_err = max(sqrt(sum(p_errs.^2))/length(p_errs), nsamples == 1 ? -1 : std(ps))
#     p = mean(ps)
#     println("p = $p +/- $p_err")
#     return p, p_err
# end
#
# p_filename(M) = "p_fits_M=$M.dat"
#
# function compute_p()
#     @time for M in M_range
#         p_list = find_p.(M, G_range)
#         writedlm(p_filename(M), [G_range [p[1] for p in p_list] [p[2] for p in p_list]])
#     end
# end
#
# function plot_p_fits()
#     for M in M_range
#         data = readdlm(p_filename(M))
#         G_range, ps, p_errs = data[:,1], data[:,2], data[:,3]
#         errorbar(G_range, ps, yerr=p_errs, capsize=3.0, fmt="-x", label="M = $M")
#     end
#     axhline(-3, label="matter", color="black", ls="--")
#     axhline(-4, label="radiation", color="black", ls=":")
#     xscale("log")
#     xlabel("G")
#     ylabel("p")
#     legend()
# end

