include("coupled_fields.jl")

############################## oscillation start (first zero crossing) ############################
function _find_H_osc(M, G, initial_ratio, H0, Hmax, divisions; debug=false)
    @show (G, M, initial_ratio, H0, Hmax)
    tmax = H_to_t(Hmax, H0)
    problem = ODEProblem(coupled_fields(M, G, initial_ratio, H0), (t0, tmax))
    print("solving...")
    sol = solve(problem, alg; settings..., saveat=tmax/(10*divisions))
    println(" done!")
    root_fn1(t) = sol(t)[2]
    root_fn2(t) = sol(t)[4]
    dt = tmax / divisions
    for i = 1:divisions
        t_start = (i - 1)*dt
        t_end = i*dt
        found1 = false
        found2 = false
        H1 = -1
        H2 = -1
        # root of phi1
        if sign(root_fn1(t_start)) != sign(root_fn1(t_end))
            t_osc = find_zero(root_fn1, (t_start, t_end), Roots.A42())
            H1 = sol(t_osc)[1]
            found1 = true
        end
        # root of phi2
        if sign(root_fn2(t_start)) != sign(root_fn2(t_end))
            t_osc = find_zero(root_fn2, (t_start, t_end), Roots.A42())
            H2 = sol(t_osc)[1]
            found2 = true
        end
        # return the earlier one if both are found
        H_osc = if found1 && found2
                    max(H1, H2)
                elseif found1
                    H1
                elseif found2
                    H2
                else
                    -1
                end
        if H_osc > 0
            if debug
                figure()
                axvline(H_to_t(H_osc, H0), color="black", ls="--", label="H_osc")
                plot(sol.t, sol[2,:], label="phi1")
                plot(sol.t, sol[4,:], label="phi2")
                xlabel("t")
                ylabel("fields")
            end
            return H_osc
        end
    end
    return nothing
end

function calc_H_osc_analytical(M, G, initial_ratio, n)
    return n * max(sqrt(1 + G*initial_ratio^2), sqrt(M + G))
end

function find_H_osc(M, G, initial_ratio;
        H0_start=calc_H_osc_analytical(M, G, initial_ratio, 20), Hmax_start=calc_H_osc_analytical(M, G, initial_ratio, 1/10),
        nmaxtrys=100, divisions=1000, inc_factor=5, check_factor=5, epsilon=1e-2)
    Hmax = Hmax_start
    H0 = H0_start
    for i = 1:nmaxtrys
        H_osc = _find_H_osc(M, G, initial_ratio, H0, Hmax, divisions)
        if H_osc != nothing
            for j = i:nmaxtrys
                H0 *= check_factor
                next_H_osc = _find_H_osc(M, G, initial_ratio, H0, Hmax, divisions)
                if next_H_osc == nothing
                    return NaN
                end
                rel_change = abs(H_osc - next_H_osc) / next_H_osc
                @show (H_osc, next_H_osc, rel_change, epsilon)
                if rel_change < epsilon
                    return next_H_osc
                end
                H_osc = next_H_osc
            end
            return NaN
        end
        H0 *= inc_factor
        Hmax /= inc_factor
    end
    return NaN
end

H_osc_filename(M, initial_ratio) = "coupled_fields_H_osc_M=$(M)_initial_ratio=$(initial_ratio).dat"

function compute_H_osc()
    @time for initial_ratio in initial_ratio_range
        for M in M_range
            H_osc_list = find_H_osc.(M, G_range, initial_ratio)
            writedlm(H_osc_filename(M, initial_ratio), [G_range H_osc_list])
        end
    end
end

function plot_H_osc(;n=1/6)
    figure(figsize=(10,5))
    for (i, initial_ratio) in enumerate(initial_ratio_range)
        subplot(2, 2, i)
        title(@sprintf("\$\\phi_1 / \\phi_2\$ = %.2f", initial_ratio))
        for (j, M) in enumerate(M_range)
            try
                data = readdlm(H_osc_filename(M, initial_ratio))
                G_range, H_osc_list = data[:, 1], data[:, 2]
                l = plot(G_range, abs.(H_osc_list), label=@sprintf("sim. M = %.2f", M))
                H_osc_analytical = calc_H_osc_analytical.(M, G_range, initial_ratio, n)
                plot(G_range, H_osc_analytical, color=l[1].get_color(),
                     label=j == 1 ? "analytical" : nothing, ls="--")
            catch
            end
        end
        xscale("log")
        yscale("log")
        xlabel("G")
        ylabel(raw"$H_\mathrm{osc}$")
        if i == 1
            legend(ncol=2)
        end
    end
    tight_layout()
    savefig("H_osc_plots.pdf")
end
