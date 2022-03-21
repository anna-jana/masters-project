using DynamicalSystems
using PyPlot
using LinearAlgebra
using LsqFit
using Statistics
using Printf
using OrdinaryDiffEq
using DelimitedFiles
using Random
using Roots

################################ the model ##################################
function coupled_fields_rhs(s, p, t)
    H, phi1, phi1_dot, phi2, phi2_dot = s
    M, G = p
    return SVector(
        -2*H^2,
        phi1_dot,
        - 3*H*phi1_dot - G*phi1*phi2^2 - phi1,
        phi2_dot,
        - 3*H*phi2_dot - G*phi2*phi1^2 - M*phi2,
    )
end

function coupled_fields(M, G, initial_ratio, H_start)
    initial = [H_start, 1.0, 0.0, initial_ratio, 0.0]
    return ContinuousDynamicalSystem(coupled_fields_rhs, initial, [M, G])
end

calc_pot(M, G, phi1, phi2) = G*phi1^2*phi2^2 + 0.5*phi1^2 + 0.5*M*phi2^2

function calc_energy(sys, orbit)
    M, G = sys.p
    H, phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    return @. 0.5*phi1_dot^2 + 0.5*M*phi2_dot^2 + calc_pot.(M, G, phi1, phi2)
end

function calc_eff_masses(sys, orbit)
    return @.(orbit[:,4]^2 + 1), @.(orbit[:,2]^2 + sys.p[1])
end

model(x, p) = @. p[1] * x + p[2]
H_to_t(H, H_start) = 0.5*(1/H - 1/H_start)

const alg = AutoTsit5(Rosenbrock23())
const settings = (reltol=1e-6, abstol=1e-6)

######################## plot trajectories and phasespace projections ###################
function plot_evolution(M, G, initial_ratio, H_start; nsteps=1000, H_end=M/3)
    tmax = H_to_t(H_end, H_start)
    t0 = H_to_t(H_start / 1.001, H_start)
    ts = [0.0; exp.(range(log(t0), log(tmax), length=nsteps - 1))]
    problem = ODEProblem(coupled_fields(M, G, initial_ratio, H_start) , (0, tmax))
    sol = solve(problem, alg; settings..., saveat=ts)
    H, phi1, phi1_dot, phi2, phi2_dot = sol[1,:], sol[2,:], sol[3,:], sol[4,:], sol[5,:]
    n = 100
    a = minimum(phi1)
    b = maximum(phi1)
    l = b - a
    phi1_range = range(a - l/10, b + l/10, length=n+1)
    a = minimum(phi2)
    b = maximum(phi2)
    l = b - a
    phi2_range = range(a - l/10, b + l/10, length=n)
    log10_V = [log10(calc_pot(M, G, a, b)) for b = phi2_range, a = phi1_range]

    figure()
    subplot(2,1,1)
    plot(H, phi1, label=raw"$\phi_1$")
    plot(H, phi2, label=raw"$\phi_2$")
    gca().invert_xaxis()
    xscale("log")
    xlabel("H")
    ylabel("fields")
    legend(ncol=2)
    subplot(2,1,2)
    pcolormesh(phi1_range, phi2_range, log10_V, shading="nearest", cmap="summer")
    colorbar(label=raw"$\log_{10}(V)$")
    plot(phi1, phi2, color="red")
    xlabel(raw"$\phi_1$")
    ylabel(raw"$\phi_2$")
    suptitle("M = $M, G = $G, \$\\phi_1 / \\phi_2\$ = $initial_ratio, \$H_0\$ = $H_start")
    tight_layout()
end


############################## analysis of the energy scaling ##########################
function compute_p(M, G;
        debug=false, nsamples=1, window=4, nsteps=200, initial_ratio=1 + 1e-2,
        H_start=1e2*M, H_fit_start=1e-1*M, H_end=1e-2*M,
        alg=SimpleATsit5(), reltol=1e-6, abstol=1e-6)
    @show (G, M, initial_ratio, H_start)
    Random.seed!(42)
    tmax = H_to_t(H_end, H_start)
    t_fit_start = H_to_t(H_fit_start, H_start)
    log_t_start = -5
    log_ts = range(log_t_start, log(tmax), length=nsteps)
    d_log_t = log_ts[2] - log_ts[1]
    ts = exp.(log_ts)
    i = ceil(Int, (log(t_fit_start) - log_t_start) / d_log_t + 1) # t = e^(log(t_start) + (i - 1) * d_log_t)
    x = @view(log_ts[i:end])
    _smooth_x = [mean(x[k-window:k+window]) for k = 1 + window : length(x) - window]
    ps = Float64[]
    p_errs = Float64[]
    param_guess = [1.0, 1.0]
    @time for k = 1:nsamples
        sys = coupled_fields(M, G, initial_ratio, H_start)
        problem = ODEProblem(sys, (0.0, tmax))
        sol = solve(problem, alg; saveat=ts, abstol=abstol, reltol=reltol)
        orbit = Dataset(collect(sol.u))
        energy = calc_energy(sys, orbit)
        y = log.(@view(energy[i:end]))
        smooth_y = [mean(y[k-window:k+window]) for k = 1 + window : length(x) - window]
        mask = isfinite.(smooth_y)
        smooth_x = _smooth_x[mask]
        smooth_y = smooth_y[mask]
        fit_res = curve_fit(model, smooth_x, smooth_y, param_guess)
        param_guess = fit_res.param
        p = 2*fit_res.param[1]
        p_err = 2*sqrt(estimate_covar(fit_res)[1, 1])
        push!(ps, p)
        push!(p_errs, p_err)
        if debug
            plot(log.(ts), log.(energy))
    title("M = $M, G = $G, \$\\phi_1 / \\phi_2\$ = $initial_ratio, \$H_0\$ = $H0")
            axvline(x[1], color="black")
            plot(smooth_x, smooth_y)
            plot(smooth_x, model(smooth_x, fit_res.param))
        end
        initial_ratio = 1 + 1e-1*(2*rand() - 1)
    end
    p_err = max(sqrt(sum(p_errs.^2))/length(p_errs), nsamples == 1 ? -1 : std(ps))
    p = mean(ps)
    println("p = $p +/- $p_err")
    return p, p_err
end

function plot_p_fits(G_range, p_list)
    errorbar(G_range, [p[1] for p in p_list], yerr=[p[2] for p in p_list], capsize=3.0, fmt="x")
    xscale("log")
    xlabel("G")
    ylabel("p")
end

function save_p_fits(G_range, p_list; filename="p_fits.dat")
    writedlm(filename, [G_range [p[1] for p in p_list] [p[2] for p in p_list]])
end

########################################## chaos tests ####################################
function lya_convergence(; M=1.0, G=1e6, initial_ratio=1 + 1e-2, H0=1e3, nsteps=50)
    sys = coupled_fields(M, G, initial_ratio, H0)
    lyapunov_spectra_steps, time = ChaosTools.lyapunovspectrum_convergence(sys, nsteps)
    lambda_max = ChaosTools.lyapunov(sys, time[end])

    figure()
    for i = 1:length(sys.u0)
        plot(time, [step[i] for step in lyapunov_spectra_steps], marker="o", label="\$\\lambda_$i\$")
    end
    axhline(0, color="black", label="0")
    axhline(lambda_max, color="tab:blue", ls="--", label=raw"$\lambda_\mathrm{max}$")
    legend(ncol=4)
    xlabel("time step")
    ylabel(raw"Lyapunov exponent, $\lambda_i$")
    title("M = $M, G = $G, \$\\phi_1 / \\phi_2\$ = $initial_ratio, \$H_0\$ = $H0")
end

function lya_G_dep(; M=1.0, initial_ratio=1 + 1e-2, H0=1e3, nsteps=3000, G_range=10 .^ (-2:0.05:7), log_scale=true)
    spectra = [lyapunovspectrum(coupled_fields(M, G, initial_ratio, H0), nsteps) for G in G_range]
    chaos_start_index = findfirst(x -> x > 0, [spectrum[1] for spectrum in spectra])
    figure()
    for i = 1:length(spectra[1])
        plot(G_range, [spectrum[i] for spectrum in spectra], label="\$\\lambda_$i\$")
    end
    plot(G_range, sum.(spectra), label="sum of lyapunov exponents", ls="--", color="red")
    if chaos_start_index != nothing
        axvline(G_range[chaos_start_index], color="black", ls="--", label="chaos start")
    end
    if log_scale
        xscale("log")
    end
    axhline(0, color="black")
    xlabel("G")
    ylabel(raw"Lyapunov spectrum $\lambda_i$")
    legend(ncol=2, framealpha=1)
    title("M = $M, G = $G, \$\\phi_1 / \\phi_2\$ = $initial_ratio, \$H_0\$ = $H0")
end


############################## oscillation start (first zero crossing) ############################
function _find_H_osc(M, G, initial_ratio, H0, Hmax, divisions)
    @show (G, M, initial_ratio, H0, Hmax)
    tmax = H_to_t(Hmax, H0)
    problem = ODEProblem(coupled_fields(M, G, initial_ratio, H0), (0, tmax))
    print("solving...")
    sol = solve(problem, alg; settings..., saveat=tmax/(10*divisions), maxiters=10^10)
    println(" done!")
    root_fn1(t) = sol(t)[2]
    root_fn2(t) = sol(t)[4]
    dt = tmax / divisions
    for i = 1:divisions
        t_start = (i - 1)*dt
        t_end = i*dt
        found1 = false
        found2 = false
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
        if found1 && found2
            return max(H1, H2)
        elseif found1
            return H1
        elseif found2
            return H2
        end
    end
    return nothing
end

function find_H_osc(M, G, initial_ratio;
        H0_start=1e2*M, Hmax_start=M/4, nmaxtrys=100, divisions=1000, inc_factor=5, check_factor=5, epsilon=1e-2)
    Hmax = Hmax_start
    H0 = H0_start
    for i = 1:nmaxtrys
        H_osc = _find_H_osc(M, G, initial_ratio, H0, Hmax, divisions)
        if H_osc != nothing
            for j = i:nmaxtrys
                H0 *= check_factor
                next_H_osc = _find_H_osc(M, G, initial_ratio, H0, Hmax, divisions)
                if next_H_osc == nothing
                    return -H_osc
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

G_range = (10 .^ (-2:0.2:6))
initial_ratio_range = [1, 1 + 1e-2, 10, 100]
M_range = [1e-2, 1e-1, 1, 10, 100]
H_osc_filename(M, initial_ratio) = "coupled_fields_H_osc_M=$(M)_initial_ratio=$(initial_ratio).dat"

function compute_H_osc()
    @time for initial_ratio in initial_ratio_range
        for M in M_range
            H_osc_list = find_H_osc.(M, G_range, initial_ratio)
            writedlm(H_osc_filename(M, initial_ratio), [G_range H_osc_list])
        end
    end
end

function plot_H_osc()
    for (i, initial_ratio) in enumerate(initial_ratio_range)
        subplot(2, 2, i)
        title(@sprintf("\$\\phi_1 / \\phi_2\$ = %.2f", initial_ratio))
        for M in M_range
            try
                data = readdlm(H_osc_filename(M, initial_ratio))
                G_range, H_osc_list = data[:, 1], data[:, 2]
                l = plot(G_range, abs.(H_osc_list), label=@sprintf("M = %.2f", M))
                # plot(G_range, H_osc_list2, color=l[1].get_color(), ls="--")
                xscale("log")
                yscale("log")
                xlabel("G")
                ylabel(raw"$H_\mathrm{osc}$")
                legend()
            catch
            end
        end
    end
    tight_layout()
end


######################################### run all the things #####################################
function main()
    # TODO: fix all parameters we want to show and
    compute_H_osc()
    plot_H_osc()
end
