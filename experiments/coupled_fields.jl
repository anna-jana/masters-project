using DynamicalSystems
using PyPlot
using LinearAlgebra
using LsqFit
using Statistics
using Printf
using OrdinaryDiffEq
using DelimitedFiles
using Random

############################ the model #############################
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

function calc_energies(sys, orbit)
    M, G = sys.p
    H, phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    rho1 = @. 0.5*(phi1_dot^2 + phi1^2)
    rho2 = @. 0.5*M*(phi2_dot^2 + phi2^2)
    pot  = @. G*phi1^2*phi2^2
    total = @. rho1 + rho2 + pot
    return rho1, rho2, pot, total
end

function calc_eff_masses(sys, orbit)
    return @.(orbit[:,4]^2 + 1), @.(orbit[:,2]^2 + sys.p[1])
end

model(x, p) = @. p[1] * x + p[2]
H_to_t(H, H_start) = 0.5*(1/H - 1/H_start)

################################### basic analysis ##################################
function simulate(M, G, initial_ratio, H_start; nsteps=1000, H_end=M/3, H_fit_start=M/2)
    sys = coupled_fields(M, G, initial_ratio, H_start)
    tmax = H_to_t(H_end, H_start)
    dt = tmax / (nsteps - 1)
    ts = 0.0:dt:tmax
    orbit = trajectory(sys, tmax; Δt=dt, diffeq=(alg=AutoTsit5(Rosenbrock23())))
    rho1, rho2, pot, total = calc_energies(sys, orbit)
    t_fit_start = H_to_t(H_fit_start, H_start)
    i = ceil(Int, t_fit_start / dt)
    log_t = log.(ts)
    fit_res = curve_fit(model, @view(log_t[i:end]), log.(@view(total[i:end])), [1.0, 0.0])
    p = 2*fit_res.param[1] # power law rho ~ a^panalyse
    m1, m2 = calc_eff_masses(sys, orbit)
    return (sys=sys, ts=ts, H=orbit[:,1],
            phi1=orbit[:,2], phi2=orbit[:,4], phi1_dot=orbit[:,3], phi2_dot=orbit[:,5],
            rho1=rho1, rho2=rho2, pot=pot, total=total,
            p=p, m1=m1, m2=m2, t_fit_start=t_fit_start, fit_res=fit_res,
            M=M, G=G, initial_ratio=initial_ratio, H_start=H_start)
end

function plot_sim(sim)
    shape = (2, 3)
    figure()

    subplot2grid(shape, (1, 1))
    plot(sim.H, sim.phi1, label=raw"$\phi_1$")
    plot(sim.H, sim.phi2, label=raw"$\phi_2$")
    xscale("log")
    xlabel("H")
    ylabel("fields")
    legend(ncol=2)

    subplot2grid(shape, (1, 2))
    plot(sim.phi1, sim.phi2)
    xlabel(raw"$\phi_1$")
    ylabel(raw"$\phi_2$")

    subplot2grid(shape, (2, 1))
    loglog(sim.ts, sim.total, label="simulation data")
    loglog(sim.ts, exp.(model(log.(sim.ts), sim.fit_res.param)), label="power law fit")
    axvline(sim.t_fit_start, color="black")
    axvline(sim.ts[end], color="black")
    xlabel("t")
    ylabel("total energy of the coupled fields")
    legend()
    title(@sprintf("\$\\rho_\\mathrm{total} \\sim a^{%.2f}\$", sim.p))

    subplot2grid(shape, (2, 2))
    plot(sim.rho1 ./ sim.total, sim.rho2 ./ sim.total, lw=0.3)
    xlabel(raw"$\rho_{\mathrm{kin}, 1} / \rho_\mathrm{total}$")
    ylabel(raw"$\rho_{\mathrm{kin}, 2} / \rho_\mathrm{total}$")

    subplot2grid(shape, (3, 1))
    plot(sim.H, sim.m1, label="m1")
    plot(sim.H, sim.m2, label="m2")
    xscale("log")
    yscale("log")
    ylim(min(minimum(sim.m1[2:end]), minimum(sim.m2[2:end])),
         max(maximum(sim.m1[2:end]), maximum(sim.m2[2:end])))
    legend()

    suptitle("M = $(sim.M), G = $(sim.G), \$\\phi_1 / \\phi_2\$ = $(sim.initial_ratio), \$H_0\$ = $(sim.H_start)")
    tight_layout()
end


############################## analysis of the energy scaling ##########################
function compute_p(M, G;
        debug=false, nsamples=1, window=4, nsteps=200, initial_ratio=1 + 1e-2,
        H_start=1e2*M, H_fit_start=1e-1*M, H_end=1e-2*M,
        alg=SimpleATsit5(), reltol=1e-6, abstol=1e-6)
    @show G
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
        _, _, _, total = calc_energies(sys, orbit)
        y = log.(@view(total[i:end]))
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
            plot(log.(ts), log.(total))
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

function lya_G_dep(; M=1.0, initial_ratio=1 + 1e-2, H0=1e3, nsteps=50, G_range=10 .^ (-2:0.5:8), log_scale=true)
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
end
