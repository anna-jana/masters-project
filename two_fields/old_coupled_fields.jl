using DynamicalSystems
using PyPlot
using LinearAlgebra
using StatsBase
using OrdinaryDiffEq
using LsqFit
using Statistics

hubble(t) = 1 / (2*t) # TODO: consider different epochs including reheating

function coupled_fields_rhs(s, p, t)
    H = hubble(t)
    phi1, phi1_dot, phi2, phi2_dot = s
    m1, m2, f1, f2 = p
    return SVector(
        phi1_dot,
        - 3*H*phi1_dot - f2*phi1*phi2^2 - m1*phi1,
        # - 3*H*phi1_dot - f2*g*phi1*phi2^2 - m1*phi1,
        phi2_dot,
        - 3*H*phi2_dot - f1*phi2*phi1^2 - m2*phi2,
        # - 3*H*phi2_dot - f1*g*phi2*phi1^2 - m2*phi2,
    )
end

names = ["\$\\phi_1\$", "\$\\dot{\\phi}_1\$", "\$\\phi_2\$", "\$\\dot{\\phi}_2\$"]

function sim(m1, m2, f1, f2;
        t0 = 1e-3, tspan = 1e5, ttr = 0.0,
        default_initial = [1.0, 0.0, 1.0, 0.0], dt = 1.0,
        solver_options = (abstol = 1e-6, reltol = 1e-6,
                          alg = AutoTsit5(Rosenbrock23())))
    default_params = [m1, m2, f1, f2]
    ds = ContinuousDynamicalSystem(coupled_fields_rhs,
                                   default_initial, default_params, t0=t0)
    ts = (t0 + ttr):dt:(t0 + ttr + tspan)
    orbit = trajectory(ds, tspan, Δt=dt, t0=t0, Ttr=ttr; solver_options...)
    return ts, orbit, ds
end

function skip_transient(ts, orbit, ttr)
    dt = ts[2] - ts[1]
    start_index = Int(floor(ttr / dt)) + 1
    return ts[start_index:end], orbit[start_index:end]
end

# TODO: figures file output

# plot field evolutions
function plot_field_evolution(ts, orbit)
    figure()
    plot(ts, orbit[:, 1], label=names[1])
    plot(ts, orbit[:, 3], label=names[3])
    # xscale("log")
    xlabel("t")
    legend()
end

# plot 2D projections of the trajectory in state space
function plot_2d_traj_projections(ts, orbit)
    fig, axes = subplots(3, 2)
    k = 1
    for (i, name1) in enumerate(names)
        for (j, name2) in enumerate(names)
            if j < i
                axes[k].plot(orbit[:, i], orbit[:, j])
                axes[k].set_xlabel(name1)
                axes[k].set_ylabel(name2)
                k += 1
            end
        end
    end
    tight_layout()
end

# plot 3D projects of the trajectory in state space
function plot_3d_traj_projections(ts, orbit)
    fig = figure()
    n = 1
    for (i, name1) in enumerate(names)
        for (j, name2) in enumerate(names)
            for (k, name3) in enumerate(names)
                if i < j < k
                    ax = fig.add_subplot(2, 2, n, projection="3d")
                    ax.plot(orbit[:, i], orbit[:, j], orbit[:, k])
                    ax.set_xlabel(name1)
                    ax.set_ylabel(name2)
                    ax.set_zlabel(name3)
                    n += 1
                end
            end
        end
    end
    tight_layout()
end

# plot poincare sections for all state variables to equal 0
# and plot for each psos two of the remaining coordinates
function plot_poincare_section(ts, orbit)
    # this doenst work for some reason
    # psos = poincaresos(ds, (var_num, crossing_val), tfinal=1e7, Ttr=1e3)
    fig, axes = subplots(3, 4)
    for var_num = 1:4
        psos = poincaresos(orbit, (var_num, crossing_val))
        k = 1
        for i = 1:4, j=1:4
            if i < j && i != var_num && j != var_num
                if k == 1
                    axes[1, var_num].set_title(names[var_num] * " = 0")
                end
                axes[k, var_num].plot(psos[:, i], psos[:, j], ".", ms=0.1)
                axes[k, var_num].set_xlabel(names[i])
                axes[k, var_num].set_ylabel(names[j])
                k += 1
            end
        end
        suptitle("Poincare section")
        tight_layout()
    end
end

calc_pot(phi1, phi2, f1, f2) = @. f1*f2*phi1^2*phi2^2

function calc_energies(orbit, ds)
    m1, m2, f1, f2 = ds.p
    phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    rho1 = (@. 0.5*f1*phi1_dot^2 + 0.5*m1*f1*phi1^2)
    rho2 = (@. 0.5*f2*phi2_dot^2 + 0.5*m2*f2*phi2^2)
    pot  = calc_pot(phi1, phi2, f1, f2)
    total = rho1 + rho2 + pot
    return rho1, rho2, pot, total
end

obs_names = ["\$\\rho_1\$", "\$\\rho_2\$", "\$\\rho_\\mathrm{pot}\$", "\$\\rho_\\mathrm{total}\$"]

# plot the total energy TODO: should this be conserved or drop somehow because of hubble friction?
function plot_energy(ts, orbit, ds)
    rho1, rho2, pot, total = calc_energies(orbit, ds)
    # timeseries plot of the energy densities
    fig, axes = subplots(2, 2)
    for (k, ob) in enumerate([rho1, rho2, pot, total])
        axes[k].plot(ts, ob, lw=0.5)
        axes[k].set_xscale("log")
        axes[k].set_yscale("log")
        axes[k].set_xlabel("t")
        axes[k].set_ylabel(obs_names[k])
    end
    tight_layout()

    # plot all energy components against each other
    figure()
    obs = [rho1, rho2, pot]
    k = 1
    for i = 1:3
        for j = 1:3
            if i < j
                subplot(2, 2, k)
                plot(obs[i], obs[j], lw=0.3, color="black", label="trajectory")
                xlabel(obs_names[i])
                ylabel(obs_names[j])
                # plot([obs[i][1]], [obs[j][1]], "o", label="start")
                # plot([obs[i][end]], [obs[j][end]], "o", label="end")
                # legend()
                k += 1
            end
        end
    end
    subplot(2, 2, k)
    plot(rho1 ./ total, rho2 ./ total, lw=0.3, color="black")
    xlabel("\$\\rho_1 / (\\rho_1 + \\rho_2 + \\rho_\\mathrm{pot})\$")
    ylabel("\$\\rho_2 / (\\rho_1 + \\rho_2 + \\rho_\\mathrm{pot})\$")
    tight_layout()

    # 3d plot of all energy components
    # fig = figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.plot(rho1, rho2, pot)
    # ax.set_xlabel("free part of \$\\phi_1\$")
    # ax.set_ylabel("free part of \$\\phi_2\$")
    # ax.set_zlabel("shared potential energy")
    # ax.plot([rho1[1]], [rho2[1]], [pot[1]], "o")
    # ax.plot([rho1[end]], [rho2[end]], [pot[end]], "o")
end


function plot_orbit_with_pot(ts, orbit, ds)
    phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    m1, m2, f1, f2 = ds.p

    num = 200
    a = maximum(phi1)
    b = minimum(phi1)
    margin = (a - b)*1e-1
    phi1_range = range(b - margin, a + margin, length=num)
    a = maximum(phi2)
    b = minimum(phi2)
    margin = (a - b)*1e-1
    phi2_range = range(b - margin, a + margin, length=num)
    V = [calc_pot(phi1, phi2, f1, f2) for phi2 in phi2_range, phi1 in phi1_range]
    pcolormesh(phi1_range, phi2_range, log.(V), shading="nearest")
    colorbar(label="\$\\log(V = f_1 f_2 \\phi_1^2 \\phi_2^2)\$")

    plot(phi1, phi2, color="red")

    xlabel(names[1])
    ylabel(names[3])
end

function sensitivity_on_initial_condition(t0, ttr, tspan, ds)
    lambda1 = lyapunov(ds, tspan, t0=t0, Ttr=ttr)
    println(lambda1)
    pertubation = ones(length(ds.u0)) * 1e-5
    ts = (t0 + ttr):dt:(t0 + ttr + tspan)
    orbit1 = trajectory(ds, tspan, Δt=dt, t0=t0, Ttr=ttr)
    orbit2 = trajectory(ds, tspan, ds.u0 + pertubation, Δt=dt, t0=t0, Ttr=ttr)
    dist = [norm(a - b) for (a, b) in zip(orbit1, orbit2)]
    figure()
    subplot(2,1,1)
    semilogy(ts, dist, label="simulated")
    plot(ts, @.(exp.(lambda1 * (ts - ts[1]) + log(dist[1]))), label="lyapunov")
    ylabel("distance between neighboring trajectories")
    xlabel("t")
    ylim(extrema(dist)...)
    legend()
    subplot(2,1,2)
    plot(ts, orbit1[:, 1])
    plot(ts, orbit2[:, 1])
    xlabel("t")
    ylabel(names[1])
end

function calc_eos(orbit, ds)
    rho1, rho2, pot, total = calc_energies(orbit, ds)
    pressure = rho1 + rho2 - pot
    eos = pressure ./ total
    return eos
end

function plot_eos(ts, orbit, ds)
    eos = calc_eos(orbit, ds)
    mean_eos = mean(eos)
    std_eos = std(eos)
    bins = range(extrema(eos)..., length=30 + 1)
    hist = normalize(fit(Histogram, eos, bins), mode=:pdf)

    figure()
    subplot(2,1,1)
    plot(ts, eos, label="evolution")
    axhline(mean_eos, label="mean", color="black", ls="-")
    axhline(mean_eos - std_eos, label="error", color="black", ls="--")
    axhline(mean_eos + std_eos, color="black", ls="--")
    legend(framealpha=1.0)
    xlabel("t")
    ylabel("equation of state")
    subplot(2,1,2)
    step(hist.edges[1][1:end-1], hist.weights, label="histogram")
    axvline(mean_eos, color="black", ls="-", label="mean")
    axvline(mean_eos - std_eos, color="black", ls="--", label="error")
    axvline(mean_eos + std_eos, color="black", ls="--")
    xlabel("equation of state")
    ylabel("frequency")
    legend()
end

function fit_total_energy_power_law(ts, orbit, ds; plot_it=true)
    rho1, rho2, pot, total = calc_energies(orbit, ds)
    model(x, p) = @. p[1] * x + p[2]
    log_ts = log.(ts)
    fit_res = curve_fit(model, log_ts, log.(total), [1.0, 0.0])
    p = 2*fit_res.param[1] # power law rho ~ a^p
    # w_fit = -1/3*p - 1
    if plot_it
        figure()
        loglog(ts, total, label="simulation")
        loglog(ts, exp.(model(log_ts, fit_res.param)), label="fit")
        xlabel("t")
        ylabel(obs_names[4])
        legend()
    end
    return p # , w_fit
end

# test code to see if the fitting of the energy power law works
#function test_rhs(s, p, t)
#    m, = p
#    phi, phi_dot = s
#    return SVector(phi_dot, -3*hubble(t)*phi_dot - m^2*phi)
#end
#m = 1.0
#test_ds = ContinuousDynamicalSystem(test_rhs, [1.0, 0.0], [m], t0=t0)
#test_tspan = 100.0
#test_dt = 0.1
#test_ts = t0:test_dt:(t0 + test_tspan)
#phi, phi_dot = columns(trajectory(test_ds, test_tspan, t0=t0, Δt=test_dt))
#energy = @. 0.5*phi_dot^2 + 0.5*m^2*phi^2
#loglog(test_ts, energy)
#start_idx = findfirst(test_ts .> 2)
#model(x, p) = @. p[1] * x + p[2]
#fit_res = curve_fit(model, log.(test_ts[start_idx:end]), log.(energy[start_idx:end]), [1.0, 0.0])
#loglog(test_ts[start_idx:end], exp.(model(log.(test_ts[start_idx:end]), fit_res.param)))
#@show fit_res.param[1]*2

function chaos_test(ds; plot_it=true)
    # GALI_k test
    k = 2
    threshold = 1e-12
    tspan = 1e3
    g, t = gali(ds, tspan, k, threshold=threshold)
    if plot_it
        figure()
        loglog(t, g, label="GALI")
        axhline(threshold, label="threshold")
        xlabel("t")
        ylabel("GALI_$k")
        legend()
    end
    println("minimal gali reached:", minimum(g), "with threshold ", threshold, "stopped at ",
            t[end], " tmax = ", tspan, "should not be under the threshold within tmax")

    # expansion-entropy
    sampler, restrainer = boxregion(.-ones(4), ones(4))
    ee = expansionentropy(ds, sampler, restrainer)
    println("expansionentropy (should be positive for chaos) = ", ee)
end

function sample_trajectories(; tspan = 1e4, num_bins = 30, num_samples = 50, num_steps = 100.0, plot_traj = true, plot_hist = true)
    samples = Float64[]
    figure()
    dt = plot_traj ? tspan/num_steps : tspan/3.0
    ts = t0:dt:(t0 + tspan)
    for i = 1:num_samples
        u0 = [2*rand() - 1.0, 0.0, 2*rand() - 1.0, 0.0]
        orbit = trajectory(ds, tspan, u0, Δt=dt,
                           t0=t0; solver_options...)
        if plot_traj
            plot(ts, orbit[:, 1], lw=0.5, alpha=0.3, color="black")
        end
        push!(samples, orbit[end, 1])
    end
    if plot_traj
        xlabel("t")
        ylabel("\$\\phi_1\$")
        axhline(0.0, color="red")
    end
    if plot_hist
        a, b = extrema(samples)
        bins = a:(b - a)/num_bins:b
        hist = normalize(fit(Histogram, samples, bins), mode=:pdf)
        figure()
        step(hist.edges[1][1:end-1], hist.weights)
        xlabel("\$\\phi\$ final")
    end
    return hist, samples
end


