using DynamicalSystems, PyPlot, LinearAlgebra, StatsBase

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

m1 = 0.0
m2 = 0.0
f1 = 1.0
f2 = 2.0 # TODO: scan parameter
t0 = 1e-3 # TODO: consider a specific initial hubble parameter


default_initial = [1.0, 0.0, 1.0, 0.0]
names = ["\$\\phi_1\$", "\$\\dot{\\phi}_1\$", "\$\\phi_2\$", "\$\\dot{\\phi}_2\$"]
default_params = [m1, m2, f1, f2]

ds = ContinuousDynamicalSystem(coupled_fields_rhs,
                               default_initial,
                               default_params, t0=t0)
tspan = 1e5
dt = 1.0
ts = t0:dt:(t0 + tspan)
if !(@isdefined orbit)
    println("**************** computing orbit *******************")
    orbit = trajectory(ds, tspan, Δt=dt, t0=t0)
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

const crossing_val = 0.0

ttr = 1e3
# for the poincare section:
#tspan_long = 1e6
#ts_long = (t0 + ttr):dt:(t0 + ttr + tspan_long)
#if !(@isdefined orbit_long)
#    println("**************** computing long orbit *******************")
#    orbit_long = trajectory(ds, tspan_long, Δt=dt, t0=t0, Ttr=ttr)
#end

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

# plot the total energy TODO: should this be conserved or drop somehow because of hubble friction?
function plot_energy(ts, orbit, ds)
    m1, m2, f1, f2 = ds.p
    phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    rho1 = (@. 0.5*f1*phi1_dot^2 + 0.5*m1*f1*phi1^2)
    rho2 = (@. 0.5*f2*phi2_dot^2 + 0.5*m2*f2*phi2^2)
    pot  = calc_pot(phi1, phi2, f1, f2)
    total = rho1 + rho2 + pot

    figure()
    plot(ts, total, lw=0.5)
    xlabel("t")
    ylabel("total energy")

    obs = [rho1, rho2, pot]
    obs_names = ["\$\\rho_1\$", "\$\\rho_2\$", "\$\\rho_\\mathrm{pot}\$"]

    figure()
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

    fig = figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot(rho1, rho2, pot)
    ax.set_xlabel("free part of \$\\phi_1\$")
    ax.set_ylabel("free part of \$\\phi_2\$")
    ax.set_zlabel("shared potential energy")
    ax.plot([rho1[1]], [rho2[1]], [pot[1]], "o")
    ax.plot([rho1[end]], [rho2[end]], [pot[end]], "o")
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

function plot_eos(ts, orbit, ds)
    m1, m2, f1, f2 = ds.p
    phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    rho1 = (@. 0.5*f1*phi1_dot^2 + 0.5*m1*f1*phi1^2)
    rho2 = (@. 0.5*f2*phi2_dot^2 + 0.5*m2*f2*phi2^2)
    pot  = calc_pot(phi1, phi2, f1, f2)
    total = rho1 + rho2 + pot
    pressure = rho1 + rho2 - pot
    eos = pressure ./ total
    mean_eos = mean(eos)
    bins = range(extrema(eos)..., length=30 + 1)
    hist = normalize(fit(Histogram, eos, bins), mode=:pdf)

    figure()
    subplot(2,1,1)
    plot(ts, eos, label="evolution")
    axhline(mean_eos, label="mean", color="black", ls="--")
    legend()
    xlabel("t")
    ylabel("equation of state")
    subplot(2,1,2)
    step(hist.edges[1][1:end-1], hist.weights, label="histogram")
    axvline(mean_eos, color="black", ls="--", label="mean")
    xlabel("equation of state")
    ylabel("frequency")
    legend()
end





