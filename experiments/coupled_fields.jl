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

############################### background cosmology #########################
const q = 2
const t0 = 0.0
const a0 = 1.0

H_to_t(H, H0) = 1/q*(1/H - 1/H0) + t0
t_to_a(t, H0) = a0 * (q*H0*(t - t0) + 1)^(1/q)
a_to_t(a, H0) = 1/(q*H0)*((a/a0)^q - 1) + t0
H_to_a(H, H0) = a0 * (H0 / H)^(1/q)
a_to_H(a, H0) = H0 * (a / a0)^(- q)

################################ the model ##################################
function coupled_fields_rhs(s, params, t)
    H, phi1, phi1_dot, phi2, phi2_dot = s
    M, G = params
    return SVector(
        -q*H^2,
        phi1_dot,
        - 3*H*phi1_dot - G*phi1*phi2^2 - phi1,
        phi2_dot,
        - 3*H*phi2_dot - G*phi2*phi1^2 - M*phi2,
    )
end

coupled_fields_from_initial(M, G, initial) = ContinuousDynamicalSystem(coupled_fields_rhs, initial, [M, G], t0=t0)
make_initial(initial_ratio, H0) = [H0, 1.0, 0.0, initial_ratio, 0.0]
coupled_fields(M, G, initial_ratio, H0) = coupled_fields_from_initial(M, G, make_initial(initial_ratio, H0))

calc_pot(M, G, phi1, phi2) = G*phi1^2*phi2^2 + 0.5*phi1^2 + 0.5*M*phi2^2

function calc_energy(M, G, sol)
    @views H, phi1, phi1_dot, phi2, phi2_dot = sol[1,:], sol[2,:], sol[3,:], sol[4,:], sol[5,:]
    return @. 0.5*phi1_dot^2 + 0.5*M*phi2_dot^2 + calc_pot.(M, G, phi1, phi2)
end

calc_eff_masses(sys, sol) = @.(sol[4,:]^2 + 1), @.(sol[2,:]^2 + sys.p[1])

const alg = TRBDF2()
const settings = (reltol=1e-6, abstol=1e-6, maxiters=10^15)

######################## plot trajectories and phasespace projections ###################
function plot_evolution()
    plot_evolution(1.0, 1e4, 1.01, 1e3, H_end=1e-2)
    savefig("example_field_evolution_quatic_coupling.pdf")
end

function plot_evolution(M, G, initial_ratio, H0; nsteps=1000, H_end=M/3)
    tmax = H_to_t(H_end, H0)
    t0 = H_to_t(H0 / 1.001, H0)
    ts = [0.0; exp.(range(log(t0), log(tmax), length=nsteps - 1))]
    problem = ODEProblem(coupled_fields(M, G, initial_ratio, H0) , (0, tmax))
    sol = solve(problem, alg; settings..., saveat=ts)
    @views H, phi1, phi1_dot, phi2, phi2_dot = sol[1,:], sol[2,:], sol[3,:], sol[4,:], sol[5,:]
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
    suptitle("M = $M, G = $G, \$\\phi_1 / \\phi_2\$ = $initial_ratio, \$H_0\$ = $H0")
    tight_layout()
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
    title("M = $M, \$\\phi_1 / \\phi_2\$ = $initial_ratio, \$H_0\$ = $H0")
    savefig("lyapunov_plot.pdf")
end


