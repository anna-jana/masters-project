using DynamicalSystems
using PyPlot
using LinearAlgebra
using StatsBase
using OrdinaryDiffEq
using LsqFit
using Statistics

hubble(t) = 1 / (2*t)

function coupled_fields_rhs(s, p, t)
    H = hubble(t)
    phi1, phi1_dot, phi2, phi2_dot = s
    M, G = p
    return SVector(
        phi1_dot,
        - 3*H*phi1_dot - G*phi1*phi2^2 - phi1,
        phi2_dot,
        - 3*H*phi2_dot - G*phi2*phi1^2 - M*phi2,
    )
end

function sim(M, G, initial_ratio, start_time;
        tspan = 1e3, ttr = 1e3, dt = 1.0,
        solver_options = (abstol = 1e-6, reltol = 1e-6,
                          alg = AutoTsit5(Rosenbrock23())))
    initial = [1.0, 0.0, initial_ratio, 0.0]
    default_params = [M, G]
    ds = ContinuousDynamicalSystem(coupled_fields_rhs,
                                   initial, default_params, t0=start_time)
    ts = (start_time + ttr):dt:(start_time + ttr + tspan)
    orbit = trajectory(ds, tspan, Δt=dt, t0=start_time,
                       Ttr=ttr; solver_options...)
    return ts, orbit
end

function calc_energies(M, G, orbit)
    phi1, phi1_dot, phi2, phi2_dot = columns(orbit)
    rho1 = @. 0.5*(phi1_dot^2 + phi1^2)
    rho2 = @. 0.5*M*(phi2_dot^2 + phi2^2)
    pot  = @. G*phi1^2*phi2^2
    total = rho1 + rho2 + pot
    return rho1, rho2, pot, total
end

function fit_power_law_to_energy(ts, energy; plot_it=true)
    model(x, p) = @. p[1] * x + p[2]
    fit_res = curve_fit(model, log.(ts), log.(energy), [1.0, 0.0])
    p = 2*fit_res.param[1] # power law rho ~ a^p
    return p
end

function make_fit(M, G, initial_ratio, start_time; options...)
    ts, orbit = sim(M, G, initial_ratio, start_time; options...)
    rho1, rho2, pot, total = calc_energies(M, G, orbit)
    return fit_power_law_to_energy(ts, total)
end
