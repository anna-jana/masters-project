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

model(x, p) = @. p[1] * x + p[2]

const alg = AutoTsit5(Rosenbrock23())
const settings = (reltol=1e-6, abstol=1e-6, maxiters=10^15)

G_range = (10 .^ (-2:0.2:6))
initial_ratio_range = [1, 1 + 1e-2, 10, 100]
M_range = [1e-2, 1e-1, 1, 10, 100]

######################## plot trajectories and phasespace projections ###################
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
        H0_start=1e2*M, Hmax_start=M/4, nmaxtrys=100, divisions=1000,
        inc_factor=5, check_factor=5, epsilon=1e-2)
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

H_osc_filename(M, initial_ratio) = "coupled_fields_H_osc_M=$(M)_initial_ratio=$(initial_ratio).dat"

function compute_H_osc()
    @time for initial_ratio in initial_ratio_range
        for M in M_range
            H_osc_list = find_H_osc.(M, G_range, initial_ratio)
            writedlm(H_osc_filename(M, initial_ratio), [G_range H_osc_list])
        end
    end
end

function calc_H_osc_analytical(M, G, initial_ratio, n)
    return n * max(sqrt(1 + G*initial_ratio^2), sqrt(M + G))
end

function plot_H_osc(;n=2)
    for (i, initial_ratio) in enumerate(initial_ratio_range)
        subplot(2, 2, i)
        title(@sprintf("\$\\phi_1 / \\phi_2\$ = %.2f", initial_ratio))
        for (j, M) in enumerate(M_range)
            try
                data = readdlm(H_osc_filename(M, initial_ratio))
                G_range, H_osc_list = data[:, 1], data[:, 2]
                l = plot(G_range, abs.(H_osc_list), label=@sprintf("simulation M = %.2f", M))
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
            legend()
        end
    end
    tight_layout()
end

############################## analysis of the energy scaling ##########################
function find_p(M, G, initial_ratio, param_guess, debug;
        window = 4, nsteps=200, end_factor=1e3, start_factor=1e1, nattempts=50, required_oscs=20)
    # an "oscillation" is actually only half an oscillation!!! :)
    @show (M, G, initial_ratio)
    # hubble scales
    H0 = calc_H_osc_analytical(M, G, initial_ratio, 1/10)
    H_end =  H0 / end_factor
    initial = make_initial(initial_ratio, H0)

    roots1 = Int[]
    roots2 = Int[]
    sizehint!(roots1, required_oscs)
    sizehint!(roots2, required_oscs)
    nroots = -1

    @time for attempt = 1:nattempts
        # @show (attempt, H0, H_end)
        #print(attempt, " ")
        # log spaced a's
        a_range = exp.(range(log(H_to_a(H0, H0)), log(H_to_a(H_end, H0)), length=nsteps))
        # time steps
        ts = a_to_t.(a_range, H0)
        # solve for the evolution
        problem = ODEProblem(coupled_fields_from_initial(M, G, initial), (ts[1], ts[end]))
        sol = solve(problem, alg; saveat=ts, settings...)

        # check for validity of the result
        # search for roots
        empty!(roots1)
        empty!(roots2)
        for i = 2:length(sol)
            if sign(sol[2, i]) != sign(sol[2, i - 1])
                push!(roots1, i)
            end
            if sign(sol[4, i]) != sign(sol[4, i - 1])
                push!(roots2, i)
            end
        end
        nroots = min(length(roots1), length(roots2))
        print(nroots, " ")

        if nroots < required_oscs # enough oscillations
            # if no oscillation were found, then we can continue from the last timestep
            if length(roots1) == 0 || length(roots2) == 0 # TODO: || or && ?
                initial = sol.u[end]
                H0 = sol[1, end]
            end
            # estimate the required integration time from the oscillation if they are at least two roots
            period1 = -1.0
            period2 = -1.0
            if length(roots1) >= 2
                period1 = ts[roots1[end]] - ts[roots1[end-1]]
            end
            if length(roots2) >= 2
                period2 = ts[roots2[end]] - ts[roots2[end-1]]
            end
            period = max(period1, period2)
            # extend the time integration
            if period > 0.0
                required_timespan_guess = required_oscs * period
                H_end = a_to_H(t_to_a(required_timespan_guess, H0), H0)
            else
                H_end /= 5
            end
            # retry
            continue
        end

        # energy to fit
        energy = calc_energy(M, G, sol)
        # prepare fitting data
        fit_start_index = max(roots1[1], roots2[1]) # both fields should oscillate

        # smoothed data for fitting
        #@views fit_y = [mean(log.(energy[k-window:k+window])) for k = fit_start_index + window : nsteps - window]
        #@views fit_x = [mean(log.(a_range[k-window:k+window])) for k = fit_start_index + window : nsteps - window]

        # not smoothed
        @views fit_y = log.(energy[fit_start_index:end])
        @views fit_x = log.(a_range[fit_start_index:end])

        # do the fit
        fit_res = curve_fit(model, fit_x, fit_y, param_guess)
        p = fit_res.param[1]
        p_err = sqrt(estimate_covar(fit_res)[1, 1])

        # make debug plot
        if debug
            figure()
            subplot(2,1,1)
            plot(log.(a_range), log.(energy))
            axvline(ooth_x[1], color="black")
            plot(fit_x, fit_y)
            plot(fit_x, model(fit_x, fit_res.param))
            xlabel("log(a)")
            ylabel("log(rho)")
            subplot(2,1,2)
            plot(log.(a_range), sol[2, :])
            plot(log.(a_range), sol[4, :])
            suptitle("M = $M, G = $G, initial_ratio = $initial_ratio")
            tight_layout()
        end
        println("")
        return p, p_err # all step were successfull -> stop
    end
    println("")
    return NaN, Float64(nroots)
end

G_list_p = 10.0 .^ [-2, 0, 3, 6]
M_list_p = 10.0 .^ [-2, -1, 1, 2]
initial_ratio_list_p = 10.0 .^ (-4:0.1:5.0)
p_filename(M, G) = "p_fit_M=$(M)_G=$(G).dat"

function compute_p_fits()
    @time for G in G_list_p
        for M in M_list_p
            p_list = find_p.(M, G, initial_ratio_list_p, Ref([1., 1.]), false)
            writedlm(p_filename(M, G), [initial_ratio_list_p [x[1] for x in p_list] [x[2] for x in p_list]])
        end
    end
end

function plot_p_fits(; errors=false)
    figure()
    i = 1
    for (k2, G) in enumerate(G_list_p)
        subplot(2, 2, k2)
        for (k1, M) in enumerate(M_list_p)
            data = readdlm(p_filename(M, G))
            @views r, p, p_err = data[:,1], data[:,2], data[:,3]
            if errors
                errorbar(r, p, p_err, capsize=3.0, fmt="-", label="M = $M")
            else
                plot(r, p, label="M = $M")
            end
            i += 1
        end
        axhline(-3, label=k2 != 1 ? nothing : "matter", color="black", ls="--")
        axhline(-4, label=k2 != 1 ? nothing : "radiation", color="black", ls=":")
        # axhline(0,  label=k2 != 1 ? nothing : "dark energy", color="black", ls="-")
        xlabel("initial ratio")
        ylabel("p")
        xscale("log")
        title("G = $G")
        legend(ncol=2)
    end
    tight_layout()
end

######################################### run all the things #####################################
function main()
    lya_G_dep()
    compute_H_osc()
    plot_H_osc()
    compute_p_fits()
    plot_p_fits()
end
