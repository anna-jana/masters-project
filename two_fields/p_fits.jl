include("coupled_fields.jl")

find_local_peaks(y) = [i for i = 2:length(y)-1 if y[i-1] < y[i] > y[i+1]]

############################## analysis of the energy scaling ##########################
model(x, p) = @. p[1] * x + p[2]

function find_p(M, G, initial_ratio;
        window=4, nsteps=200, end_factor=1e2, start_factor=1e3, nattempts=50,
        required_oscs=20, required_peaks=10, inc_factor=10, debug=false)
    # NOTE: an "oscillation" is actually only half an oscillation!!! :D
    @show (M, G, initial_ratio)
    H0 = calc_H_osc_analytical(M, G, initial_ratio, 1/10)
    H_end =  H0 / end_factor
    initial = make_initial(initial_ratio, H0)

    roots1 = Int[]
    roots2 = Int[]
    sizehint!(roots1, required_oscs)
    sizehint!(roots2, required_oscs)

    energy_so_far = calc_energy(M, G, initial)
    a_so_far = Float64[a0]
    ts_so_far = Float64[t0]
    phi1_so_far = Float64[initial[2]]
    phi2_so_far = Float64[initial[4]]

    ncollected = 0

    @time for attempt = 1:nattempts
        @assert H0 > H_end
        # log spaced a's
        a_range = exp.(range(log(H_to_a(H0, H0)), log(H_to_a(H_end, H0)), length=nsteps))
        # time steps
        ts = a_to_t.(a_range, H0)
        # solve for the evolution
        problem = ODEProblem(coupled_fields_from_initial(M, G, initial), (ts[1], ts[end]))
        sol = solve(problem, alg; saveat=ts, settings...)

        # search for roots
        for i = 2:length(sol)
            if sign(sol[2, i]) != sign(sol[2, i - 1])
                push!(roots1, i + ncollected - 1)
            end
            if sign(sol[4, i]) != sign(sol[4, i - 1])
                push!(roots2, i + ncollected - 1)
            end
        end
        nroots = min(length(roots1), length(roots2))
        print((length(roots1), length(roots2)), " ")

        # this should be done if nroots < required_oscs or not
        if nroots > 0 # if nroots > 0 once then it will be for all succeding iterations
            energy = calc_energy(M, G, sol)
            @views append!(energy_so_far, energy[2:end])
            @views append!(a_so_far, a_so_far[end] .* a_range[2:end])
            @views append!(ts_so_far, ts_so_far[end] .+ ts[2:end])
            @views append!(phi1_so_far, sol[2, 2:end])
            @views append!(phi2_so_far, sol[4, 2:end])
            ncollected += length(energy) - 1
        end

        if nroots >= required_oscs # we found enough oscillations
            best_p = NaN
            best_p_err = Inf
            best_fit_res = nothing
            best_skip_oscs = -1

            function prepare_fiting_data(skip_oscs)
                fit_start_index = max(roots1[skip_oscs + 1], roots2[skip_oscs + 1]) # both fields should oscillate
                # prepare fitting data
                # smoothed data for fitting
                #@views fit_x = [mean(log.(a_so_far[k-window:k+window])) for k = fit_start_index + window : nsteps - window]
                #@views fit_y = [mean(log.(energy_so_far[k-window:k+window])) for k = fit_start_index + window : nsteps - window]
                # not smoothed
                #@views fit_x = log.(a_so_far[fit_start_index:end])
                #@views fit_y = log.(energy_so_far[fit_start_index:end])
                @views x = log.(a_so_far[fit_start_index:end])
                @views y = log.(energy_so_far[fit_start_index:end])
                peaks = find_local_peaks(y)
                fit_x = x[peaks]
                fit_y = y[peaks]
                if length(fit_y) < required_peaks
                    print("*")
                    fit_x = x
                    fit_y = y
                end
                return fit_x, fit_y
            end

            param_guess = [1., 1.]
            for skip_oscs = 0:nroots - 4
                fit_x, fit_y = prepare_fiting_data(skip_oscs)
                # do the fit
                fit_res = nothing
                p = NaN
                try
                    fit_res = curve_fit(model, fit_x, fit_y, param_guess)
                    p = fit_res.param[1]
                    param_guess = fit_res.param
                catch
                end
                try
                    p_err = sqrt(estimate_covar(fit_res)[1, 1])
                    if p_err < best_p_err
                        best_p = p
                        best_p_err = p_err
                        best_fit_res = fit_res
                        best_skip_oscs = skip_oscs
                    end
                catch
                    if !isfinite(best_p)
                        best_p = p
                    end
                end
            end

            # make debug plot
            if debug
                figure()
                subplot(2,1,1)
                log_a  = log.(a_so_far)
                plot(log_a, log.(energy_so_far))
                fit_x, fit_y = prepare_fiting_data(best_skip_oscs)
                plot(fit_x, fit_y)
                plot(fit_x, model(fit_x, best_fit_res.param))
                xlabel("log(a)")
                ylabel("log(rho)")
                subplot(2,1,2)
                plot(log_a, phi1_so_far)
                plot(log_a, phi2_so_far)
                suptitle("M = $M, G = $G, initial_ratio = $initial_ratio")
                tight_layout()
            end

            println("")
            return best_p, best_p_err # all step were successfull -> stop

        elseif length(roots1) == 0 || length(roots2) == 0 # no oscillations at all -> continue
            H0 = sol[1, end]
            H_end /= inc_factor
            initial = sol.u[end]

        elseif length(roots1) >= 2 && length(roots2) >= 2 # some oscillations
            # if no oscillation were found, then we can continue from the last timestep
            # estimate the required integration time from the oscillation if they are at least two roots
            period1 = ts_so_far[roots1[end]] - ts_so_far[roots1[end-1]]
            period2 = ts_so_far[roots2[end]] - ts_so_far[roots2[end-1]]
            # extend the time integration
            period = max(period1, period2)
            required_timespan_guess = required_oscs * period
            H0 = sol[1, end]
            H_end = a_to_H(t_to_a(required_timespan_guess, H0), H0)
            initial = sol.u[end]

        else # a single oscillation in one of the fields
            H_end /= inc_factor
            initial = sol.u[end]
        end
    end

    println("")
    return NaN, NaN
end

G_list_p = 10.0 .^ [-2, 0, 3, 6]
M_list_p = 10.0 .^ [-2, -1, 1, 2]
initial_ratio_list_p = 10.0 .^ (-4:0.05:6.0)
p_filename(M, G) = "p_fit_M=$(M)_G=$(G).dat"

function compute_p_fits(;args...)
    @time for G in G_list_p
        for M in M_list_p
            p_list = find_p.(M, G, initial_ratio_list_p; args...)
            writedlm(p_filename(M, G), [initial_ratio_list_p [x[1] for x in p_list] [x[2] for x in p_list]])
        end
    end
end

function plot_p_fits(; errors=false)
    figure(figsize=(10,5))
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
        axhline(-3, label="matter", color="black", ls="--")
        axhline(-4, label="radiation", color="black", ls=":")
        xlabel("initial ratio")
        ylabel("p")
        xscale("log")
        title("G = $G")
        if k2 == 1
            legend(ncol=2)
        end
    end
    tight_layout()
    savefig("p_fit_plots.pdf")
end

function analyse_peaks()
    peak_list = Tuple{Float64, Float64, Float64}[]
    for (k2, G) in enumerate(G_list_p)
        for (k1, M) in enumerate(M_list_p)
            data = readdlm(p_filename(M, G))
            @views r, p, p_err = data[:,1], data[:,2], data[:,3]
            localpeaks = find_local_peaks(p)
            dominant_peaks = localpeaks[find_local_peaks(p[localpeaks])]
            threshold = -2.0
            high_dominant_peaks = dominant_peaks[p[dominant_peaks] .> threshold]
            for k in high_dominant_peaks
                push!(peak_list, (M, G, r[k]))
            end
        end
    end
    return peak_list
end
