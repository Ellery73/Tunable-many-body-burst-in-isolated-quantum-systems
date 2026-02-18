using LinearAlgebra, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter

# Functions

function Heisenberg(N, s, Jx, Jy, Jz, hx, hy, hz)
    os_phys = OpSum()
    for j in 1:(N - 1)
        os_phys += Jx, "Sx", j, "Sx", j + 1
        os_phys += Jy, "Sy", j, "Sy", j + 1
        os_phys += Jz, "Sz", j, "Sz", j + 1
        os_phys += hx, "Sx", j
        os_phys += hy, "Sy", j
        os_phys += hz, "Sz", j
    end
    os_phys += hx, "Sx", N
    os_phys += hy, "Sy", N
    os_phys += hz, "Sz", N
    return MPO(os_phys,s)
end

function Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, dt)
    gate = ITensor[]

    for j in 1:(N - 1)
        s1 = s[j]
        s2 = s[j + 1]
        hj =
        Jx * op("Sx", s1) * op("Sx", s2) +
        Jy * op("Sy", s1) * op("Sy", s2) +
        Jz * op("Sz", s1) * op("Sz", s2) +
        hx * op("Sx", s1) * op("Id", s2) +
        hy * op("Sy", s1) * op("Id", s2) +
        hz * op("Sz", s1) * op("Id", s2)
        Gj = exp(-im * dt / 2 * hj)
        push!(gate, Gj)
    end
    hN = hx * op("Sx", s[N]) + hy * op("Sy", s[N]) + hz * op("Sz", s[N])
    GN = exp(-im * dt / 2 * hN)
    push!(gate, GN)
    append!(gate, reverse(gate))

    return gate
end

function Eq_beta(N, s, beta, O, Jx, Jy, Jz, hx, hy, hz; dbeta = 0.0001, maxdim=10024, cutoff=1e-14)
    Id = MPO(s, n -> "Id")
    rho = copy(Id)
    gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, -im * dbeta)
    steps = round(Int, abs(beta / dbeta))
    for step in 1:steps
        rho = apply(gate_beta, rho; maxdim=maxdim, cutoff=cutoff)
    end
    tr_rho = inner(Id, rho)
    eq_O = real(inner(O, rho) / tr_rho)
    return eq_O
end

function calc_beta_and_obs(N, s, psi, O, Jx, Jy, Jz, hx, hy, hz; dbeta_abs=0.0001, max_steps=10000, maxdim=1024, cutoff=1e-14)

    Id = MPO(s, n -> "Id")
    H = Heisenberg(N, s, Jx, Jy, Jz, hx, hy, hz)
    
    # Energy of state psi
    exact_E = real(inner(psi', H, psi) / inner(psi', Id, psi))
    
    # Energy at infinite temperature (beta=0)
    rho = copy(Id)
    tr_rho_inf = inner(Id, rho)
    E_inf = real(inner(H, rho) / tr_rho_inf)

    # Determine beta search direction (positive/negative) based on energy relation
    if exact_E < E_inf
        dbeta = dbeta_abs
    else
        dbeta = -dbeta_abs
    end

    beta = 0.0
    current_O = 0.0
    
    # Imaginary time evolution operator
    gate_beta = Trotter(N, s, Jx, Jy, Jz, hx, hy, hz, -im * dbeta)

    for step in 1:max_steps
        beta += dbeta
        rho = apply(gate_beta, rho; maxdim=maxdim, cutoff=cutoff)
        tr_rho = inner(Id, rho)
        current_E = real(inner(H, rho) / tr_rho)
        current_O = real(inner(O, rho) / tr_rho)

        # Branch termination condition depending on search direction
        if (dbeta > 0 && current_E <= exact_E) || (dbeta < 0 && current_E >= exact_E)
            break
        end
    end
    
    return beta, current_O
end

function main()

    Ls = [10, 20, 30, 40]
    chi = 10
    dt = 0.2
    rep = 150
    ttotal = dt * rep
    maxdim_obs = 2048
    maxdim_state = 128
    trunc = 1e-7
    Jx, Jy, Jz, hx, hy, hz = 0.0, 0.0, 1.0, 0.9045/2, 0.0, 0.8090/2
    beta = 0.1
    penalty_coeff = 72.0
    observable = "Magz"
    num_parts = 30
    
    # Added: Number of trials
    num_trials = 3

    burst_results = Dict{Int, Vector{Float64}}()
    eq_results = Dict{Int, Vector{Float64}}()   # current_O
    dyn_results = Dict{Int, Vector{Float64}}()  # -real(inner(...))

    local ts

    for L in Ls
        lambda = penalty_coeff/L^2

        # --- Reconstruct filename ---
        local cache_file
        cache_file = "$(observable)_Ising_L$(L)_dt$(dt)_t$(ttotal)_bd$(maxdim_obs)_parts$(num_parts).jld2"

        # --- Load Data ---
        if isfile(cache_file)
            # println("File found: $cache_file")
            data = load(cache_file)
            s = data["s"]
            ts = data["ts"]
            O_Us = data["O_Us"]
            # println("Data loading completed.")
        else
            println("File not found: $cache_file")
            return # quit if file does not exist
        end

        local O
        if observable == "Szc"
            c = div(L, 2) + 1 # center site
            os_Szc = OpSum()
            os_Szc += "Sz", c
            O = MPO(os_Szc,s) # Sz of site c
        elseif observable == "Magy"
            O = Heisenberg(L, s, 0.0, 0.0, 0.0, 0.0, 1/L, 0.0)
        elseif observable == "Magz"
            O = Heisenberg(L, s, 0.0, 0.0, 0.0, 0.0, 0.0, 1/L)
        end

        gate = Trotter(L, s, Jx, Jy, Jz, hx, hy, hz, dt)
        gate_dag = Trotter(L, s, Jx, Jy, Jz, hx, hy, hz, -dt)
        H_phys = Heisenberg(L, s, Jx, Jy, Jz, hx, hy, hz)
        Id = MPO(s, n -> "Id")
        E_target = Eq_beta(L, s, beta, H_phys, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)
        H_diff = H_phys - E_target * Id
        
        H_penalty = lambda * apply(H_diff, H_diff; maxdim=maxdim_obs, cutoff=trunc)

        burst_values = Float64[]
        eq_values = Float64[]
        dyn_values = Float64[]

        @showprogress "Calculating L=$L" for i in 1:(num_parts+1)
            tau = ts[i]
            O_tau = O_Us[i]

            current_max_burst = -Inf # Variable to hold maximum value
            best_eq_val = 0.0        # ★Added: Equilibrium value at maximum
            best_dyn_val = 0.0       # ★Added: Dynamic value at maximum

            # Trial loop added
            for trial in 1:num_trials
                # Change seed for each trial (Depend on L, i, trial for reproducibility)

                if trial != num_trials
                    Random.seed!(1000 + L*100 + i*10 + trial)
                    psi0 = random_mps(ComplexF64, s, linkdims = 1)
                
                else
                    if observable == "Magy"
                        psi0 = MPS(ComplexF64, s, n -> "Y-")
                    else
                        psi0 = MPS(ComplexF64, s, n -> "Dn")
                    end
                    for j in 1:(i-1)*div(rep, num_parts)
                        psi0 = apply(gate_dag, psi0; maxdim=maxdim_state, cutoff=trunc)
                        normalize!(psi0)
                    end
                end

                nsweeps = 50
                # Fix: Match array length to nsweeps
                maxdim = fill(chi, nsweeps)
                noise_schedule = [1e-3, 1e-3, 1e-4, 1e-4, 1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-8, 1e-9, 1e-9, 1e-10, 1e-10, 1e-11, 1e-11, 1e-12, 1e-12, 1e-13, 1e-13, 1e-14, 1e-14, 0.0]

                eigs, psi = dmrg([O_tau, H_penalty], psi0; nsweeps=nsweeps, maxdim=maxdim, noise=noise_schedule, outputlevel=0)

                psi_evolved = psi
                for j in 1:(i-1)*div(rep, num_parts)
                    psi_evolved = apply(gate, psi_evolved; maxdim=maxdim_state, cutoff=trunc)
                    normalize!(psi_evolved)
                end

                # Inverse temperature and equilibrium value
                beta_true, current_O = calc_beta_and_obs(L, s, psi, O, Jx, Jy, Jz, hx, hy, hz; maxdim=maxdim_obs)
                
               # Calculate result for current trial
                dyn_val = -real(inner(psi_evolved', O, psi_evolved)) # Dynamic part
                val = current_O + dyn_val # Burst value
                
                # Update maximum value
                if val > current_max_burst
                    current_max_burst = val
                    best_eq_val = current_O
                    best_dyn_val = dyn_val
                end
            end
            
            # Save maximum values
            push!(burst_values, current_max_burst)
            push!(eq_values, best_eq_val)
            push!(dyn_values, best_dyn_val)
        end

        burst_results[L] = burst_values
        eq_results[L] = eq_values
        dyn_results[L] = dyn_values
    end

    results_filename = "Ltauvsburst_UOU_$(observable)_T$(ttotal)_lambda$(penalty_coeff)_chi$(chi).jld2"
    println("Saving results to $results_filename...")
    save(results_filename, "plot_ts", ts, "burst_results", burst_results, "eq_results", eq_results, "dyn_results", dyn_results)
    println("Finished saving.")
    
end

main()
