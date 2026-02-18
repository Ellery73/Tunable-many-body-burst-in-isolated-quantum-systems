using LinearAlgebra, ITensors, ITensorMPS, Printf, Random, JLD2, ProgressMeter

# ==========================================
# 0. Configuration and Mode Selection
# ==========================================
# ★Change this to switch modes (:z or :y)
const CALC_MODE = :z

println("Calculation Mode: $CALC_MODE")


# ==========================================
# 1. Constant Definitions
# ==========================================
const d = 2
const sx = [0.0 1.0; 1.0 0.0] ./ 2
const sy = [0.0 -im; im 0.0] ./ 2
const sz = [1.0 0.0; 0.0 -1.0] ./ 2
const id = Matrix{ComplexF64}(I, d, d)

# ==========================================
# 2. Function Definitions
# ==========================================

# --- Initialization: All Z down |↓↓...⟩ ---
function initialize_all_z_down(d, χ)
    Γ = zeros(ComplexF64, d, χ, χ)
    Λ = zeros(Float64, χ)
    Γ[2, 1, 1] = 1.0
    Λ[1] = 1.0
    
    noise_level = 1e-16
    Γ .+= (rand(ComplexF64, size(Γ)...) .- 0.5 .- 0.5im) * noise_level
    Λ .+= rand(Float64, size(Λ)...) * noise_level
    Λ ./= norm(Λ)
    return Γ, Λ
end

# --- Initialization: All Y down |y- y- ...⟩ ---
function initialize_all_y_down(d, χ)
    Γ = zeros(ComplexF64, d, χ, χ)
    Λ = zeros(Float64, χ)
    # |y-> = (1/sqrt(2)) * (|0> - i|1>)
    Γ[1, 1, 1] = 1.0 / sqrt(2)
    Γ[2, 1, 1] = -im / sqrt(2)
    Λ[1] = 1.0
    
    noise_level = 1e-16
    Γ .+= (rand(ComplexF64, size(Γ)...) .- 0.5 .- 0.5im) * noise_level
    Λ .+= rand(Float64, size(Λ)...) * noise_level
    Λ ./= norm(Λ)
    return Γ, Λ
end

# --- iTEBD one step update ---
function apply_gate_and_truncate(Γ_L, Λ_center, Γ_R, Λ_right, Λ_left, gate, χ_max)
    d_phys, χ_L, χ_C = size(Γ_L)
    _, _, χ_R = size(Γ_R)
    
    Γ_L_w = Γ_L .* reshape(Λ_left, (1, χ_L, 1)) 
    Γ_R_w = Γ_R .* reshape(Λ_right, (1, 1, χ_R))
    
    M_L = reshape(permutedims(Γ_L_w, (1, 2, 3)), (d_phys * χ_L, χ_C))
    M_L = M_L .* reshape(Λ_center, (1, χ_C))
    M_R = reshape(permutedims(Γ_R_w, (2, 1, 3)), (χ_C, d_phys * χ_R))
    
    theta = M_L * M_R
    theta = reshape(theta, (d_phys, χ_L, d_phys, χ_R))
    
    theta_p = permutedims(theta, (1, 3, 2, 4))
    theta_mat = reshape(theta_p, (d_phys*d_phys, χ_L*χ_R))
    gate_mat = reshape(gate, (d_phys*d_phys, d_phys*d_phys))
    
    theta_new_mat = gate_mat * theta_mat
    
    theta_new = reshape(theta_new_mat, (d_phys, d_phys, χ_L, χ_R))
    theta_new = permutedims(theta_new, (1, 3, 2, 4))
    M_svd = reshape(theta_new, (d_phys * χ_L, d_phys * χ_R))
    
    F = svd(M_svd)
    
    χ_new = min(length(F.S), χ_max)
    U_trunc = F.U[:, 1:χ_new]
    S_trunc = F.S[1:χ_new]
    V_trunc = F.Vt[1:χ_new, :]
    
    S_trunc = S_trunc / norm(S_trunc)
    
    eps = 1e-16
    A_new = reshape(U_trunc, (d_phys, χ_L, χ_new))
    A_new = A_new ./ reshape(Λ_left .+ eps, (1, χ_L, 1))
    
    B_new = reshape(V_trunc, (χ_new, d_phys, χ_R))
    B_new = permutedims(B_new, (2, 1, 3))
    B_new = B_new ./ reshape(Λ_right .+ eps, (1, 1, χ_R))
    
    return A_new, S_trunc, B_new
end

# --- Physical quantity measurement: Magnetization <Z> ---
function measure_z(Γ, Λ_L, Λ_R)
    d_size, χL, χR = size(Γ)
    Ψ = Γ .* reshape(Λ_L, (1, χL, 1)) .* reshape(Λ_R, (1, 1, χR))
    norm_sq = sum(abs2, Ψ)
    
    z_exp = 0.0
    Ψ_mat = reshape(Ψ, (d_size, χL*χR))
    for i in 1:(χL*χR)
        z_exp += abs2(Ψ_mat[1, i]) * (1.0)
        z_exp += abs2(Ψ_mat[2, i]) * (-1.0)
    end
    return z_exp / norm_sq / 2.0
end

# --- Physical quantity measurement: Magnetization <Y> ---
function measure_y(Γ, Λ_L, Λ_R)
    d_size, χL, χR = size(Γ)
    Ψ = Γ .* reshape(Λ_L, (1, χL, 1)) .* reshape(Λ_R, (1, 1, χR))
    norm_sq = sum(abs2, Ψ)
    
    y_exp = 0.0
    Ψ_mat = reshape(Ψ, (d_size, χL*χR))
    for i in 1:(χL*χR)
        up = Ψ_mat[1, i]
        down = Ψ_mat[2, i]
        # <σy> = 2 * Im(up* * down) -> spin operator Sy = σy/2 => Im(...)
        y_exp += imag(conj(up) * down) # * 2.0 / 2.0
    end
    return y_exp / norm_sq # Sy expecting values in -0.5 to 0.5
end

# --- 1-site physical quantity measurement (General) ---
function measure_1site(Γ, Λ_L, Λ_R, Op)
    d_size, χL, χR = size(Γ)
    Ψ = Γ .* reshape(Λ_L, (1, χL, 1)) .* reshape(Λ_R, (1, 1, χR))
    norm_sq = sum(abs2, Ψ)
    Ψ_mat = reshape(Ψ, (d_size, χL*χR))
    
    val = 0.0 + 0.0im
    for i in 1:(χL*χR)
        vec = Ψ_mat[:, i]
        val += dot(vec, Op * vec)
    end
    return real(val / norm_sq)
end

# --- 2-site physical quantity measurement ---
function measure_2site(ΓL, ΛC, ΓR, ΛR, ΛL, Op)
    d_size, χL, χC = size(ΓL)
    _, _, χR = size(ΓR)

    Γ_L_w = ΓL .* reshape(ΛL, (1, χL, 1)) 
    Γ_R_w = ΓR .* reshape(ΛR, (1, 1, χR))
    
    M_L = reshape(permutedims(Γ_L_w, (1, 2, 3)), (d_size * χL, χC))
    M_L = M_L .* reshape(ΛC, (1, χC))
    M_R = reshape(permutedims(Γ_R_w, (2, 1, 3)), (χC, d_size * χR))
    
    theta = M_L * M_R
    theta = reshape(theta, (d_size, χL, d_size, χR))
    theta_p = permutedims(theta, (3, 1, 2, 4))
    theta_vec = reshape(theta_p, (d_size*d_size, χL*χR))
    
    op_theta = Op * theta_vec
    
    val = dot(theta_vec, op_theta)
    norm_val = dot(theta_vec, theta_vec)
    
    return real(val / norm_val)
end

function measure_energy(ΓA, ΛA, ΓB, ΛB, H_bond)
    E_AB = measure_2site(ΓA, ΛA, ΓB, ΛB, ΛB, H_bond)
    E_BA = measure_2site(ΓB, ΛB, ΓA, ΛA, ΛA, H_bond)
    return 0.5 * (E_AB + E_BA)
end

# --- MPS Truncation ---
function truncate_mps(ΓA, ΛA, ΓB, ΛB, χ_target)
    d_size, χ_curr, _ = size(ΓA)
    if χ_target >= χ_curr
        return ΓA, ΛA, ΓB, ΛB, 0.0
    end
    range_new = 1:χ_target
 
    ΛA_new = ΛA[range_new]
    trunc_err_A = sum(abs2, ΛA[χ_target+1:end])
    ΛA_new ./= norm(ΛA_new)

    ΛB_new = ΛB[range_new]
    trunc_err_B = sum(abs2, ΛB[χ_target+1:end])
    ΛB_new ./= norm(ΛB_new)

    ΓA_new = ΓA[:, range_new, range_new]
    ΓB_new = ΓB[:, range_new, range_new]

    total_err = (trunc_err_A + trunc_err_B) / 2.0
    return ΓA_new, ΛA_new, ΓB_new, ΛB_new, total_err
end

# --- For thermal equilibrium state search ---
function construct_thermal_gate_consistent(H_bond, d_size, dbeta)
    U_phys = exp(-dbeta * H_bond)
    G = zeros(ComplexF64, d_size*d_size, d_size*d_size, d_size*d_size, d_size*d_size)
    for p1=1:d_size, p2=1:d_size, l1=1:d_size, l2=1:d_size
        idx_H_row = (p1-1)*d_size + p2
        idx_H_col = (l1-1)*d_size + l2
        u_val = U_phys[idx_H_row, idx_H_col]
        for a1=1:d_size, a2=1:d_size
            k1 = (p1-1)*d_size + a1
            k2 = (p2-1)*d_size + a2
            m1 = (l1-1)*d_size + a1
            m2 = (l2-1)*d_size + a2
            G[k1, k2, m1, m2] = u_val
        end
    end
    return reshape(G, ((d_size*d_size)*(d_size*d_size), (d_size*d_size)*(d_size*d_size)))
end

function expand_hamiltonian_for_thermal(H_bond, d_size)
    H_exp = zeros(ComplexF64, d_size*d_size, d_size*d_size, d_size*d_size, d_size*d_size)
    for p1=1:d_size, p2=1:d_size, l1=1:d_size, l2=1:d_size
        idx_H_row = (p1-1)*d_size + p2
        idx_H_col = (l1-1)*d_size + l2
        val = H_bond[idx_H_row, idx_H_col]
        for a1=1:d_size, a2=1:d_size
            k1 = (p1-1)*d_size + a1
            k2 = (p2-1)*d_size + a2
            m1 = (l1-1)*d_size + a1
            m2 = (l2-1)*d_size + a2
            H_exp[k1, k2, m1, m2] = val
        end
    end
    H_exp_p = permutedims(H_exp, (2, 1, 4, 3))
    return reshape(H_exp_p, ((d_size*d_size)*(d_size*d_size), (d_size*d_size)*(d_size*d_size)))
end

function find_beta_and_magnetizations(ΓA, ΛA, ΓB, ΛB, H_bond, d_phys, χ_max; beta_max=10.0, dbeta_abs=0.0001)
    E_target = measure_energy(ΓA, ΛA, ΓB, ΛB, H_bond)
    
    D = d_phys * d_phys
    ΓAt = zeros(ComplexF64, D, 1, 1)
    ΓBt = zeros(ComplexF64, D, 1, 1)
    ΛAt = ones(Float64, 1)
    ΛBt = ones(Float64, 1)
    for i in 1:d_phys
        k = (i-1)*d_phys + i
        ΓAt[k, 1, 1] = 1.0
        ΓBt[k, 1, 1] = 1.0
    end
    ΓAt ./= norm(ΓAt)
    ΓBt ./= norm(ΓBt)
    
    H_exp = expand_hamiltonian_for_thermal(H_bond, d_phys)
    E_current = measure_energy(ΓAt, ΛAt, ΓBt, ΛBt, H_exp)
    
    step_sign = (E_target < E_current) ? 1.0 : -1.0
    current_beta = 0.0
    dbeta = step_sign * dbeta_abs
    
    Gate_full = construct_thermal_gate_consistent(H_bond, d_phys, dbeta)
    Gate_half = construct_thermal_gate_consistent(H_bond, d_phys, dbeta / 2.0)
    
    max_steps = Int(beta_max / dbeta_abs)
    converged = false
    
    for s in 1:max_steps
        ΓAt, ΛAt, ΓBt = apply_gate_and_truncate(ΓAt, ΛAt, ΓBt, ΛBt, ΛBt, Gate_half, χ_max)
        ΓBt, ΛBt, ΓAt = apply_gate_and_truncate(ΓBt, ΛBt, ΓAt, ΛAt, ΛAt, Gate_full, χ_max)
        ΓAt, ΛAt, ΓBt = apply_gate_and_truncate(ΓAt, ΛAt, ΓBt, ΛBt, ΛBt, Gate_half, χ_max)
        
        current_beta += dbeta * 2.0
        E_new = measure_energy(ΓAt, ΛAt, ΓBt, ΛBt, H_exp)
        
        if step_sign > 0
            if E_new <= E_target; converged = true; end
        else
            if E_new >= E_target; converged = true; end
        end
        if converged; break; end
    end
    
    function measure_thermal_obs(Obs)
        O_exp = zeros(ComplexF64, d_phys*d_phys, d_phys*d_phys)
        for p1=1:d_phys, p2=1:d_phys
            val = Obs[p1, p2]
            for a=1:d_phys
                k1 = (p1-1)*d_phys + a
                k2 = (p2-1)*d_phys + a
                O_exp[k1, k2] = val
            end
        end
        val_A = measure_1site(ΓAt, ΛBt, ΛAt, O_exp)
        val_B = measure_1site(ΓBt, ΛAt, ΛBt, O_exp)
        return 0.5 * (val_A + val_B)
    end

    mz = measure_thermal_obs(sz)
    my = measure_thermal_obs(sy)
    return current_beta, mz, my
end

# ==========================================
# 3. Parameter Settings & Mode Dependent Settings
# ==========================================
Jx, Jy, Jz, hx, hy, hz = 0.0, 0.0, 1.0, 0.9045/2, 0.0, 0.8090/2
maxdim_state = 128
chi = 10
dt = 0.2

# Hamiltonian
H_bond = Jx * kron(sx, sx) + Jy * kron(sy, sy) + Jz * kron(sz, sz) + 
         0.5 * hx * (kron(sx, id) + kron(id, sx)) + 
         0.5 * hy * (kron(sy, id) + kron(id, sy)) + 
         0.5 * hz * (kron(sz, id) + kron(id, sz))

Gate = reshape(exp(-im * H_bond * dt), (d, d, d, d))
Gate_dag = reshape(exp(im * H_bond * dt), (d, d, d, d))

# Function and label settings according to mode
if CALC_MODE == :z
    init_func = initialize_all_z_down
    measure_func = measure_z
    extract_eq_val = (mz, my) -> mz
    file_suffix = "z"
    observable = "Magz"
elseif CALC_MODE == :y
    init_func = initialize_all_y_down
    measure_func = measure_y
    extract_eq_val = (mz, my) -> my
    file_suffix = "y"
    observable = "Magy"
else
    error("Invalid CALC_MODE. Choose :z or :y.")
end


# ==========================================
# 4. Main Process (Burst Simulation)
# ==========================================
println("Starting Burst Simulation (Mode: $CALC_MODE)...")

taus = 0.0:1.0:30.0
exp_vals = Float64[]
eq_vals = Float64[]
errors = Float64[]

@showprogress for tau in taus
    # Use initialization function according to mode
    ΓA, ΛA = init_func(d, maxdim_state)
    ΓB, ΛB = init_func(d, maxdim_state)

    # --- 1. Backward Time Evolution (Heating) ---
    t = 0.0
    while t < tau
        ΓA, ΛA, ΓB = apply_gate_and_truncate(ΓA, ΛA, ΓB, ΛB, ΛB, Gate_dag, maxdim_state)
        ΓB, ΛB, ΓA = apply_gate_and_truncate(ΓB, ΛB, ΓA, ΛA, ΛA, Gate_dag, maxdim_state)
        t += dt
    end

    # --- 2. Truncation ---
    ΓA, ΛA, ΓB, ΛB, err = truncate_mps(ΓA, ΛA, ΓB, ΛB, chi)
    push!(errors, err)
    
    # --- 3. Find corresponding Thermal State ---
    beta_found, mz_final, my_final = find_beta_and_magnetizations(ΓA, ΛA, ΓB, ΛB, H_bond, d, maxdim_state; beta_max=10.0)
    
    # Get equilibrium value (z or y)
    push!(eq_vals, extract_eq_val(mz_final, my_final))

    # --- 4. Forward Time Evolution (Burst Recovery) ---
    t = 0.0
    while t < tau
        ΓA, ΛA, ΓB = apply_gate_and_truncate(ΓA, ΛA, ΓB, ΛB, ΛB, Gate, maxdim_state)
        ΓB, ΛB, ΓA = apply_gate_and_truncate(ΓB, ΛB, ΓA, ΛA, ΛA, Gate, maxdim_state)
        t += dt
    end

    # Measure final magnetization (z or y)
    push!(exp_vals, measure_func(ΓA, ΛB, ΛA))
end

burst_vals = eq_vals .- exp_vals

# ==========================================
# 5. Plotting and Saving
# ==========================================
println(burst_vals)
println(errors)

results_filename = "inftauvsburst_$(observable)_T$(maximum(taus))__chi$(chi).jld2"
println("Saving results to $results_filename...")
save(results_filename, "taus", taus, "burst_vals", burst_vals, "exp_vals", exp_vals, "eq_vals", eq_vals, "errors", errors)
println("Simulation completed. Result saved to $results_filename.")
