using SpecialFunctions, LinearAlgebra

# Variables specs:
"""
T: Max number of days before departure
t: time ∈ {,1,...,T}
q̄: maximum seat count
Δq: change in seats ∈ {0,1,..., q̄}
Pt: clustered prices
γ: probability of type Business
μ[t,:] : Arrival rates (to be estimated)
bL: Leisure demand parameter (to be estimated)
bB: Business demand parameter (to be estimated)
β : demand parameters (to be estimated)
"""

methods(+)

EC = 0.5772156649 # Euler constant

X₀ = [ # Initial values
    2.49999999, 2.49999999, 2.49999999, 2.49999999, 2.49999999, 
    2.49999999, 2.49999999, -1.05185291, -0.72189149, -13.39650409, 
    0.27373386, 0.0, 1.91183252, 2.46138227, 1.82139054, 2.35728083, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22463165
]

Pt = [1.93584019, 2.51706816, 3.06912093, 3.68721778, 4.35668806, 5.03306089, 5.8804312 , 6.74935852];

VAR = X₀

β = X₀[1:7]
bL = min(X₀[8], X₀[9])
bB = max(X₀[8], X₀[9])
γ = 1 ./ (exp.(-X₀[10] .- (0:59) .* X₀[11] .- (0:59) .^ 2 .* X₀[12]) .+ 1)
μT = [X₀[13] .* ones(T - 20); X₀[14] .* ones(7); X₀[15] .* ones(7); X₀[16] .* ones(6)]
μD = [1; X₀[17:22]]
μ = μT * μD'
σ = X₀[end]

ER = deepcopy(ER0)

# --------------------------------------------------------------------------
# creating the Transition Probability Matrix:
t = T
# generating the EV at t = T, where EV(T+1) = 0
EV = Dict()
V = Dict()
CCP = Dict()

EV[t+1] =  [zeros(8,7) for i in 1:q̄]
grp = []

for q in 1:(q̄)
    push!(grp, gpr_qt(ER[q-1][t], EV[t+1][q] , Pₜ, t, σ))
end

grp = grp .|> x-> ifelse.(x .== 0, -Inf, x);
V[t] = V_T.(grp, EC, σ);
CCP[t] = grp .|> x -> x .- log.(sum(exp.(x), dims=2));
replace!.(CCP[t],NaN => 0.0)
EV[t] = [zeros(8,7) for i in 1:q̄]



for t in (T-1):-1:2
    grp = []
    for q in 1:(q̄)
        push!(grp, gpr_qt(ER[q-1][t], EV[t+1][q] , Pₜ , t, σ))
    end
    grp = grp .|> x-> ifelse.(x .== 0, -Inf, x);
    CCP[t] = grp .|> x -> x .- log.(sum(exp.(x), dims=2));
    V[t] = V_T.(grp, EC, σ);
    
    EV[t] = []
    for q in 1:(q̄)
        g_q = f[q-1] .|> x -> x[t-1]
        V_q = collect(vcat([zeros(1,7)], V[t][1:(length(g_q)-1)]) )
    
        for q_r in 1:q
            push!(EV[t], g_q[q_r] .* V_q[q_r])
        end
    end
end

grp


size(f0[1])
Pₜ = Matrix(Pₜ)

gpr_qt(ER, EV, Pₜ, t, σ) = ((ER + EV) ./ σ) .* Pₜ[t, 2:end]
V_T(grp_q, EC, σ) = σ * (log.(sum(exp.(grp_q), dims=1)) .+ EC)
EV_calc(gq,V,t,b) = sum(gq .* vcat([0],(V[t] .|> x -> x[b])[1:(length(gq)-1)])) .* Pₜ[t,2:end]'


function dynEst(f, ER,σ,T)
    t = T
    # generating the EV at t = T, where EV(T+1) = 0
    EV = Dict()
    V = Dict()
    CCP = Dict()

    EV[t+1] =  [zeros(8,7) for i in 1:q̄]
    grp = []

    for q in 1:(q̄)
        push!(grp, gpr_qt(ER0[q][t], EV[t+1][q] , Pₜ, t))
    end

    grp = grp .|> x-> ifelse.(x .== 0, -Inf, x);
    V[t] = V_T.(grp, EC, σ);
    CCP[t] = grp .|> x -> x .- log.(sum(exp.(x), dims=1));

    EV[t] = [zeros(8,7) for i in 1:q̄]

    for t in (T-1):-1:2
        grp = []
        for q in 1:(q̄)
            push!(grp, gpr_qt(ER[q][t], EV[t+1][q] , Pₜ, t, σ))
        end
        grp = grp .|> x-> ifelse.(x .== 0, -Inf, x);
        CCP[t] = grp .|> x -> x .- log.(sum(exp.(x), dims=1));
        V[t] = V_T.(grp, EC, σ);
        
        EV[t] = []
        for q in 1:(q̄)
            g_q = f[q] .|> x -> x[t-1]
            V_q = collect(vcat([zeros(1,7)], V[t][1:(length(g_q)-1)]) )
        
            for q_r in 1:q
                push!(EV[t], g_q[q_r] .* V_q[q_r])
            end
        end
    end
    return CCP
end









# define the CCP endand EV for the dynamic firm problem
function optDynNoError(f, ER, γ, β)
    # [ ] See how can you make this better

    setprecision(16) # handling floating point errors

    # create storage for EV, V, CCP
    EV = zeros(T, q̄, numP, length(β))
    V = zeros(q̄, T, length(β))
    CCP = zeros((T, q̄, numP, length(β)))

    # The t=1 is treated differently, so we will do that first
    grp = ER[:, :, :, end-1] .* Pt[1:(end-1), 1]'  # Assuming Pt is a matrix
    V[:, :, end-1] = fill(maximum(grp, dims=1), size(V, 1))

    tmp = zeros(size(grp))

    tmp .= setindex!(
        tmp, 
        ones(size(grp))[argmax(grp, dims=1)], 
        argmax(grp, dims=1)
        )

    CCP = setindex!(CCP, tmp, :, :, lastindex)

    r, c = trilindices(size(f[:,:,1,1,1]))

    for b in 1:(length(beta)+1)
        
        g = copy(f[:, :, end - 2, :, b])
        g = setindex!(g, g[r, r - c, :], r, c, :)
        EV = setindex!(
            EV, 
            sum(g .* V[:, end - 1, b][1, :, 1:end], dims=1) .* Pₜ[end - 1, 1:end],
            :, :, b)
    end

    for t in 2:(T)
        grp = ER[:, :, :, end-t] .* Pt[1:end-t, 1]'  # Assuming Pt is a matrix
        V[:, :, end-t] = fill(maximum(grp, dims=1), size(V, 1))
    
        tmp = zeros(size(grp))
    
        tmp .= setindex!(
            tmp, 
            ones(size(grp))[argmax(grp, dims=1)], 
            argmax(grp, dims=1)
            )
    
        CCP = setindex!(CCP, tmp, :, :, lastindex)
    
        r, c = tril(size(f[:,:,1,1,1]))
    
        for b in 1:(length(beta)+1)
            
            g = copy(f[:, :, end - t - 1, :, b])
            g = setindex!(g, g[r, r - c, :], r, c, :)
            EV = setindex!(
                EV, 
                sum(g .* V[:, end - t, b][1, :, 1:end], dims=1) .* Pₜ[end - t, 1:end],
                :, :, b)
        end
        
        if t!=T
            XX = (ER[:, end-t, :, :] + EV[end-t+1, :, :, :]) .* Pt[end-t, 1:end][1, :, 1:end]
            tmp = zeros(size(XX))
            tmp .= setindex!(tmp, ones(size(XX))[argmax(XX, dims=1)], argmax(XX, dims=1))
            CCP = setindex!(CCP, tmp, :, :, lastindex-t)
        end
    end

    return CCP
end

function optStatic(f, ER, β)
    T = size(f, 1)
    q̄ = size(f, 2)
    numP = size(f, 3)
    EV = zeros(T, q̄, numP, length(β))
    V = zeros(q̄, T, length(beta))
    CCP = zeros(T, q̄, numP, length(β))
    for t in 1:T
        grp = ER[:, end-t, :, :] .* Pₜ[end-t, 1:end][1, :, 1:end]
        V[:, end-t, :] = maximum(grp, dims=1)
        tmp = zeros(size(grp))
        tmp .= setindex!(tmp, ones(size(grp))[argmax(grp, dims=1)], argmax(grp, dims=1))
        CCP[end-t, :, :, :] = tmp
    end
    return CCP
end

function optStaticInf(rate, prices, Pt)
    V = rate .* prices[1, :, 1] .* Pt[:, 1:end][1:end, :, 1:end]
    return argmax(V, dims=1)
end

# Calculate the opt uniform price
function optUniform(f, ER, γ, β)
    EV = zeros(q̄, length(γ), length(prices), length(β))
    V = zeros(q̄, length(γ), length(prices), length(β))
    for t in 1:T
        # work backwards in time. In the last period, we just get last period revenues
        V[:, end-t, :, :] = ER[:, end-t, :, :] + (t == 1).*EV[:, end-t+1, :, :]
        r, c = tril(size(f[:, :, 1, 1, 1]))
        for b in 1:length(beta)
            if t != T #update expected value function, this is for not the last period
                g = copy(f[:, :, end-t - 1, :, b])
                g = setindex!(g, g[r, r - c, :], r, c, :)
                EV[:, end-t, :, b] = sum(g .* V[:, end-t, :, b][1, :, 1:end], dims=1)
            end
        end
    end
    return argmax(V[:, 1, :, :], dims=1)
end

function calcAPDvalue(f, ER, γ, q̄, b, pvec)
    pstar =  pvec * [39,7,7,4,3]
    EV = zeros(q̄, length(γ))
    V = zeros(q̄, length(γ))

    for t in 1:T
        V[:, end-t] = ER[:, end-t, pstar[end-t], b] + (t==1) .* EV[:, end-t]
        r, c = tril(size(f[:,:,0,0,0]))
        if t != T
            g = copy(f[:, :, end-t - 1, :, b])
            g[r, c, :] = g[r, r - c, :]
            EV[:, end-t] = sum(g[:, :, pstar[end-t]] * V[:, end-t][nothing, :], dim = 1)
        end

    end
    return [pstar, V[:, 0]]

end




# this one works well
function checker(Pₜ, pset)
    X = []
    truths = []
    for (i,rng) in enumerate([1:39,57:60,53:57,46:53,39:46])
        Vm = maximum(Pₜ[rng,2:end],dims=1)
        push!(truths, Vm)
    end

    for p in pset
        check = 0
        for (i,v) in enumerate(truths)
            check += v[p[i]]
        end
        if check == 5
            push!(X,p)
        end
    end
    
    return X
end




function optAPD(f, ER, γ, q̄, Pₜ)
    pset = collect(with_replacement_combinations(1:(size(Pₜ)[2]-1),5))
    pset = checker(Pₜ, pset)
    Pstar = zeros(q̄, length(γ), length(beta))

    for b in 1:length(beta)
        results = [calcAPDvalue(f, ER, γ, q̄, b, p) for p in pset]
        V = [results[i][2] for i in 1:length(results)]
        P = [results[i][1] for i in 1:length(results)]
        Pstar[:, :, b] = P[argmax(V, dims = 1), :]
    end

    return Pstar
end


function allPT(Pt)
    Pt[:, 2:end] = 1
    return Pt
end


function poissonProcess(R, aB, bL, bB, prices, p, remainCap)
    low = count(x -> x == 0, R)
    high = count(x -> x == 1, R)
    if remainCap == 0
        sales = 0
        sales_B = 0
        sales_L = 0
        CS_L = 0 
        CS_B = 0 
        CS_ALL = 0 
    elseif remainCap > 0
        sj = exp.(aB .+ ((1 .- R) .* bL .+ R .* bB) .* prices[p]) ./ \
            (1 .+ exp.(aB .+ ((1 .- R) .* bL .+ R .* bB) .* prices[p]))
        QD = (rand(length(R)) .< sj) .* 1
        cs = (-1 / ((1 .- R) .* bL .+ R .* bB)) .* \
            log.(1 .+ exp.(aB .+ ((1 .- R) .* bL .+ R .* bB) .* prices[p]))
        if sum(QD) <= remainCap
            CS_L = sum(cs[R .== 0])
            CS_B = sum(cs[R .== 1])
            CS_ALL = sum(cs)
            sales = sum(QD)
            sales_B = sum(QD[R .== 1])
            sales_L = sum(QD[R .== 0])

        elseif sum(QD) > remainCap
            dif = Int(sum(QD) - remainCap)
            csHitCap = copy(cs)
            csHitCap[findall(QD .== true)[1:dif]] .= 0
            QDHitCap = copy(QD)
            QDHitCap[findall(QD .== true)[1:dif]] .= 0
            CS_L = sum(csHitCap[R .== 0])
            CS_B = sum(csHitCap[R .== 1])
            CS_ALL = sum(csHitCap)
            sales = remainCap
            sales_B = sum(QDHitCap[R .== 1])
            sales_L = sum(QDHitCap[R .== 0])

        end

    end
    return sales, CS_L, CS_B, CS_ALL, sales_B, sales_L

end


