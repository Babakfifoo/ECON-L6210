include("main_counterfactual.jl")


function gradientSig(X₀, data, prices, q̄)
    β = X₀[1:7]
    bL = min(X₀[8], X₀[9])
    bB = max(X₀[8], X₀[9])
    γ = 1 ./ (exp.(-X₀[10] .- (0:59) .* X₀[11] .- (0:59) .^ 2 .* X₀[12]) .+ 1)
    μT = [X₀[13] .* ones(T - 20); X₀[14] .* ones(7); X₀[15] .* ones(7); X₀[16] .* ones(6)]
    μD = [1; X₀[17:22]]
    μ = μT * μD'
    σ = X₀[end]
    # first FE
    f0 = allDemand(β, bL, bB, γ, μ, prices);
    multi = ((Vector(1:q̄) .- 1) * prices')
    ER0 = deepcopy(f0);
    for i in CartesianIndices((1:q̄,1:q̄,1:length(γ)))
        ER0[i[1]][i[2]][i[3]]= (f0[i[1]][i[2]][i[3]] .* multi[i[2],:])
    end


    CCP0 = dynEst(f0, ER0, γ, σ + 1e-4, β)
    loss0 = sum(CCP0[data[:, 3], data[:, 1], data[:, 4], data[:, 5]])

    CCP1 = dynEst(f0, ER0, γ, σ - 1e-4, β)
    loss1 = sum(CCP1[data[:, 3], data[:, 1], data[:, 4], data[:, 5]])

    return (loss0 - loss1) / (2 * 1e-4)
end

size(f0[1][1][1])


# FIX The sum in numpy needs implementation. I am going crzy...
for i in CartesianIndices((1:q̄,1:q̄,1:length(γ)))
    sum.(ER0[i[1]][i[2]])
    break
end





getindex.(ER, 1)


ER = deepcopy(ER0);



((ER[:, end, :, :]./ σ).*Pₜ[end, 2:end]')[end,:,:]


ER[end][end][end]


function dynEst(f, ER, γ, σ, β)
    """
        f : demand function 120x60x8x7
        ER: expected revenue 120x60x8x7
        γ : consumer type probabilities 60x1
        σ : standard deviation of demand 1x1
        β : demand parameters 7x1
    """

    EV = zeros(T, q̄, numP, length(β))
    V = zeros(q̄, T, length(β))
    CCP = zeros(T, q̄, numP, length(β))
    for t in 1:(T)
        if t == 1
            grp = ER[:, end, :, :] ./ (σ .*Pₜ[end, 2:end])'
            grp = ifelse.(grp .== 0, -Inf, grp)
            V[:, end, :] = σ .*(log.(sum(exp.(grp), dims = 2)).+ EC)
            V[1, end, :] .= 0
            CCP[end, :, :, :] = grp .- log.(sum(exp.(grp), dims = 2))
        else
            grp = (ER[:, end, :, :] ./ σ .+ EV[end - t + 1, :, :, :] ./ σ) .* Pₜ[end - t + 1, 2:end]'
            grp = ifelse.(grp .== 0, -Inf, grp)
            V[:, end - t + 1, :] = σ .* (log.(sum(exp.(grp), dims = 2)) .+ EC)
            V[1, end - t + 1, :] .= 0
        end

        lower_tri = findall(x -> x == 1.0, tril(f[:, :, 1, 1, 1]) .|> y -> y != 0)
        
        for b in (1:length(β)).+1

            if t != T
                
                g0 = copy(f[:, :, end - t, :, b])
                g = zeros(size(g0))

                for e in lower_tri
                    g[e[1],e[2],:] = g0[e[1], e[1] - e[2] + 1,:]
                    
                end

                EV[end - t, :, :, b] = sum(g .* V[:, end - t + 1, b], dims = 2) .* Pₜ[end - t + 1, 2:end]
            end

            if t != 1
                XX = (ER[:, end - t, :, :] .+ EV[end - t + 1, :, :, :]) ./ σ .* Pₜ[end - t + 1, 2:end]'
                XX = ifelse.(XX .== 0, -Inf, XX)
                CCP[end - t, :, :, b] = XX .- log.(sum(exp.(XX), dims = 2))
            end

        end

        return CCP

    end
end



prices= [1.93584019, 2.51706816, 3.06912093, 3.68721778, 4.35668806,
5.03306089, 5.8804312 , 6.74935852]

# function logLike(X₀, data)


f0 = allDemand(β, bL, bB, γ, μ)


ER0 = copy(f0)

for i in CartesianIndices((size(f0,1),size(f0,2),size(f0,3)))
    ER0[i,:,:] = f0[i,:,:] .* prices
end

ER0 = sum(ER0, dims = 1)[1,:,:,:,:] # this is where the issue is ... 

f0_sub = zeros(size(data))
for i in 1:size(data,1)
    idx = data[i,:] .+ 1
    f0_sub[i,:] = f0[idx]
end

loss0 = sum(exp.(f0_sub))
CCP0 = dynEst(f0, ER0, γ, σ, β)

    # define loss associated with transition probabilities
eachindex(idx)




i=1

f0_sub = 
idx = 
CartesianIndices((idx))

    loss0 = jnp.sum(jnp.log(
        f0 
        
        

    ))
    # solve the DP, calculate the CCP and EV
    CCP0 = dynEst_jit(f0, ER0, gamma, sig, beta)
    # define loss for CCP
    loss1 = jnp.sum(CCP0[data[:, 2], data[:, 0], data[:, 3], data[:, 4]])
    ## SECOND FE
    return loss0 + loss1


M = rand(2, 3, 4, 5, 6)

# Create a vector of length 5 to represent the indices
v = [2, 1, 3, 4, 2]

# Use CartesianIndices to get the CartesianIndex of the vector
ci = CartesianIndices((2, 3, 4, 5, 6))[v]
    

size(f0)


f0[:,:,:,:,data[1,5]]


f0[1:4]


