
function log_demandQ_tγ(β, bL, bB, γt, μt, q, prices)
    """
        γt: Consumer type prob at time t 1x1
        μt: mean demand at time t 7x1
        q : seat count 1x1
        β : demand parameters 7x1
        bL: Leisure demand parameters 1x1
        bB: Business demand parameters 1x1KT
    """

    # Probability of Leisure booking on time t
    purchInL = (1 .-γt) ./(1 .+ exp.(-β' .- bL .* prices))
    # Probability of Business booking on time t
    purchInB = (γt) ./(1 .+ exp.(-β' .- bB .* prices))

    purchIn = purchInL .+ purchInB
    # demand calculation for time t, seat count q
    res = q.*(log.(purchIn) .+ log.(μt)') .- (purchIn'.*μt)'  .- loggamma(q+1)
    # res = res .|> x -> ifelse((x > 0) & (x < 1e-100), 1e-100, x)
    return res
end

function demandQ_tγ(β, bL, bB, γt, μt, q, prices)
    """
        β : demand parameters 7x1
        bL: leisure demand parameters 1x1
        bB: business demand parameters 1x1
        γ : consumer type probabilities 60x1
        μ : mean demand 60×7
        q : seat count 1x1
    """

    res = exp.(log_demandQ_tγ(β, bL, bB, γt, μt, q, prices))
    # res = res .|> x -> ifelse((x > 0) & (x < 1e-100), 1e-100, x)
    return res
end

# from jax.scipy.special import gammaln, logsumexp

# logsumexp :    
# gammaln   :      logabsgamma(x)



function allDemand(β, bL, bB, γ, μ, prices)
    """
    
        β : demand parameters 7x1
        bL: leisure demand parameters 1x1
        bB: business demand parameters 1x1
        γ : consumer type probabilities 60x1
        μ : mean demand 60×7
        prices: 8x1
    """

    vlookup = []
    # pay attention to the 0
    for q in 0:(q̄-1)
        push!(vlookup, ([demandQ_tγ(β, bL, bB, γt, μ[t,:], q, prices) for (t,γt) in enumerate(γ)]))
    end

    for i in CartesianIndices((1:q̄,1:length(γ)))
        vlookup[i[1]][i[2]] = vlookup[i[1]][i[2]] .|> x -> ifelse((x > 0) & (x < 1e-100), 1e-100, x)
    end
    # vlookup: 120   x 60 x 8    x 7
    #          seats x T  x A(t) x β
    first_z = repeat([repeat([zeros(8,7)],length(γ))],q̄);
    f = []
    
    push!(f,first_z)
    for q in 2:q̄

        v_t = vcat(
            copy(vlookup[1:q]),repeat([repeat([zeros(8,7)], length(γ))],(q̄ - q))
            );
        push!(f,v_t)
    end

    for i in CartesianIndices((1:q̄,1:q̄))
        f[i[1]][i[2]][end] = max.(1 .- sum(f[i[1]][i[2]][1:(end-1),:]), 1e-100)
        break
    end
    return f
end