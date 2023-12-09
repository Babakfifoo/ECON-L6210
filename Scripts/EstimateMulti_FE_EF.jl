using DataFrames, JuMP
using Statistics, Clustering, HypothesisTests, StatsBase
using Query, FreqTables
using ShiftedArrays
using Dates, Missings, Random
using Pipe: @pipe
import Parquet2, CSV


# Create model for optimization


# Round df preparation:

df = Parquet2.Dataset("data/efdata_clean.parquet") |> DataFrame

function determine_OD(O,D)
    res = string(min(O,D), "_", max(O,D))
    return res
end

df[!, "route"] = determine_OD.(df[!,"origin"], df[!,"dest"])

df[!, "ttdate" ] = -df.tdate .+ 60

df = sort(df, [:origin, :dest, :ddate, :flightNum, :tdate], rev = true)

cols = ["origin", "dest", "ddate", "flightNum"]

df.difS = combine(
    groupby(df,cols),
    :seats => x -> ShiftedArrays.lead(x) - x; 
    renamecols=false
    ).seats

df.difP = combine(
    groupby(df,cols),
    :fare => x -> ShiftedArrays.lead(x) - x; 
    renamecols=false
    ).fare

df[!, :dd_dow] = dayofweek.(df.ddate).-1


# if market is BOS_MCI, then early periods also had Frontier as a nonstop carrier.
# Remove these entries.

MARKET = "BOS_SEA"
df.route|> unique
df_route = df[df.route .== MARKET, :]
df_route = df_route[.!ismissing.(df_route.difS), :]
df_route.difS = ifelse.(df_route.difS .> 0, 0, df_route.difS)
df_route.difS = abs.(df_route.difS)

df_route = df_route[df_route.seats .> 0,:]
df_route.tdate = maximum(df_route.tdate) .- df_route.tdate

# condition for BOS_MCI
if MARKET == "BOS_MCI"
    println("date filtered for BOS_MCI")
    df_route = df_route[df_route.ddate .> Date("2012-05-17"), :]
end

# FROM PAPER:
    # Next, winsorize the data to remove entries in which a large number of seats disappear
    # This could happen when:
    #  - seat maps get smaller
    #  - seat map errors
    #  - measurement error in processing data
    #  - Delta market has more errors which influences log-like, constrain data more.


mark = ifelse(MARKET == "BOS_MCI", 0.985, 0.995)
df_route = df_route[df_route.difS .< quantile(df_route.difS, mark), :]


numFlights = @pipe df_route[!, ["flightNum", "ddate"]] |> unique |> size |> _[1] 
numDDates = @pipe df_route[!, ["ddate"]] |> unique |> size |> _[1]
numObs = size(df_route)[1]
df_route = df_route[!, ["fare", "tdate", "seats", "difS", "dd_dow"]]
df_route = df_route[.!ismissing.(df_route.fare), :]

k=8


it = 2
while true
    k = it
    fares = copy(collect(skipmissing(df_route.fare))) # correcting data type

    Random.seed!(567)
    kmean_res = kmeans(fares', k, tol=1e-4, init=Vector(1:k)) 
    rank_conversion = Dict(zip(sortperm(kmean_res.centers, dims=2, rev=true), 1:it))
    cents = sort(kmean_res.centers, dims=2, rev=true)
    assignments = get.(Ref(rank_conversion), kmean_res.assignments, missing)


    df_route[!,"fareI"] = assignments
    df_route[!,"fareC"] = cents[assignments]

    cc = cor(df_route.fare, df_route.fareC)^2

    println(it, ":", round(cc, digits = 2))

    it += 1
    if round(cc,digits=2) >= 0.99
        it -= 1
        println("Found it! : ", it)
        break
        
    end

end

prices = @pipe df_route.fareC |> unique |> sort |> collect |> _ ./ 100

T = length(countmap(df_route.tdate))
countmap(df_route.fareI)

data = convert.(Int,Matrix(df_route[!, ["seats", "difS", "tdate", "fareI", "dd_dow"]] ))

data

Pₜ = df_route[!, ["tdate", "fareI"]] |> unique 
Pₜ = unstack(Pₜ, :tdate, :fareI, 1, combine=sum)
Pₜ = sort(Pₜ, :tdate)
Pₜ = Pₜ[!, (1:it .|> x -> string(x))]
Pₜ = hcat((1:T).-1, (.!ismissing.(Pₜ) .* 1))
Pₜ = Matrix(Pₜ)

q̄ = Int(maximum(df_route.seats)+1)

numP = length(prices)
obs = length(df_route.tdate)
# This block of code creates the core estim data as well as key data summaries that enter LLN

first =  map(x -> x[1],argmin(Pₜ, dims = 1)) .-1
last = T .- map(x -> x[1], argmax(reverse(Pₜ, dims = (1,2)), dims = 1)) .+ 1

EC = 0.5772156649 # Euler constant

X₀ = [ # Initial values
    2.49999999, 2.49999999, 2.49999999, 2.49999999, 2.49999999, 
    2.49999999, 2.49999999, -1.05185291, -0.72189149, -13.39650409, 
    0.27373386, 0.0, 1.91183252, 2.46138227, 1.82139054, 2.35728083, 
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.22463165
]

VAR = X₀


test_dta = deepcopy(df_route[!, ["seats", "difS", "tdate", "fareI", "dd_dow"]])

unique(test_dta)

# Create model for optimization

bndsLo = [
        -10, -10, -10, -10, -10, -10, -10, -10, -10,
        -250, -10, -.06, .1, .1, .1, .1,
        .01, .01, .01, .01, .01, .01, .02
    ]
bndsUp = [
    15, 15, 15, 15, 15, 15, 15, 0, 0, 40, 10, .15,
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2
]



df_route








using Optim # for BFGS -> KN_HESSOPT_BFGS
using MKL # Using Julia with Intel's MKL -> KN_BLASOPTION_INTEL
using NLopt


n = length(X₀)
model = Model(NLopt.Optimizer)
set_optimizer_attribute(model, "method", BFGS())

df_route[df_route.seats .==120,:]




@variable(
    model, 
    X[i = 1:n], 
    start = X₀[i], 
    lower_bound = bndsLo[i], 
    upper_bound = bndsUp[i]
    )



@NLobjective(model, )


# [ ] Set lower boundaries of parameters
# [ ] set upper boundaries of parameters
# [ ] set variable types
# [ ] set the model objective
# [ ] set callback | investigate what it is ... 
