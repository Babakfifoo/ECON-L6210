using DataFrames
import CSV

df_route = CSV.read("data/estimation/BOS_SAN/BOS_SAN.csv", DataFrame)
df_route_pt = CSV.read("data/estimation/BOS_SAN/BOS_SAN_pt.csv", DataFrame, header=false)
prices = CSV.read("data/estimation/BOS_SAN/BOS_SAN_prices.csv", DataFrame, header=false)

# betaHat
xInit = CSV.read("data/estimation/BOS_SAN/BOS_SAN_robust_params.csv", DataFrame, header=false)



Pt = Matrix(df_route_pt)[:,2:end]

data = Matrix(df_route)[:,2:end]
qBar = (df_route.seats |> maximum) + 1
obs = length(df_route.tdate)


e_c = 0.5772156649 # Euler constant
T  = 60


VAR = Vector(xInit.Column1)



# The first parameters are consumer preferences:

β = VAR[0:7]
b_L = min(VAR[7], VAR[8])
b_B = max(VAR[7], VAR[8])


γ = 1 ./ (exp.(-VAR[9] .- collect(0:60) .* VAR[10] .- (collect(0:60).^2) .* VAR[11]) .+ 1)

μₜ = VAR[12] * (T - 20) + VAR[13] * 7 + VAR[14] * 7 + VAR[15] * 6

