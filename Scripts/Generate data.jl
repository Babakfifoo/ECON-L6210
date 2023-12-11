# Create model for optimization
# Round df preparation:
import Parquet2, CSV
using DataFrames
using Pipe: @pipe

function get_data(MARKET = "BOS_SEA")
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

    data = convert.(Int,Matrix(df_route[!, ["seats", "difS", "tdate", "fareI", "dd_dow"]] ))

    Pt_m = df_route[!, ["tdate", "fareI"]] |> unique 
    Pt_m = unstack(Pt_m, :tdate, :fareI, 1, combine=sum)
    Pt_m = sort(Pt_m, :tdate)
    Pt_m = Pt_m[!, (1:it .|> x -> string(x))]
    Pt_m = hcat((1:T).-1, (.!ismissing.(Pt_m) .* 1))
    Pt_m = Matrix(Pt_m)

    q̄ = Int(maximum(df_route.seats)+1)

    return(data, prices, Pt_m, q̄, it)
end