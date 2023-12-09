using DataFrames, JuMP
using Statistics, Clustering, HypothesisTests, StatsBase
using Query, FreqTables
using ShiftedArrays
using Dates, Missings, Random, CategoricalArrays 
using Pipe: @pipe
import Parquet2, CSV

function determine_OD(O,D)
    res = string(min(O,D), "_", max(O,D))
    return res
end

INPUT = "data"

df = Parquet2.Dataset("$INPUT/asdata_clean.parquet") |> DataFrame

df[!, "route"] = determine_OD.(df[!,"origin"], df[!,"dest"])

cols = ["origin", "dest", "ddate", "flightNum"]
df[!,"ones"] .= 1

df[!,"numObs"] = transform(
    groupby(df,cols),
    :ones => length
    ).ones_length

df[!, "dd_dow"] = dayofweek.(df.ddate) .+1
df = filter(row -> row.numObs >= 59, df)

df[!, "ttdate"] = -df.tdate .+ 60

cols = ["origin", "dest", "ddate", "flightNum", "tdate"]
df.ddate = categorical(string.(Date.(df.ddate)))
df = df.sort_values(cols, ascending = False).reset_index(drop = True)

