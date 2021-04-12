module MultiArmBandit

using Statistics, StatsBase, Distributions
using DataFrames
using RCall

export Bandit, CreateBandit, Train!
export GetHistory, TrackHistory!, ClearHistory!
export GetHistoryDF


"""Bandit

struct holding multi arm bandit state 

args: 

QN: Dict{NamedTuple,Dict{Symbol,Float64}} Holds reward Q and sample size N for each group 

Y: Contains data for each group in a Dict 

keylist: vector of keys for hashing the bandit 

history: Contains history of QN values if track_hist == true in CreateBandit(). User should 
use GetHistory() or GetHistoryDF() to access this in a cleaner data structure. 


"""
mutable struct Bandit
    QN::Dict{NamedTuple,Dict{Symbol,Float64}}
    Y::Dict{NamedTuple,Array{Float64,1}}
    keylist::Vector{NamedTuple}
    history:: Union{Vector{Dict{NamedTuple,Dict{Symbol,Float64}}},Nothing}
end

"""CreateBandit(data,groups,response;Q₀=0,track_hist=false)

This function instantiates the Multi Arm Bandit object where each arm is started with 
a Q-value drawn from Unif(Q₀,Q₀+1e-10). The slight perturbation of Q from Q₀ with 
uniform noise breaks ties on step 1 when the bandit is trained in the ϵ-greedy manner
to maximize Q. 

args: 

data: DataFrame containing data 

groups: Vector of Symbols containing Groups (arms) to group by 

response: Symbol corresponding to response Y 

Q₀: Initial Q value to start each group at (before), 0 is default. Starting higher than the expected 
reward for any group will lead to more exploration by the bandit  

track_hist: boolean to track history of the bandit 

"""
function CreateBandit(data,groups,response;Q₀=0,track_hist=false)
    gdf = groupby(data,groups)
    QN = Dict{NamedTuple,Dict{Symbol,Float64}}()
    Y = Dict{NamedTuple,Array{Float64,1}}()
    for key in keys(gdf)
        QN[NamedTuple(key)] = Dict(:Q=>rand(Uniform(Q₀,Q₀+1e-10)),:n=>0.0)
        Y[NamedTuple(key)] = gdf[key][:,response]
    end
    if track_hist==false
        return(Bandit(QN,Y,collect(keys(QN)),nothing))
    elseif track_hist==true
        return(Bandit(QN,Y,collect(keys(QN)),[deepcopy(QN)]))
    end
end

#Allows bandit to be hashed like a dict 

Base.getindex(bandit::Bandit,key) = Base.getindex(bandit.QN,key)

Base.keys(bandit::Bandit) = Base.keys(bandit.QN)
Base.values(bandit::Bandit) = Base.values(bandit.QN)

# 
"""
    argmax(bandit::Bandit)

Returns the group which has the current maximum Q value 

"""
function Base.argmax(bandit::Bandit)
    keylist = bandit.keylist
    idx = argmax(getindex.(values(bandit.QN),:Q))
    return(keylist[idx])
end

""" Train!(bandit::Bandit,steps;ϵ=0.05,η=nothing)

This function trains the bandit for a given number of steps in an ϵ-greedy manner. η is an 
optional learning rate to modify the update rule: 

Default: Q = Q + (R-Q)/N 

With Learning Rate: Q = Q + (R-Q)*η

At each step, the bandit selects a Y (response) from the group with the maximum Q at the moment as 
the reward R. ϵ of the time, the bandit will select a group at random for which to select the reward. 

The default update rule is derived from computing the sample mean in an online manner. 

""" 
function Train!(bandit::Bandit,steps;ϵ=0.05,η=nothing)
    lrn = ifelse(isnothing(η),0,1)
    track_hist = ifelse(isnothing(bandit.history),0,1)
    for i=1:steps
        u = rand(Uniform(0,1))
        if u <= ϵ
            action = sample(bandit.keylist,1)[1]
        else
            action = argmax(bandit)
        end
        Qcurrent = bandit[action][:Q]
        Reward = sample(bandit.Y[action],1)[1]
        diff = Reward - Qcurrent
        curr = bandit[action]
        curr[:n] += 1
        if lrn == 0
            curr[:Q] += 1/curr[:n] .* diff
        else
            curr[:Q] += η .* diff
        end
        if track_hist == 1
            push!(bandit.history,deepcopy(bandit.QN))
        end
    end
    return(bandit)
end

"""GetHistory(band::Bandit)

Returns the bandit Q history per group in a Dict if history was tracked

"""
function GetHistory(band::Bandit)
    isnothing(band.history) && throw(DomainError("No History was tracked"))
    intermediate_dict = Dict()
    for x in band.history, (k, v) in x
        push!(get!(intermediate_dict, k, Vector{typeof(v)}()), v)
    end
    histresult = Dict{NamedTuple,Dict{Symbol,Vector{Float64}}}()
    for key in keys(intermediate_dict)
        temp = Dict{Symbol,Vector{Float64}}()
        for x in intermediate_dict[key], (k, v) in x
            push!(get!(temp, k, Vector{Float64}()), v)
        end
        histresult[key] = temp
    end
    return(histresult)
end

"""TrackHistory!(band::Bandit)

Turns on bandit history tracking if history was not being tracked before 

"""
function TrackHistory!(band::Bandit)
    isnothing(band.history)==false && throw(DomainError("Bandit History is already being tracked"))
    band.history = [deepcopy(band.QN)]
    print("\n Bandit training history will now be tracked \n")
end

""" ClearHistory(band::Bandit;off=false)

Clears the bandit history. Can also turn off history tracking if off==true 

""" 
function ClearHistory!(band::Bandit;off=false)
    if off==true
        band.history = nothing
        print("\n Bandit training history cleared and turned off \n")
        return(nothing)
    else
        band.history = [deepcopy(band.QN)]
        print("\n Bandit training history cleared")
        return(nothing)
    end
end

""" GetHistoryDF(band::Bandit)

Returns history of the bandit per group in a DataFrame in long format. This is recommended 
if one wants to plot the history with other packages using Grammar of Graphics, such as 
Gadfly in Julia or ggplot2 in R. 

"""
function GetHistoryDF(band::Bandit)
    hist = GetHistory(band)
    histdf = DataFrame()
    for k in keys(hist)
        Q = hist[k][:Q]
        n_Arm = hist[k][:n]
        n=length(Q)
        d = DataFrame(steps=1:n,Group=fill(k,n),Q=Q,n_Arm=n_Arm)
        histdf = vcat(histdf,d)
    end
    histdf.Group = categorical(string.(histdf.Group))
    return(histdf)
end

""" summary(band::Bandit)

Returns a summary of the current bandit status (Q and N) as a DataFrame 

"""
function Base.summary(band::Bandit)
    summarydf = DataFrame()
    for (k,v) in band.QN
        df = DataFrame(Group=k,n=v[:n],Q=v[:Q])
        summarydf = vcat(summarydf,df)
    end
    return(summarydf)
end

end

