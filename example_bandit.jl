### A Pluto.jl notebook ###
# v0.14.1

using Markdown
using InteractiveUtils

# ╔═╡ db4a6b48-99a7-11eb-097b-676aed417c48
begin 

include("MultiArmBandit.jl")

using .MultiArmBandit 
using DataFrames 
using StatsBase, Statistics 
using Distributions
using Gadfly 
import Random 

end

# ╔═╡ b65dd189-456b-40ac-88fc-d05b5e29b396
md" 

# Multi-Arm Bandits

A Multi-Arm Bandit is a concept from Reinforcement Learning which is useful for decision making under uncertainty. Take a situation where you have limited time or resources to test multiple drugs, perhaps the COVID vaccines for example. While you could use the standard approach of a clinical trial with randomization, a balanced design, and performing AB tests, this results in many participants not recieving the best treatment, as some of them will be allocated to inferior choices. Ideally, we would like to do both: 1) Show a given treatment is the best choice and 2) Have most people end up benefitting from it. 


Using a strategy known as $\epsilon$-greedy Q learning, the Multi Arm bandit can help accomplish this task in an online manner where the treatments are allocated on the spot. The arm refers to different treatment groups that the bandit can select from. In the beginning, each treatment arm is assigned the same value of $Q_{tmt}=Q_{0}$. Some small amount of noise can be added to break ties. At each step, $(1-\epsilon)$ of the time the bandit randomly selects from the treatment with the highest $Q_{tmt}$ and updates its corresponding value. $\epsilon$ of the time, the bandit will select a random treatment instead of the one with the highest Q. The update rule is as follows: 


$Q_{tmt,t} = Q_{tmt,t-1} + \frac{R-Q_{tmt,t-1}}{N_{tmt,t}}$ 

where R is the reward, which is usually just the response Y corresponding to the treatment given. $N_{t+1}$ corresponds to the sample size in the given treatment group after the sample from the group is randomly selected. The bandit will be trying to go for the treatment which maximizes the reward, and so care should be taken that the response is transformed or sign-flipped if necessary so that higher is better. The above update rule can be shown to correspond to the calculation of the sample mean in an online manner if $Q_{0} = 0$. However, we can modify  the updates by adding an explicit learning rate parameter $\eta$:

$Q_{tmt,t+1} = Q_{tmt,t} + \eta(R-Q_{tmt,t})$

This can help improve learning. In this notebook, we will simulate data from 5 groups and examine the impact of setting different parameters for $Q_{0}$ and $\eta$. 

The source code for the functions used in this demo Pluto notebook can be found in the **MultiArmBandit.jl** module in this repo.

"

# ╔═╡ 38528382-a1ec-4117-88d4-fbcc019c7302
md"

## Simulate Data 

Below, we simulate n=10000 points from 5 groups each A, B, C, D, E from a normal distibution with mean $\mu = (-0.3,0.0,0.5,0.8,1.1)$ and $\sigma = (1, 0.5, 0.3, 0.3, 0.5)$ respectively. The bandit will select points from this simulated dataset. Note that in the ideal situation the ground truth in this simulated example is: 

$ E > D > C > B > A $

Ideally, the bandit will come to this same conclusion and choose E most of the time. However, the different SDs for the groups can complicate this choice, as the bandit is greedy and thus converging to E in a finite number of iterations (in the real world, this would correspond to resources or number of patients in the trial) can be more complicated. We can track the bandit's history and plot it over time to examine when convergence occurs. 

The sample mean/SD of each Arm are shown below. The bandit will be selecting samples from each group in an online manner, and so at each step it does NOT have access to every single data point. Instead it must figure out in the finite number of iterations which group is the best

"

# ╔═╡ 60e50c4e-8120-4306-9c6c-9282fff8eb47
begin 
	
n = 1000
Random.seed!(101)
A= rand(Normal(-0.3,1),n)
B= rand(Normal(0,0.5),n)
C = rand(Normal(0.5,0.3),n)
D = rand(Normal(0.8,0.3),n)
E = rand(Normal(1.1,0.5),n)

simdata = DataFrame(Arm = vcat(fill("A",n),fill("B",n),fill("C",n),fill("D",n),fill("E",n)),
                    Y = vcat(A,B,C,D,E))

summary_simdata = combine(groupby(simdata,:Arm),nrow,:Y=>mean=>:meanY,:Y=>std=>:sdY)
summary_simdata
end 

# ╔═╡ 65107f38-7916-43a4-b36f-1ef012c43b51
md"

## Create Bandit: $Q_0=0, \eta=\frac{1}{N_{tmt,t}}, \epsilon = 0.1$ 

Below we instantiate the first bandit with all default settings, and allow it to explore 10% of the time while 90% of the time it selects the current best group in a greedy manner. 

"

# ╔═╡ 8ed00c06-7348-49dc-a9f2-041465beb969
band1=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,track_hist=true)

# ╔═╡ 472fee46-6e48-47a7-9235-ccb7c460f735
begin 
Random.seed!(1001)
MultiArmBandit.Train!(band1,999,ϵ = 0.1)
end 

# ╔═╡ 1243d0ab-f95c-482d-b81e-3b3eb60907eb
band1_df = MultiArmBandit.GetHistoryDF(band1)

# ╔═╡ 794d15a3-a4b4-4ffa-9afb-bb19d79e79e5
plot(band1_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ 5e8e7408-aa48-4a2d-a427-40e8adf81d3c
band1_summary = summary(band1)

# ╔═╡ 32f83952-c660-4fde-8bd5-ae1a97e7bacd
md"

Based on the above figure, the correct ordering is found after around 250 steps. For the first few steps, the bandit picks Group D which is the 2nd best but eventually finds Group C as a result of the exploration. Lets try to see what happens if we lower the bandit's exploration by setting $\epsilon = 0.03$. That is, the bandit will select a group randomly (instead of the group it deems the best group at the moment) 3% of the time. 

## Create Bandit: $Q_0=0, \eta=\frac{1}{N_{tmt,t}}, \epsilon = 0.03$ 

"

# ╔═╡ 65945e4f-17e9-40ba-870f-88da0c5d13d0
begin
band2=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,track_hist=true)
Random.seed!(1102)
MultiArmBandit.Train!(band2,999,ϵ = 0.03)
band2_df = MultiArmBandit.GetHistoryDF(band2)
band2_summary = summary(band2)
end 

# ╔═╡ 3566a579-84dc-4cb6-91ab-833d0d5e9782
plot(band2_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ a15c8f8f-f077-4758-a668-01c8b562ca95
md"

In this situation, it took longer (~100 iterations) for the bandit to arrive at the best group

In the next section, we will change the starting value to be higher at $Q_{0} = 5$. This will tend to encourage a lot more early exploration as this value is much higher than what is possible based on sampling. 

## Create Bandit: $Q_0=5, \eta=\frac{1}{N_{tmt,t}}, \epsilon = 0.1$ 

"

# ╔═╡ 77d6e956-219e-40e9-978d-f7e0906a8d6c
begin
band3=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,Q₀=5,track_hist=true)
Random.seed!(901)
MultiArmBandit.Train!(band3,99,ϵ = 0.1)
band3_df = MultiArmBandit.GetHistoryDF(band3)
band3_summary = summary(band3)
end 

# ╔═╡ 043f9d99-e8b5-4b33-82ac-3870a2238f85
plot(band3_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ 50196d6d-8a52-4043-aa9f-e3f43a9ed2a1
md"

In this case, starting at a value much higher than possible leads to earlier exploration, resulting in the Q value for each group falling rapidly. However, the bandit arrives at the best option E much quicker since it is pulled down less and is essentially guaranteed to be selected since the bandit desperately switches options more in the beginning to try to maintain a high Q value. 

In this next part, we keep $Q_0 = 5$ but set the learning rate $\eta=0.01$

## Create Bandit: $Q_0=5, \eta=0.01, \epsilon = 0.1$ 


"

# ╔═╡ 9f3d4773-ac0a-4dd2-992d-b549f58f4021
begin
band4=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,Q₀=5,track_hist=true)
Random.seed!(555)
MultiArmBandit.Train!(band4,2999,ϵ = 0.1,η=0.01)
band4_df = MultiArmBandit.GetHistoryDF(band4)
band4_summary = summary(band4)
end 

# ╔═╡ 3789a09f-73fb-4ca8-b267-bbe2d7b32848
plot(band4_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ 7825e432-9f3b-4099-bf37-0b1b3ccf38c1
md"

In this case, setting the learning rate to 0.01 leads the bandit to take longer to reach the optimal solution E. Until around 1500 steps, the Q value for all the groups is even. 

What happens if we increase the learning rate to 1 (but keep $Q_0 = 5$) ? 

## Create Bandit: $Q_0=5, \eta=1, \epsilon = 0.1$ 



"

# ╔═╡ 2342907c-75e0-4126-b2df-6ce5a5901bf1
begin
band5=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,Q₀=5,track_hist=true)
Random.seed!(1556)
MultiArmBandit.Train!(band5,2999,ϵ = 0.1,η=1)
band5_df = MultiArmBandit.GetHistoryDF(band5)
band5_summary = summary(band5)
end 

# ╔═╡ 7c511272-01f8-4270-bdf7-f4f66e78df07
plot(band5_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ c1deba65-a93f-4992-958e-38a987346691
md"

Now the plot is extremely noisy, and the bandit cannot seem to differentiate between Group D and Group E very well. We can see in the summary that the bandit tended to select D the most but in the end the Q values of both D/E are about the same. 

In this final parts, we now set $Q_0=0$ again and examine the impact of the two learning rates. 

## Create Bandit: $Q_0=0, \eta=0.01, \epsilon = 0.1$ 


"

# ╔═╡ 8bbfae0c-6648-4f42-a40c-ada03d17bb3a
begin
band6=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,Q₀=0,track_hist=true)
Random.seed!(121)
MultiArmBandit.Train!(band6,2999,ϵ = 0.1,η=0.01)
band6_df = MultiArmBandit.GetHistoryDF(band6)
band6_summary = summary(band6)
end 

# ╔═╡ 613e3215-4402-4ba2-8402-2ab5bf1f1247
plot(band6_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ e7772e4c-b798-4bdd-85c1-6ee291fce282
md"

In this case, starting at $Q_0=0$ and setting the learning rate $\eta = 0.01$ leads to the bandit favoring Group C, a suboptimal group, for a long time. It is only after ~2700 iterations that it picks up on Group E. 

What happens if the learning rate is increased? Will we see the same noisy behavior as before when starting from $Q_0=0$? 

## Create Bandit: $Q_0=0, \eta=1, \epsilon = 0.1$ 



"

# ╔═╡ bf545060-f55c-4a91-99f8-c41543c25ebb
begin
band7=MultiArmBandit.CreateBandit(simdata,:Arm,:Y,Q₀=0,track_hist=true)
Random.seed!(13)
MultiArmBandit.Train!(band7,2999,ϵ = 0.1,η=1)
band7_df = MultiArmBandit.GetHistoryDF(band7)
band7_summary = summary(band7)
end 

# ╔═╡ 5af03195-5199-48b9-8792-f02da75dc320
plot(band7_df,x=:steps,y=:Q,color=:Group,Geom.line)

# ╔═╡ a94f1ce9-b874-4da2-af2b-ee8c6840a2d2
md"

The higher learning rate again leads to noise, and in the end, the bandit has trouble differentiating Group D and E just as before. 

In summary, lowering the learning rate appears to increase the chance of arriving at a suboptimal solution. Increasing the learning rate too much confuses the Bandit and the Q value curves become extremely noisy. 

Starting at a higher Q (with no set learning rate) encourages quicker exploration between the groups

"

# ╔═╡ Cell order:
# ╟─b65dd189-456b-40ac-88fc-d05b5e29b396
# ╠═db4a6b48-99a7-11eb-097b-676aed417c48
# ╟─38528382-a1ec-4117-88d4-fbcc019c7302
# ╠═60e50c4e-8120-4306-9c6c-9282fff8eb47
# ╟─65107f38-7916-43a4-b36f-1ef012c43b51
# ╠═8ed00c06-7348-49dc-a9f2-041465beb969
# ╠═472fee46-6e48-47a7-9235-ccb7c460f735
# ╠═1243d0ab-f95c-482d-b81e-3b3eb60907eb
# ╠═794d15a3-a4b4-4ffa-9afb-bb19d79e79e5
# ╠═5e8e7408-aa48-4a2d-a427-40e8adf81d3c
# ╟─32f83952-c660-4fde-8bd5-ae1a97e7bacd
# ╠═65945e4f-17e9-40ba-870f-88da0c5d13d0
# ╠═3566a579-84dc-4cb6-91ab-833d0d5e9782
# ╟─a15c8f8f-f077-4758-a668-01c8b562ca95
# ╠═77d6e956-219e-40e9-978d-f7e0906a8d6c
# ╠═043f9d99-e8b5-4b33-82ac-3870a2238f85
# ╟─50196d6d-8a52-4043-aa9f-e3f43a9ed2a1
# ╠═9f3d4773-ac0a-4dd2-992d-b549f58f4021
# ╠═3789a09f-73fb-4ca8-b267-bbe2d7b32848
# ╟─7825e432-9f3b-4099-bf37-0b1b3ccf38c1
# ╠═2342907c-75e0-4126-b2df-6ce5a5901bf1
# ╠═7c511272-01f8-4270-bdf7-f4f66e78df07
# ╟─c1deba65-a93f-4992-958e-38a987346691
# ╠═8bbfae0c-6648-4f42-a40c-ada03d17bb3a
# ╠═613e3215-4402-4ba2-8402-2ab5bf1f1247
# ╟─e7772e4c-b798-4bdd-85c1-6ee291fce282
# ╠═bf545060-f55c-4a91-99f8-c41543c25ebb
# ╠═5af03195-5199-48b9-8792-f02da75dc320
# ╟─a94f1ce9-b874-4da2-af2b-ee8c6840a2d2
