############################
#= STOCHASTIC VARIABLES [Slides]
John's commentary primarily follows the slides. I'll add comment for each slide where color commentary is required
=#
############################

#SLIDE 1
## Lots of heterogeneity in real world situations. Not just one type of agent! There could be random latent "types" for whom we don't observe the Distributions

##We need to estimate the proportion of observations that may fit a "type", for example.

##Or: What if we don't have a closed form solution for a distribution? 

##Continuation valuies: Depend on a conditional expectation, which is a function of agent action

#SLIDE 2

#SLIDE 3: Birthday problem
## If there ar 'N' people in a room, what is the probability that at least two of them share the same birthday?

## Problem bucks traditional statistical intuition. When you think about it probabilistically, you need all people have UNIQUE birthdays!
## Presume all days are equally likely

## Nasty binomial integral results in analytic solution

############################
#From Code Along: Birthday Problem
############################

#Packages
using Random, Plots, Distributions, Statistics, Parameters


function birthday(n, sims)
    results = zeros(sims) #Results matrix starts with 0'safely
    for i = 1:sims #loop over simulations
        days = rand(1:365, n) #draw birthdays
        results[i] = length(unique(days))
    end
    results
end

res_20 = birthday(20,100000)
histogram(res_20)
#Need to define a relationship between n and the probability that at least two people share teh same birthday!

prob_same_bday_20 = sum(res_20.< 20.0)/length(res_20) #Vector of conditional sums #How many times are there at least 2 people sharing 1 bday?

res_50 = birthday(50,100000)
histogram(res_50)
#Need to define a relationship between n and the probability that at least two people share teh same birthday!

prob_same_bday = sum(res_50.< 50.0)/length(res_20) #Vector of conditional sums #How many times are there at least 2 people sharing 1 bday?



############################
#From Code Along: Static Arrays

#CONTAINS INFO ON HW 1 # 7!
############################

# In the homework 1 question 7, you had to calculate the expectation w.r.t the 
# transition matrix Π. 

# You could do this via matrix multiplication: EV = Π[1,:]' * V. Here, V = [V_g; V_b].

#=
Static arrays work by storing the vector in a different part of the memory. It's convenient because
no assumptions are made about size or type. However, the computational load increases with size.
=#

# Working with arrays such as matrices can be quite slow because although the number of
# dimensions is fixed, the size of each dimension can change. 

#EXAMPLE OF STATIC ARRAYS IN Use

using BenchmarkTools, StaticArrays

function add_up(M,R) #M is a 2x2 matrix, R is a 2x1 vector. Notice we're not defining either object!
    total = 0.0

    for i = 1:1000000
        total += M[1,:]' * R #Matrix multiplication but first row
        total += M[2,:]' * R #Matrix multiplication but second row
    end

    total

end



#If we drop the slicing syntax:

function add_u2(M,R)
    total =0.0

    for i = 1:100000
        total += M[1,1]*R[1]+M[1,2]*R[2]
        total += M[2,1]*R[1]+M[2,2]*R[2]
    end
    
    total
end


M = rand(2,2)
R = rand(2)

@benchmark add_up(M,R) #slow

@benchmark add_u2(M,R)

#Further improvements: We can use a static matrix/vector for our inputs!
M_static = SMatrix{2,2}(rand(2,2)) #Imposing static 2x2 definition

R_static = SVector{2}(R) #Always has 2 elements- we're just choosing R for convenience

@benchmark add_up(M_static,R_static) #Much faster! 

@benchmark add_u2(M_static,R_static) #This is the same with addup 2.

#=
Notice that there's not much gain with add_u2, once we change inputs to a static configuration. We'd probably
want to use add_up with static inputs since it allows for more flexible dimensionality.
=#

# Static arrays are generally good for "small" arrays. This is static arrays are stored
# in a different part of memory, and it takes a longer time to generate them. 
# For example, the following line of code on my computer never ran in a reasonable time.
# M_big_static = SMatrix{100,100}(rand(100,100))
# In general, use static arrays when they are "Small". You might have to play around 
# with this. A very rough rule of thumb is that you should consider using a normal 
# Array for arrays larger than 100 elements. 

################################################################################
# Average distance between two random points in a cube

#Example of Monte Carlo Integration

#Look at slide 6 for the integral equivalent. Nasty!
################################################################################

function point_distance(sims)
    results = zeros(sims)
    for i = 1:sims #loop over simulations
        p1, p2 = rand(3), rand(3) #two points in 3D space, each dimension has domain [0,1]
        results[i] = sqrt(sum(p1.-p2).^2)
    end
    mean(results)
    
end

point_distance(1000000) #Very performant! And not difficult to dream up computationally


################################################################################
# Expected Value of College Given Wage offer Shock
#Start on slide 7. Second subscripts need to be corrected

#s is a discrete choice variable: Will you work or not
#two period model. In stage 1, you decide whether to get a degree. In stage 2, you decide whether to work.

#Crucial piece of this model is the continuation value 's'. If ϵ is too low, we won't work! It also 
#affects the education decision.

#Need to solve for the cutoff value of ϵ for each schooling decision, and then take a conditional expectation.

#We will simulate the above.
################################################################################

#Create primitives

##primitives with setup similar to HW:
using Parameters

@with_kw struct Primitives
    β_0::Float64 = 2.7 #wage constant
    β_1::Float64 = 0.47 #college premium
    σ::Float64 = .597 #wage offer SD
    α::Float64 = 1.0 #leisure
    B::Float64 = 5.0 #base consumption
    d::Float64 = .25 #Opportunity Cost of College
end

##results to recover

mutable struct Results
    emax::Array{Float64,1} #First for no college, second for college
    lfp::Array{Float64,1} #LFP probabilities
    ewage::Array{Float64,1} #Average wages
    ewage_obs::Array{Float64,1} #Average observed wages
end

#solve model, i.e. solve for value function 
function Solve_model(sims)
    prim = Primitives() #defined in the struct
    res = Results(zeros(2),zeros(2),zeros(2),zeros(2)) #Why is each guess component size 2? Because we want to measure both utility stages
    #initialize results
    compute_emax(prim,res,sims) #this function is defined below
    prim,res
end

#compute_emax: the output below will go into res.

function compute_emax(prim,res,sims)
    #unpack the primitives so we can manipulate them
   @unpack β_0, β_1, σ, α, B, d = prim
   dist = Normal(0,σ)
   val_nwork = α + log(B) #the agent's utility model if they choose not to work


#Set 
   utils, lfps, wages = zeros(2, sims), zeros(2,sims), zeros(2,sims)

   for s = 0:1 #loop over schooling levels
    for i = 1:sims #loop over simulations
        ϵ = rand(dist) #draw shocks to compute resultant wage
        wage = exp(β_0 + β_1*s + ϵ) #Agents consume their entire wage? Yes: Two stage model so no savings incentive
        util = max(log(wage), val_nwork) #choose the max of no work vs. log wage
        utils[s+1, i] = util #update the result. Two stages of utility.
        lfps[s+1,i] = (log(wage)>val_nwork) #labor factor productivities: Essentially your policy function
        wages[s+1,i] = wage #store the wage 
    end
end

#The above computes our optimal values, but we still need to store them! 
res.emax[1], res.emax[2] = mean(utils[1,:]), mean(utils[2,:]) #expected max utilities
res.lfp[1], res.lfp[2] = mean(lfps[1,:]), mean(lfps[2,:]) #LFP probabilities
res.ewage[1], res.ewage[2] = mean(wages[1,:]), mean(wages[2,:]) #Average Wages
#we want to take the interaction between the wage people get versus the probability of their edu. choice
res.ewage_obs[1], res.ewage_obs[2] = mean(wages[1,:].*(lfps[1,:]))/res.lfp[1], mean(wages[2,:].*(lfps[2,:]))/res.lfp[2]

end

#Solve the model
prim, res = Solve_model(100)

res.ewage[2]/res.ewage[1]

res.ewage_obs[2]/res.ewage_obs[1]


#Observed college wage premium is biased down! Can you figure out why?

res.lfp[2]/res.lfp[1]

################################################################################
# Halton Sequences

#Good if you want to achieve better coverage of a distribution at a given sample size, better than
#that provided by a random process
################################################################################

using HaltonSequences, Plots

B = 1000
r = rand(B)
histogram(r)

halt = Halton(3)[1:B] #Uses base 3, creates vector of length B. Consider using shuffle!
histogram(halt)


################################################################################
# Quadrature
#Slide 12
#Sum approximation of an integral. Unlike Reimann where we use interval length,
#we instead use a set of weights for each function value ("nodes").
################################################################################

using FastGaussQuadrature, LinearAlgebra

#Gauss-Legendre Quadrature for approximating integral between -1 and 1
#Works well with n nodes if function is well approximated by polynomial of degree n
#Get nodes x and weights w
x, w = gausslegendre(3) #2 sets of 3-length nodes, weights

#Function F
F(x) = x^4 - 3x^3 - 4x^2
#Derivative of function f
f(x) = 4x^3 - 9x^2 - 8x
#Calculate intergral appoximation using nodes and weights
I = sum(w .* f.(x))
#Calculate true value
F(1) - F(-1)


# Gauss-Hermite Quadrature for approximating integral of function g(x) = f(x)*exp(-x^2) - integral over normal dist
# Again, works well with n nodes if f is well approximated by polynomial of degree n
# Here we will approximate the expection E[f(x)] when x is distributed according to standard normal

#Get nodes and weights
x,w = gausshermite(3)
#Change of variable: https://en.wikipedia.org/wiki/Gauss-Hermite_quadrature
g_tilde(x,f) = f(sqrt(2)x)/sqrt(π)

#Expectation approximation
sum(w.*g_tilde.(x,f))

#Monte Carlo Simulation Approximation
val = 0
N = 1000000
for i = 1:N
    val += f(rand(Normal()))/N
end
val