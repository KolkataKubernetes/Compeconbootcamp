#THE OPTIMAL INVESTMENT PROBLEM: See slides for conceptual walkthrough, and notes from Job's quarter if you need a refresher.

using Parameters #This package gives us @with_kw, which lets us define default values within a struct

@with_kw struct ModelParameters

    β::Float64 = 0.99 #Discount rate. Recall we can define these starting values because of @with_kw
    δ::Float64 = .025 #Depreciation rate
    α::Float64 = .36 #Production technology

    #next, specify a grid of values for capital to search over
    k_grid::Vector{Float64} = collect(range(.1,length = 1800,stop = 45.0)) #1800 evenly spaced values between .1 and 45
    N_k::Int64 = length(k_grid)

    tol::Float64 = 10^-4 #Tolerance for when function stops iterating through the contraction mapping; we have convergence when v_t-v_{t-1} is less than tol

end 

@with_kw struct ModelSolutions #Model solution will store two things
    V::Vector{Float64} #Store the current guess of the value function
    kp::Vector{Float64} #Store the policy function, k' that the agent chooses

end 

#So how do we construct model solutions? Start at 1:02.

function build_ModelSolutions(para)

    V = zeros(Float64,para.N_k) #Start with guess of zeros that are the length of the grid
    kp = zeros(Float64,para.N_k) #Empty vector to begin guessing. Intuition: Need policy and value function for each state that I'm in 
 
    sols = ModelSolutions(V,kp) #Creates ModelSolution struct (defined above which will return answers )

    return sols
end

#To walk through this again, try listening to Jon's lecture from 1:02 onward.

function build_structs()
    para = ModelParameters() #the default values
    sols = build_ModelSolutions(para) #Builds the solution objects from the Parameters

    return para, sols

end

## The above will take time to understand. I recommend recreating step-by-step in Notion and annotating the value of each step

#= NEXT: CONSTRUCTING THE BELLMAN OPERATOR
The Bellman operator will take the current guess and compute the next value of the Bellman iteration

=#

### Bellman operator

function bellman(para, sols)
    #Take the structs and unpack all of them: This gives us delta, and other parameters
    @unpack_ModelParameters para
    @unpack_ModelSolutions sols

    V_next = zeros(Float64, para.N_k) #Vector of new value function defined on each of the possible state values
    kp_next = zeros(Float64, N_k) #Next policy function defined on each possible state value

    #The goal: Look for the k' that maximizes utility on a discrete set of values

    for i_k = eachindex(k_grid) #For each index in the grid of possible capital values... (iterating over index of k grid)
        max_util = -1e10 #Starting utility value. To find the max, start with the worst we could do. Then check/replace each capital value in the grid to see iteratively what beats new maxutil.
        k = k_grid[i_k] #access the specific value k cited in the index
        budget = k^α + (1-δ)k #given current capital, here's the budget constraint
        for i_kp = eachindex(k_grid)

            c = budget - k_grid[i_kp] #Calculate consumption from budget - capital in k grid

            if c > 0 

                V_temp = log(c) + β*V[i_kp]

                if V_temp > max_util
                    max_util = V_temp #Update the maximum utility value
                    kp_next[i_k] = k_grid[i_kp] #Update the optimal choice of capital for tomorrow (policy function)
                end
            end
        end
        V_next[i_k] = max_util #Based on the maximum utility we found, update the value function at the ith position
    end

    return V_next, kp_next
end


#Finally, we need a function to solve the model! 

### Solve Model
function solve_model(para,sols)
    para,sols = build_structs()
    @unpack_ModelParameters para
    @unpack_ModelSolutions sols

    V_next = zeros(Float64, N_k) #Start with an empty guess for the value function
    max_diff = tol + 10.0 #The starting error: Arbitrarily large to kick off the process
    n = 0 #number of iterations

    while max_diff > tol #While the value functions are sufficiently far apart
        n += 1
        V_next, kp_next = bellman(para,sols) #Update value and policy functions based on Bellman eq.

        max_diff = maximum(abs.(V_next - V)) #update the max diff. Recall "." applies to all values in the vector.

        sols.V .= V_next #Update struct solutions
        sols.kp .= kp_next 

        @show n, max_diff

    end

    sols 

end

para, sols = build_structs();

@time sols = solve_model(para,sols)

#Plot the results

using Plots
plot(para.k_grid, sols.V)
plot(para.k_grid, sols.kp)
plot!(collect(0:45), collect(0:45))