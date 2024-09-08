# MACHINE CODE AND OPTIMIZATION: PRELIMINARY INTUITION (Line 230 onwards: Optimal investment problem)

#= 
What is Julia during "under the hood" when I run code? The @code_native function tells us how Julia translates things to Machine Code.
=#
@code_native 2+2

# SCOPE: LOCAL VS. GLOBAL

## Scope is essentially the "domain" under which code and objects are accessible.

## Global Scope: Anything defined outside of functions, for loops, conditionals etc. 

x = 2 ### This is defined globally: It is a global variable.

x + 4 ### We can operate on X in the global scope 

## Local scope 

### Functions

function f() ### When you make a function, things are defined in a local scope 
    z = 3 ### Things defined inside this function are ONLY defined in the local scope
    return z
end
f()


## Things defined in local scope and not be recalled in the global environment. Calling z in the terminal will result in an error.

### It's easy to get mixed up between global and local variables. Define global variables sparingly. 
### Your code will also be more performant as you define more variables locally! 
### Stay away from global variables unless you absolutely have to use them.

### For Loops

for i in 1:10
    y = 7
    @show y + i
end 

### y is similarly undefined in global scope.

# FUNCTIONS WITH MULTIPLE OUTPUTS

##This works similarly to Python.

function find_eq()
    quantity = 10.0
    price = 2.0
    return quantity, price
end 


eq_q, eq_p = find_eq() #assigning each output to a variable

println("The equilibrium quantity is: ", eq_q)
println("The equilibrium price is: ", eq_p)

eq = find_eq() #alternatively you can use one variable to contain both OUTPUTS
println(eq)
println("The equilibrium quantity is: ", eq[1])
println("The equilibrium price is: ", eq[2])

# MACHINE CODE AND OPTIMIZATION: TIPS FOR SUCCESS

#=
TIP 1: Write things inside functions!

Julia is fast because of JIT compilation. A piece of code only needs to be compiled once; Julia pre-compiles functions.
Also, functions make your code very readable.
=#

# Example 1: VERY BAD. Everything here is defined in global. 

m = 3 
count = 0

@time for i = 1:10^7 #Time function will tell us how much time it takes to execute.
    count = 0
    count += m*i
    count -= m*i #This function adds and subtracts a number from 0 a HUGE number of times.
end 

# Example 2: A little better...

    ##Define a function... The function replaces teh for loop, though my variables are still defined in Global.

function update(x,m,i)
    x += m*i
    x -= m*i
    return x
end 

@time for i = 1:10^7 #Same for loop as last time, but just use the function:
    count = 0
    count = update(count,m,i)
end 

#= 
Much fewer allocations, more garbage collection 
but we have a lower execution time since code is pre-compiled
=#

# Example 3: Everything in local scope, much faster! 

function g(m) #takes input m 
    
    count = 0.0 #count is now defined in the local scope! 

    for i = 1:10^7
        count = update(count,m,i)
    end

    return count

end 

@time g(m)

#= 
This is the crowning jewel feature of Julia; right things in functions so you can 
take advantage of pre-compilation!
=#

#= TIP 2: Types and Structs
Types describe the structure and behavior of an element
We have already seen many types in lect 1.

=#

#TYPES

x = 40 
typeof(x)
x = 40.0
typeof(x)

#See codealong 2 for different data type examples

#= 
Moral: it takes time to convert things from integer to float. If you know
that you'll be using mult/division operations to begin with, just define things 
as floats. There are efficiency gains to be had from not switching between types.
=#

## Types of arrays/vectors: notice that these types default to floats.

typeof(zeros(2))

typeof(ones(3,2))

    ### I can coerce the matrix to an integer type as part of a zeros arg:

typeof(zeros(Int64,3,3,3,3))

# You can coerce a type when calling a variable. This is mostly unnecessary.

x = 32.0::Float64
typeof(x)

# When you define a function, you can specify argument types:

function h(x::Int64)
    return x*100
end

h(10) #The function works with integers
h(10.0) #The function will not work with floats

##Julia can infer types, which allows for more flexibility, so this isn't stricly necessary; inferring types does take more time however.

## It's considered a best practice to declare types: It speeds up code and also helps you catch errors.

#Structs

##you can define your own type using the struct keyword. 

struct MyType 
    a
    b ##this Struct has two fields
end

##Using the structure:
example = MyType(25.0,"abc")

example.a


#= TIP 2 FROM CODEALONG:
Define concrete types in your structs. It saves time!
=#

struct Container_any
    a
end

struct Container_float
    a::Float64 #I will get an error if I don't assign a float to 'a'.
end

#What happens if we assign a float to both? We get both as float 64.
c_any = Container_any(1.0)
c_float = Container_float(1.0)

#let's define a function that operates on our containers

function f(x)
    tmp = x.a
    #Random calculations
    for i in 1:10^7
        tmp = i + x.a
    end
    return tmp
end 

#Let's see how fast our function is, based on the struct with a defined type versus a struct without a defined type

@time f(c_any)
@time f(c_float) #Again: HUGE efficiency gains


#=
DIAGNOSTIC TIP: Use Benchmark to time things precisely and get a sense of how long things take to run
=#

using BenchmarkTools
@benchmark f(c_any)
@benchmark f(c_float)


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
    @unpack_ModelSolutions sors 

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

        #show n, max_diff

    end

    sols 

end

para, sols = build_structs();
