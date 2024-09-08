#How many threads/cores available on computer?

versioninfo()

################################################################################
# Multi-Threading
################################################################################

# You will need to load julia with multiple threads. 
# In VScode, go to Gear Icon in bottom-left, settings, search threads, click edit in 
# settings.json file, then change the number of threads. The number must be less than
# or equal to the number of threads available on your computer. 
# You will also need to restart the Julia REPL to update the setting. 

# On Linstat/slurm, to start julia with N threads use: julia --threads=N

versioninfo()

#get the thread ID 
Threads.threadid()

#Number of threads
Threads.nthreads()

################################################################################
# Basic threading example: For loop
################################################################################

#not multithreaded

for i = 1:Threads.nthreads()
    println(Threads.threadid())
end

#Multithread...

Threads.@threads for i = 1:Threads.nthreads()
    println(Threads.threadid())
end

#= These don't return in a deterministic sequential order. 
You have to be careful about threads changing the same array, especially if order matters.
We call this issue a "data race" issue.
=#

################################################################################
# Basic threading example: data race, pros/cons of threading ("sleep" example)
################################################################################

# All threads share the same memory, which can cause problems
# Let's add together some random numbers

my_vec = rand(1_000_000)

#single thread sum
sum(my_vec)

#multi-threaded sum
function sum_mt(x)
    total = 0.0
    Threads.@threads for i = 1:length(x)
        total += x[i]
    end
    total
end

sum_mt(my_vec) #The sum is different! Why?

#= The threads are overwriting eachother based on relative speeds. There's some 
randomness in the order of returning things. So how do we fix this?
=#

#solution
function sum_mt2(x)
    total = zeros(Threads.nthreads()) #now a vector totals! One space for each thread
    Threads.@threads for i = 1:length(x) #For each thread,
        total[Threads.threadid()] += x[i] #Add to specific thread's running total
    end
    sum(total)
end

sum_mt2(my_vec)

@time sum_mt2(my_vec)

#Threading is actually slower than looping in series because this was not a complicated computation. 
@time sum(my_vec)

#There's a lot more writing and processing that has to be done when you multithread - be judicious.

#Multithreading is faster when calculation inside for loop takes non-trivial amount of time. 

#For loop in series (not multi-threaded)
@time for i = 1:Threads.nthreads()
    sleep(1) #Wait for one second
end

#For loop multi-threaded
@time Threads.@threads for i = 1:Threads.nthreads() #each thread gets a copy of this code
    sleep(1)
end

################################################################################
# Multi-Processing
################################################################################

using Distributed #Gives functions and macros for distributed/multi-processing

################################################################################
# Set-Up for Parallelization
################################################################################

#How many processes are running in my session?
nprocs()

#How many workers?
nworkers()

#Add 2 processes/workers
addprocs(2)

#=
Note: One of the processes is the master if you have more than one process. 
Master distributes tasks and work to the workers, but typically does not do 
any work itself. In order to get speed gains from distributing, you will need
to add at least two workers
=#

#each process has an id
myid()

#@everywhere will run the following line of code on all processes. 
@everywhere @show myid()

#Get a list of processes and workers
procs()
workers()

#@spawn sends a line of code to an available worker

chosen_worker = @spawn 4+5  #Returns a Future object, which is a promise from the worker to do this when you want it. 
#Fetch will make worker fulfill its promise. 
fetch(chosen_worker)

#We can specifically choose the worker by using @spawnat
chosen_worker = @spawnat 2 myid() #This will always go to worker 2

################################################################################
# Memory across processes
################################################################################

#Define r on master processes
r = rand()

#Other workers don't know what r is. 
@everywhere @show r

#None of the other workers have any idea what you're talking about!

#Define a different r on every process
@everywhere r = rand()
@everywhere @show r #Every process generated a different random number!

#But what if we want to have one version of r?

function my_function(x,y)
    val = x+y
    val
end
my_function(4,5)
@everywhere my_function(4,5)

#alternatively... 
@everywhere function my_function(x,y)
    val = x+y
    val
end

@everywhere @show my_function(4,5)

#What if we want to pass multiple objects at the same time to every core? Look into this later.


################################################################################
# Simple Example
################################################################################

#Suppose we have a vector of numbers x = [x₁, x₂, ..., xₙ]
#We want to calculate the sum of their square roots S = Σ(xᵢ^0.5)

@everywhere function sqrt_sum(A)
    S = 0.0
    for i = 1:size(A)[1]
        S += sqrt(A[i])
    end
    S
end

A = rand(1000)
@time sqrt_sum(A)

#= We defined the function everywhere, but ran it on a single core. How about distributing it?
=#

#We can also break the sum up into batches and send to our workers using @distributed. 
#The @sync macro tells the master to wait until the workers are done with the for loop. 
    function sqrt_sum_distributed(A, N_batches)
        N = length(A)
        batch_size = Int(length(A)/N_batches)

# The (+) in front of the for loop will add together the results of the calculations
# inside the for loop and saves it as S
S = @sync @distributed (+) for batch in [(1:batch_size) .+ offset for offset in 0:batch_size:(N-1)] #list comprehension 

sqrt_sum(A[batch])

    end
end


@time sqrt_sum_distributed(A, 100) #Not that much faster...


#What if we allocated tasks dynamically using pmap?

#We can also run this in parellel using pmap()
function sqrt_sum_pmap(A, N_batches)

    N = length(A)
    batch_size = Int(length(A) / N_batches)

    S = sum(
            pmap(
                batch -> sqrt_sum(A[batch]), #Operation to be done by workers 
                [(1:batch_size) .+ offset for offset in 0:batch_size:(N-1)] #Vector of inputs to be broken up accross workers
                )
            )

    return S
end

@time sqrt_sum_pmap(A, 100) #takes the same amount of time

#=
Here, because the calculations we are doing inside the for loop (sqrt and add), 
there is not much speed gain from parallelization. Although we can go through these
calculations twice as fast on our two workers, there is an "overhead" cost of 
passing information back and forth between master and worker. Paralleization has
performance advantage when this overhead cost is cost is small relative to the computation
time inside the loop. 

If you add a sleep command inside sqrt_sum(), you will see that @distributed and pmap
are now about twice as fast as just runnning on the master process. 

@distributed tends to run faster than pmap when the calculations inside the for loop are quick. 

pmap is better when each iteration of the for loop takes a longer time. 
=#

################################################################################
# Shared Arrays
################################################################################

#Shared Arrays are Arrays that both master and workers have read and write access to. 
using SharedArrays

# Normal Array defined on the Master Process
A = rand(10000)

# Inside a distributed for loop, workers will look for variables in the master's global Memory
# if that name variable doesn't exist in the worker's memory. 
# Worker's will copy this variable to their own global memory
@sync @distributed for i = eachindex(A)
    A[i] + 1
end

#The master array didn't change! We need an array type that can be accessed and shared across workers.

#With shared arrays, workers have read and write ability onto the master's memory
A_shared = SharedArray{Float64}(A)

#Now let's try again...

@sync @distributed for i = eachindex(A_shared)
    A_shared[i] = A_shared[i] + 1
end

A_shared


###################################################################################################
# Parallelized Optimal Investment Problem

# Grid search similar to hw 1 
###################################################################################################

###################################################################################################

using Distributed
addprocs(2)

@everywhere using Parameters, SharedArrays

### Struct for our model paramters
@everywhere @with_kw struct ModelParameters

    β::Float64 = 0.99 
    δ::Float64 = 0.025
    α::Float64 = 0.36
    k_grid::Vector{Float64}=collect(range(0.1, length = 1000, stop = 45.0))       
    N_k::Int64 = length(k_grid)
    z_grid::Vector{Float64} = [1.25; 0.2]
    N_z::Int64 = 2
    Π::Array{Float64,2} = [0.977 0.023; 0.074 0.926]
    tol::Float64 = 10^-4

end

### Struct for our Model solutions
@with_kw struct ModelSolutions

    V::SharedArray{Float64,2}
    kp::SharedArray{Float64,2}

end 

function build_ModelSolutions(para)

    V = SharedArray{Float64}(zeros(para.N_k, para.N_z))
    kp = SharedArray{Float64}(zeros(para.N_k, para.N_z))

    sols = ModelSolutions(V,kp)

    return sols

end

function build_structs()

    para = ModelParameters()
    sols = build_ModelSolutions(para)

    return para, sols

end


### Bellman operator
function bellman(para, sols)
    @unpack_ModelParameters para
    @unpack_ModelSolutions sols

    V_next = SharedArray{Float64}(zeros(N_k,N_z))
    kp_next = SharedArray{Float64}(zeros(N_k,N_z))

    Threads.@threads for i_k in eachindex(k_grid)
        update_next!(i_k, para, V, V_next, kp_next)
    end

    return V_next, kp_next

end

@everywhere function update_next!(i_k, para, V, V_next, kp_next)
    @unpack_ModelParameters para

    for i_z = eachindex(z_grid)
        
        max_util = -1e10
        k = k_grid[i_k]
        z = z_grid[i_z]
        budget = z*k^α + (1-δ)*k

        for i_kp = eachindex(k_grid)
            
             c = budget - k_grid[i_kp]

            if c > 0
                
                V_temp = log(c) + β*(Π[i_z,1]*V[i_kp,1] + Π[i_z,2]*V[i_kp,2])
            
                if V_temp > max_util
                    max_util = V_temp
                    kp_next[i_k,i_z] = k_grid[i_kp]
                end

            end

        end

        V_next[i_k,i_z] = max_util

    end

end



### Solve model
function solve_model(para, sols)    
    @unpack_ModelParameters para
    @unpack_ModelSolutions sols

    V_next = zeros(N_k,N_z)
    kp_next = zeros(N_k,N_z)
    max_diff = tol + 10.0
    n = 0

    while max_diff > tol
        n +=1
        V_next, kp_next = bellman(para,sols)
        
        max_diff = maximum(abs.(V_next - V))
        V .= V_next
        kp .= kp_next

        @show n, max_diff

    end

end

para, sols = build_structs();

@elapsed solve_model(para,sols)

using Plots
plot(para.k_grid, sols.V)
plot(para.k_grid, sols.kp)
plot!(collect(0:45), collect(0:45))



