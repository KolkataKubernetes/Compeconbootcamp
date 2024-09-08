#NUMERIC OPTIMIZATION

## LECTURE notes

### For Univariate Box Optimization: Brent's method


### For multivariate: Newtons method if we have a Hessian. However in econ we often don't know the functional form- this is infeasible.

#= Quasi-Newtonian methods
 We can instead use Quasi-Newtonian methods: Slide 8 uses a method that is more stable than just using one side

 Algorithm: BFGS or L-BFGS give parameter guesses.

 Finite differences for a 20 parameter model can be computationally intensive using the Quasi-Newtonian method;
    One gradient is 40 calcs! (20 parameters, 2 calcs) And the Hessian will require 400 calcs (20x20)

    We saw last time that a Bellman eq will take 20 seconds to solve. So the Quasi-Newtonian methods will 
    become intractable.
 =#

 #= Derivative Free methods: When function evaluations become tedious
 NELDER MEAD:
 Constructs a simplex of 3 points, searches for new points and replaces the worst candidate in the Simplex
 The goal is for the simplex to shrink enough such that the difference between function values is sufficiently
small.

##Tons of drawbacks:
Doesn't directly use info about function behavior
Can get stuck with local minima
Convergence is generally slow
 =#

 #= Dealing with Nelder-Mead weaknesses via Randomization

    Great for if we have many local minima, or Non-smooth objective functions. We'll discuss two types:

1) Basin Hopping
        Guess an initial point and run an algorithm (such as NM)
        Hop to a new initial point and try again
    
2) Laplace-type estimator
        Randomly jump around parameter space
        "accept" the better guesses
        Size of jump updates based on fraction of "accepted" guesses

 =# 

 #= Algorithm parameter of importance: Tolerance
Most optimization algorithms have some sort of nested tolerance level.
Lower tolerance level improves precision but also increases compute time.

If you choose too high a tolerance, the predicted actions taken by your agent will not represent the "true" decisions 
under convergence.

But, too low a tolerance really slows down convergence. 

Common procedure in econ:
                1) Inner loop: solves for model given some parameter guess
                2) Outer loop: choose parameter to minimize difference between model (theoretical choices), 
                statistical observations (observed choices)

Think about 2 levels of tolerance: 
                "inner loop" tolerance: Tolerance for value function iteration
                "outer loop" tolerance: Tolerance when you try to determine whether the function itself has converged 

 =#

 #EXAMPLES: START CODEALONG

 using Plots, Optim

 ## Univariate Boxed Constrained Optimization

 f(x,y) = (x-y)^2


 ### Recall that Brent's method is the way to solve univariate boxed constrained optimization problems.
 ###the optim package will default to this method.
 opt = optimize(x->f(x,1.0), #Declare the function using optimize function in the Optim package: minimize the function give a fixed y
                -5.0, #Lower bound
                5.0)  #Upper bound

#We can use a command on the 'opt' variable to access different parts of the optimization output:

opt.minimizer
opt.minimum

#The "x->f(x)" operator defines a function inline that maps x to f(x)
# this is useful for when optimizing over a subset of parameters. 
# Here, we optimize over x for a fixed y. 

 ## Multivariate Optimization, Derivative Free EXAMPLES

 #= ROSEONBROCK: ONE MINIMUM
Instead of guessing one value, we want to solve for multiple optimal values. We'll 
use the Rosenbrock function as an example, which takes 2d vector as an argument: EXAMPLE OF GRID SEARCH
 =# 

 function Rosenbrock(x::Vector{Float64})
        val = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
        val
 end

 #Grid search: Use a grid search to find the optimal (lowest) value

 x_grid = collect(-3:0.01:3) #Create a grid with domain [-3,3], 
 nx = length(x_grid) #Define the length for loop iteration
 z_grid = zeros(nx, nx) #Function outputs go here

 for i = 1:nx, j = 1:nx #Loop over all x values, then over all y values
        guess = [x_grid[i], x_grid[j]]
        z_grid[i,j] = Rosenbrock(guess)
 end

 #Let's plot our result!

 ##Surface plot
 surface(x_grid, x_grid, z_grid, seriescolor =:viridis, camera = (50,50)) #Takes grid, function values
##Contour plot. log transform
 contourf(x_grid, x_grid, log.(1 .+ z_grid))
 scatter!([1],[1], color = "white", label = "minimizer")
 #= What does [1],[1] do?
Notice we ran a grid search on the Rosenbrock problem to find a minimum.

Let's use the other techniques we've learned to solve the same problem.
 =#

 ##ROSENBROCK NELDER-MEAD

 ### Recall Nelder-Mead is the default choice for multivariate optimizations in Julia.

 ### Do we get the same result?
 guess = [0.0,0.0]

 opt = optimize(Rosenbrock, guess)

 opt.minimizer
 opt.minimum

 #= 
We had to specify an initial vector guess. Unlike the grid, we're actually guessing
vectors of coordinates instead of components of a grid.

At the multivariate level, you HAVE to specify a vector setup, not points! (Why)
 =#

 ##ROSENBROCK: LBGFS(): Quasi-Newtonian!

 opt = optimize(Rosenbrock, [-5.0;-5.0], LBFGS()) #initial guess -5,-5
opt.minimizer

#= 
Smaller converge then Nelder-Mead. More function/gradient calls. Minimizers much closer to 1

Here, the LBGFS method had to use finite differences to find the computational derivative.

What if we were able to specificy a closed-form derivative?

CONTINUE AT 1:05:39
=#

## ROSENBROCK: MULTIVARIATE OPTIMIZATION WITH DERIVATIVES

#Define gradient of the Rosenbrock Function w.r.t. each variable

function g(G, x::Vector{Float64})
       G[1] = -2.0*(1.0 - x[1]) -400*x[1]*(x[2]-x[1]^2) #derivative w.r.t. x 
       G[2] = 200.0*(x[2]-x[1]^2) #derivative w.r.t. y
       return G
end

#Rosenbrock's Hessian

function h(H, x::Vector{Float64})
       H[1,1] = 2 - 400.0*x[2] + 1200.0*x[1]^2 #x-x
       H[1,2] = -400.0 * x[1] #x-y
       H[2,1] = -400.0 * x[2] #y-x
       H[2,2] = 200.0 #y-y
       return H #return
end

#Notice optim defaults to Newton's method when provided with gradient, Hessian

guess = [0.0,0.0]
opt = optimize(Rosenbrock,g,h,guess) #Needed less iterations, function calls than previous 2 approaches
opt.minimizer 
opt.minimum


#= GREIWANK: WHEN A FUNCTIION HAS MANY LOCAL minima
When a function has more than 1 local minima- when do each of the algorithms fail?
=#

## Define the Greiwank function

function Greiwank(x::Array{Float64})
       val = (1/4000)*sum(x.^2) - prod(cos.(x./sqrt(length(x)))) + 1

       val
end

## Evaluate the function at a bunch of points
x_grid = collect(-5:.01:5)
nx = length(x_grid)
z_grid = zeros(nx,nx)

for i = 1:nx, j = 1:nx
       guess = [x_grid[i], x_grid[j]] #Use [] otherwise Julia treats it as a tuple!
       z_grid[i,j] = Greiwank(guess)
end

#Plot the result!

surface(x_grid,x_grid,z_grid, seriescolor = :viridis, camera = (50,70))
contourf(x_grid,x_grid,z_grid,seriescolor = :inferno)

## Using default Nelder_Mead process to propose minima

guess_init = [3.0,3.0]

opt = optimize(Greiwank,guess_init) #this fails! We have convergence, but look at the minimizer

opt.minimizer #That's not actually the true local min.
opt.minimum
##Adjusting the guess to get the right answer:

guess_init = [2.0,2.0]
r
opt = optimize(Greiwank,guess_init) #this is the correct answer: Based on change in guess

opt.minimizer  #For all essential purposes this is 0 
opt.minimum 


#= Key Nelder-Mead take away:
If you run NM for multiple guess points, it MIGHT give you the right answer.
=#

#Multiple init guesses:
x_grid = collect(-5:2.0:5)
nx = length(x+grid)
minimum,minimizers = 100,[100,100] #Pre-allocated bad guesses for min, minimizers: So we can update them later

for i = 1:nx, j = 1:nx
       guess = [x_grid[i], x_grid[j]] #Starting guesses
       opt = optimize(Greiwank,guess)

       #new starting guesses
       if opt.minimum < minimum
              minimum = opt.minimum
              minimizers = opt.minimizer
       end
end
minimum, minimizers


#################################################
#OLS Example
#################################################

using Distributions, Random

##run same OLS as first class

dist = Normal(0,1)
β_0 = 1.0
β_1 = 2.0
β_2 = 3.0
n = 10000
x = rand(n).*10
x2 = x.^2
Random.seed!(1234) ###important to remember!! This ensures that we get the same result each time we run the code
ϵ = rand(dist,n) ## idiosyncratic error shocks
Y_true = β_0 .+ β_1.*x + β_2.*x2 + 5.0 .*ϵ

#Create matrix X: Intercept estimate, coefficient estimates for x, x2
X = hcat(ones(n),x,x2)

#Create OLS estimator
β_ols = inv(X' * X)*(X'*Y_true)

#We obtained results that approximate the true parameter values, but what if we used numerical optimization?

#= USING NELDER MEAD
The goal is to minimize the squared error
=#

###Define 'bad' squared error function
function sq_error(β::Array{Float64})
       β_0, β_1, β_2 = β[1], β[2], β[3] #Unpack β_ols
       Random.seed!(1234) #If this isn't in the function, you will draw a new epsilon every time!
       ϵ = rand(dist,n) #draw epsilons: Seed makes this the same
       Y_true = 1.0 .+ 2.0.*x + 3.0.*x2 + 5.0 .*ϵ
       Y_predict = β_0 .+ β_1.*x + β_2.*x2 
       error = sum((Y_true.-Y_predict).^2) #sum squared error
       error #return
end


###Use Nelder-Mead

guess_init = [0.0, 0.0, 0.0]
opt = optimize(sq_error, guess_init)
β = opt.minimizer

##basically the same values

Y_guess = β[1] .+ β[2].*x + β[3].*x2
scatter([Y_true,Y_guess],x)

