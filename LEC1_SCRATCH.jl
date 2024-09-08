β = 3;
ζ = 4; 
β + ζ

#Updating variables
x = 8 #Initial value
x = x + 1; 
x
##Alternatively
x += 1

##To dynamically subtract:
x -= 1

##multiplication
#x = 2*x
x *= 2

#Arrays: Vectors
v1 = rand(4); #vector of 4 uniformly distributed draws between 0 and 1
v2 = collect(1:2:41) #Vector with odd valued integers between 1 and 41, inclusive. "Start at 1, end at 41, has every 2 numbers between start and end

##Alternative way to define v2, but defined as a float:
v2 = collect(range(start = 1, stop = 41, length = 21))

v3 = collect(range(start = 1, stop = 10, length = 100));
print(v3)

##Create column vector with specific values:
v1 = [4.0, 0.3, 17.0, -0.2]; 
v1 #print(v1) gives the entry as typed, without showing the shape of the matrix

##Same but for row vector
v1_r = [4.0 .3 17.0 -.2];
v1_r 

v_zero = zeros(4)

##To get a 'kth' entry in a vector, enter the index number in square brackets right after the vector name
v_zero[2]

#Arrays: Matrices
## A matrix is a 2d array.

A3 = rand(4,5); #Matrix with 4 rows and 5 columns, each entry from uniform dist between 0 and 1

A3

##You can define specific matrices either column-wise or row-wise:

###Column-wise
A1 = [[1.0, 3.0] [2.0, 4.0]]

###Row-wise
A1 = [1.0 2.0;3.0 4.0]

#Operations on vectors and matrices

x = rand(4)
z = rand(4,4)

##Transpose
z'
transpose(z)
##Inverse
inv(z)
##Matrix multiplication
z*x
x' * z

##Broadcasting: a "." 'broadcasts' a function to each element in an array
z_log = log.(z)

z_sqrt = sqrt.(z)

z_log
z_sqrt

#Packages
##See the Codealong for instructions on how to install packages. 

##Load packages into memory:
using Plots, Distributions

#For Loops
#=
Suppose you want to simulate an AR1 process for 100 periods.
We'll construct the process followijg x_{t+1} = x_{t} + ϵ, ϵ ~ N(0,1)
=#

##Create empty vector, to be populated by the AR process
x_vec = zeros(10000)

x_vec[4]

for i in 2:10000
    x_vec[i] = x_vec[i-1] + rand(Normal())
end

x_vec[4]

#Plotting the ARIMA process:
plot(x_vec)


#Functions:
## You can write your own! Example: OLS

function OLS(Y,X)
    β = inv(X'*X)*(X'*Y)
    return β
end

##So, how do we estimate OLS using matrix operations?

###Create empty coefficient list
coefs = zeros(100)


###Data generation process and coefficient estimation

####Run OLS on 100 fake datasets, each of size 100
for i in 1:100
    #epsilon shocks in length 100
    ϵ_vec = rand(Normal(),100) #data gen
    #normal vector of length 100, called x
    X = rand(Normal(),100)
    #data generating process: assume Y = X + normal(0,1)
    Y = X + ϵ_vec


#Compute OLS coefficient and store in ith entry of coefs.
    coefs[i] = OLS(Y,X) #Coefficient estimation
end

histogram(coefs)




#=
In this case, we generated 100 datasets to get 100 estimates of Β.
The value of Y is the Value of X, plus some random normal shock.
The true Beta should be 1, and the relationship is linear. With ϵ
=#

#Booleans and Conditionals:
##Similar to Python

true
false

2 == 3
2 != 3
6 <= 7

#Conditional statements

x = 3

if x > 5
    print("yes")
else
    print("no")
end

##Now, how do I check for an entire vector?

x = rand(10)

#One way: Use a for loop
indicator = zeros(length(x))
for i in 1:length(x)
    if x[i] > .5
        indicator[i] = 1
    else 
        indicator[i] = 0
    end
end

#A much simpler way; Use the broadcasting method!
indicator_simple = x.>.5
##Notice that the datatype is a bit different.
indicator_simple == indicator


#"While" loops: Repeating an operation until some condition is method
x = 0

while x <100
   @show x += 1
end
@show x

#Checking for multiple conditionals:
(3>2) & (3<1)

(3>2) || (3<1)

#Multiple cases: elsif

x = -6
if x < 0 
    #code for the cases when x < 0
    print("x is negative")
elseif x > 0
    #code for when x > 0
    print("x is positive")
else #the remaining possibility is that x = 0
    #code for x=0
    print("x = 0")
end

#More complex printing: See code_along 1 for additional syntax. 