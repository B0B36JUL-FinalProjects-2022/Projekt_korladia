using LinearAlgebra
export logistic_loss, logistic_loss_gradient, logistic_loss_gradient_descent
export dimension_lifting, lift, posteriori, Lifting, Arctan, SinCos

# Logistic regression loss
function logistic_loss(X::Matrix, y::Vector, w::Vector)
    n = size(X)[2]
    y = reshape(y,1,size(X)[2])
    pwr = -y .* collect(w'*X) 
    summ = sum(log.(1 .+ exp.(pwr)))
    result = (1/n) * summ
    return result
end

# Calculates gradient of the logistic loss function
function logistic_loss_gradient(X::Matrix, y::Vector, w::Vector)
    m, n = size(X)
    h = y' .* X
    y = reshape(y,1,size(X)[2])
    pwr = y .* collect(w'*X) 
    summ = sum((1 ./ (1 .+ exp.(pwr))) .* h, dims = 2)
    g = -summ./n
    return g
end

"""
    logistic_loss_gradient_descent(X::Matrix, y::Vector, w_init::Vector; eps::Real = 1e-2, step_size::Float64 = 1.0, max_iter::Int = 300)

Performs gradient descent optimization of the logistic loss function - the logistic regression problem is 
thus solved as an optimization one where the goal is to minimize the loss. The starting point is `w_init`.
"""
function logistic_loss_gradient_descent(X::Matrix, y::Vector, w_init::Vector; eps::Real = 1e-2, step_size::Float64 = 1.0, max_iter::Int = 300)
    n = length(w_init)
    iter = copy(max_iter)
    step = step_size
    w = w_init
    wt = [zeros(n)]
    wt = append!(wt, [w])
    Et = [logistic_loss(X,y,w)]
    g = logistic_loss_gradient(X, y, w)
    while norm(wt[end] - wt[end-1]) > eps && iter > 0
        w_new = vec((w - step * g))               
        E_new = logistic_loss(X, y, w_new)
        g_new = logistic_loss_gradient(X, y, w_new)
        if E_new < Et[end]
            Et = append!(Et, [E_new])
            w = w_new
            wt = append!(wt, [w])
            g = g_new
            step *= 2
        else
            step /= 2
        end
        iter -= 1
    end
    return w, wt', Et
end

abstract type Lifting end

struct Arctan <: Lifting end
struct SinCos <: Lifting end

function lift(::Arctan, a::Float64, a_ones::Vector, a_zeros::Vector)
    res = atan.(a.*a_ones + (a_zeros.-15))
    # res = atan.(a.*a_ones + a_zeros)
    res = append!(res,1)
    return res
end

function lift(::SinCos, a::Float64, a_ones::Vector, a_zeros::Vector)
    els = [sin(a/2), cos(a), sin(a)^2 - cos(a)]
    res = a_ones.*els + a_zeros
    res = append!(res,1)
    return res
end

"""
    dimension_lifting(x::Vector; dims::Int = 3, lifter = SinCos())

Inseparable data can be oftentimes helped by dimension lifting which translates the data into
a higher dimension space.
"""

function dimension_lifting(x::Vector; dims::Int = 3, lifter = SinCos())
    indices = sortperm(x)
    x_n = copy(x)[indices]
    alpha_zeros = range(1, 4*dims,step=4) |> collect
    alpha_ones = ones(dims).*(dims÷ 2)
    res = map((el)->lift(lifter, el, alpha_ones, alpha_zeros), x_n)
    return(res)
end

function get_posteriori(a::Float64)
    return 1/(1+exp(a))
end

function posteriori(v::Vector)
    return map((el)->get_posteriori(el), v)
end