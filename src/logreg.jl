using LinearAlgebra
export logistic_loss, logistic_loss_gradient, logistic_loss_gradient_descent
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