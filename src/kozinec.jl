using Statistics
export Algorithm, Perceptron, Kozinec
export kozinec, get_mean, multi_w, multivar

abstract type Algorithm end

struct Perceptron <: Algorithm end
struct Kozinec <: Algorithm end

function update(::Kozinec, alpha::Vector, x_t::Vector)
    k = get_k(alpha, x_t)            
    alpha = (1-k)*alpha + k*x_t  
    return alpha
end
update(::Perceptron, v::Vector, x_t::Vector) = v + x_t

function get_k(a::Vector, x::Vector)
    res = -sum(a.*(x-a))/sum((x-a).^2)
    return res
end

"""
    kozinec(X::Matrix, y::Vector; alg::Algorithm = Kozinec(), max_iter::Int = 2000)

Implements the Kozinec algorithm which is a variation of the perceptron algorithm. Produces a vector
of values which form the dividing hyperplane. Depending on the value of `alg` either the Kozinec
variation of the algorithm is used or the original perceptron.
"""
function kozinec(X::Matrix, y::Vector; alg::Algorithm = Kozinec(), max_iter::Int = 2000)
    X_my = copy(X')
    iter = copy(max_iter)
    # Get indices where yi = 1
    indices = findall(a->a==-1,y)       
    for i in indices
        X_my[i, :] *= -1        
    end
    alpha = X_my[1, :]
    while iter > 0
        okays = 0       
        dots = sum(alpha' .* X_my, dims = 2)        
        # Find the first wrongly classified vector - the dot product is negative
        idx = findfirst(a->a <= 0, dots)  
        if typeof(idx) != Nothing
            idx = idx[1]
            alpha = update(alg, alpha,X_my[idx, :])           
        else
            okays += 1
        end        
        # If you can't find a faulty one
        if okays == 1
            break
        end
        iter -= 1
    end
    if iter == 0
        throw(ErrorException("Not enough iterations. The data might be inseparable."))
        return
    end
    return alpha
end

get_mean(X::Matrix) = mean(X, dims=2)

function multi_w(X::Matrix, letter_counts::Vector)
    lc = letter_counts.÷2    
    curr = 1
    X1 = X[:, curr : lc[1]]
    curr += lc[1]
    X2 = X[:, curr : curr+lc[2]-1]
    curr += lc[2]
    X3 = X[:,curr : curr+lc[3]-1]
    m1 = mean(X1, dims=2)
    m2 = mean(X2, dims=2)
    m3 = mean(X3, dims=2)
    w = hcat(m1,m2,m3)
    return w, X1, X2, X3
end

"""
    multivar(X::Matrix, lbls::Vector, w_init::Matrix, class_num::Int; max_iter::Int=2000)

Implements the perceptron for multivariate classification - it can handle more than two classes. `w_init`
is an initial vector of weights calculated with means and can be obtained as a result of the `multi_w` function.
"""

function multivar(X::Matrix, lbls::Vector, w_init::Matrix, class_num::Int; max_iter::Int=2000)
    iter = copy(max_iter)
    w = copy(w_init)
    b = zeros(class_num)
    while iter > 0  
        iter -= 1       
        ind = findmax(w'*X .+ b,dims=1)[2]
        indc = [el[1] for el in ind]
        y_hat, y_t = 0,0
        x_t = []
        for i in 1:size(lbls)[1]
            y_hat_c = indc[i]
            y_t_c = lbls[i]
            if y_hat_c != y_t_c
                y_hat = y_hat_c
                y_t = floor(Int,y_t_c)
                x_t =  X[:, i]
                break
            end
        end
    
        # Couldn't find a x_t
        if y_hat==y_t
            break
        end
        w[:, y_t+1] .+= x_t
        b[y_t+1] += 1
        w[:,y_hat] .-= x_t
        b[y_hat] -= 1
    end
    return w,b
end