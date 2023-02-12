export Algorithm, Perceptron, Kozinec
export kozinec

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
    hey = 0 
    while iter > 0
        okays = 0
        hey+=1
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

