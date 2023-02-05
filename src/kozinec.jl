export kozinec, get_k, perceptron

function get_k(a::Vector, x::Vector)
    res = -sum(a.*(x-a))/sum((x-a).^2)
    return res
end

function kozinec(X::Matrix, y::Vector; max_iter::Int = 2000)
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
            k = get_k(alpha, X_my[idx, :])            
            alpha = (1-k)*alpha + k*X_my[idx, :]                     
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
        alpha = Nothing
    # else
    #     w = alpha[1:end-1]
    #     b = alpha[end]
    end
    return alpha
end

function perceptron(X::Matrix, y::Vector; max_iter::Int = 400)
    X_my = copy(X')
    iter = max_iter
    # Get indices where yi = 1
    indices = findall(a->a==-1,y)       
    for i in indices
        X_my[i, :] *= -1        
    end
    v = zeros(size(X_my)[2])       
    while iter > 0
        okays = 0
        dots = sum(v' .* X_my, dims = 2)        
        # Find the first wrongly classified vector - the dot product is negative
        idx = findfirst(a->a <= 0, dots)            
        if typeof(idx) != Nothing
            idx = idx[1]
            v += X_my[idx, :]                    
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
        v = Nothing
    # else
    #     w = alpha[1:end-1]
    #     b = alpha[end]
    end
    return v
end

