using NPZ
using Plots
export get_data, classification, compute_error, create_plot

# Load data
function get_data(path::String)
    labels, images = npzread(path, ["images_test", "labels_test"])
    # We need the second element of the Pair labels
    labels = labels[2]
    # Get the images from the Pair as an array
    images = collect(images[2])
    return Int.(labels), Int.(images)
end

# Produce labels for each image
function classification(X::Matrix, w::Vector, approach::String)
    if approach == "logreg"
        a = collect(w'*X) 
        lbls = vec([ifelse(el>0,1,-1) for el in a])
    elseif approach == "kozinec"
        summ = sum(X' * w, dims = 2)
        lbls = vec([ifelse(el>0,1,-1) for el in summ])
    else
        throw(ArgumentError("Please enter a valid classification approach."))
        return
    end
    return lbls
end

# Compare the produced labels to the true ones
function compute_error(lbls::Vector, true_lbls::Vector)
    d = true_lbls - lbls
    diff = findall(a->a != 0, d)
    err = size(diff)[1]/size(lbls)[1]
    return err
end

# Create separation line
separ3(x::Real, alpha) = (-alpha[3]-alpha[1]*x)/alpha[2]
separ2(x::Real, alpha) = -alpha[2]-alpha[1]*x

# Create plot
function create_plot(x::Vector, alpha::Vector, dim_num::Int64, title::String)
    n = size(x)[1]÷2
    x1 = x[1:n]
    x2 = x[n+1:end]
    p = plot(x1,zeros(n),seriestype=:scatter, color=:pink, title=title)
    p = plot(p, x2,zeros(n),seriestype=:scatter, color=:purple)
    xlims = extrema(x) .+ [-0.1, 0.1]
    if dim_num == 3
        p = plot(p,xlims, x -> separ3(x,alpha); label = "Separation", line = (:black,3))
    elseif dim_num == 2
        p = plot(p,xlims, x -> separ2(x,alpha); label = "Separation", line = (:black,3))
    else
        throw(ArgumentError("Invalid dimension"))
    end
    return p
end
