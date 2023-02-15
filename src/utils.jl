using NPZ
using Plots
export Approach, Logreg, Kznc
export get_data, classification, compute_error
export create_plot, test_train_plot, show_classification

# Load data
function get_data(path::String)
    labels, images = npzread(path, ["images_test", "labels_test"])
    # We need the second element of the Pair labels
    labels = labels[2]
    # Get the images from the Pair as an array
    images = collect(images[2])
    return Int.(labels), Int.(images)
end

abstract type Approach end

struct Logreg <: Approach end
struct Kznc <: Approach end

classif(::Logreg, X::Matrix, w::Vector) = collect(w'*X) 
classif(::Kznc, X::Matrix, w::Vector) = sum(X' * w, dims = 2)

# Produce labels for each image
function classification(X::Matrix, w::Vector, approach::Approach)
    if typeof(approach) <: Approach
        class = classif(approach, X, w)
        lbls = vec([ifelse(el>0,1,-1) for el in class])
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

# Show the train and test set points separately
function test_train_plot(x_trn::Vector, x_tst::Vector)
    x1 = x_trn[1:size(x_trn)[1]÷2]
    x2 = x_trn[size(x_trn)[1]÷2+1:end]
    x3 = x_tst[1:size(x_tst)[1]÷2]
    x4 = x_tst[size(x_tst)[1]÷2+1:end]
    p1 = plot(x1,zeros(size(x_trn)[1]÷2),seriestype=:scatter, color=:pink, title="Train data", size=(900,400))
    p1 = plot(p1, x2,zeros(size(x_trn)[1]÷2),seriestype=:scatter, color=:skyblue)
    p2 = plot(x3,zeros(size(x_tst)[1]÷2),seriestype=:scatter, color=:pink, title="Test data", size=(900,400))
    p2 = plot(p2, x4,zeros(size(x_tst)[1]÷2),seriestype=:scatter, color=:skyblue)
    p = plot(p1, p2,layout=2, legend=false)
    return p
end

# Create separation line from weights vectors
separ3(x::Real, w::Vector, b::Real) = (w[1]*x+b)/w[2]
separ3(x::Real, alpha) = (-alpha[3]-alpha[1]*x)/alpha[2]
separ2(x::Real, alpha) = -alpha[2]-alpha[1]*x

# Create plot
function create_plot(x::Vector, alpha::Vector, dim_num::Int64, title::String)
    n = size(x)[1]÷2
    x1 = x[1:n]
    x2 = x[n+1:end]
    p = plot(x1,zeros(n),seriestype=:scatter, color=:pink, title=title)
    p = plot(p, x2,zeros(n),seriestype=:scatter, color=:skyblue)
    xlims = extrema(x) .+ [-0.1, 0.1]
    if dim_num == 3
        p = plot(p,xlims, x -> separ3(x,alpha); label = "Separation", line = (:purple,1))
    elseif dim_num == 2
        p = plot(p,xlims, x -> separ2(x,alpha); label = "Separation", line = (:purple,1))
    else
        throw(ArgumentError("Invalid dimension"))
    end
    return p
end

"""
    show_classification(imgs::Array, indices::Vector, new_lbls::Vector)

Shows how the classification turned out - plots the images according to their newly acquired labels.
"""
function show_classification(imgs::Array, indices::Vector, new_lbls::Vector)
    res_n = [ifelse(l==1, indices[1], indices[2]) for l in new_lbls]
    letters_classed = []
    for letter in indices
        mask = res_n .== letter
        amount = count(mask)
        images = zeros(10,10,amount)
        for ix in (1:amount)
            images = imgs[:,:,mask]
        end
        push!(letters_classed, images)
    end
    for res_imgs in letters_classed
        ps = [heatmap(reverse(res_imgs[:, :, i], dims=1), legend=:none) for i = 1:size(res_imgs)[3]]
        l = @layout[grid(6,size(res_imgs)[3]÷6+1)]
        display(plot(ps..., layout=size(res_imgs)[3], axis=false, background_color=:transparent))
    end
end
