using NPZ
export get_data, classification, compute_error

# Load data
function get_data()
    labels, images = npzread("/home/dianka/Documents/School/JUL/data_33rpz_03_minimax.npz", ["images_test", "labels_test"])
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