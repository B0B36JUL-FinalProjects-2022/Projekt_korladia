using NPZ
using Statistics
export compute_measurements, create_work_set, crossval, flip_lbls, add_padding, prep_data
export MeasurementType, LeftRight, TopBottom

abstract type MeasurementType end

struct LeftRight <: MeasurementType end
struct TopBottom <: MeasurementType end

first_half(::LeftRight, images::Array) = sum(sum(images, dims=1)[:, 1:Int(size(images)[2] ÷ 2), :], dims=2)
first_half(::TopBottom, images::Array) = sum(sum(images, dims=2)[1:Int(size(images)[1] ÷ 2), :, :], dims=1)
second_half(::LeftRight, images::Array) = sum(sum(images, dims=1)[:, Int(size(images)[2] ÷ 2) + 1:end, :], dims=2)
second_half(::TopBottom, images::Array) = sum(sum(images, dims=2)[Int((size(images)[1] ÷ 2)) + 1 : end, :, :], dims=1)

# Compute features
function compute_measurements(images::Array; meas_type::MeasurementType = LeftRight())
    one_half = first_half(meas_type, images)
    other_half = second_half(meas_type, images)    
    x = vec(one_half .- other_half)    
    # Normalize
    x = (1/std(x)) .* (x .- mean(x))
    return x
end

"""
    create_work_set(alphabet::Array, images::Array, labels::Vector; letters::String = "CN")

From the dataset with `images` containing all letters from `alphabet` only choose the images of the `letters`
specified.
"""
function create_work_set(alphabet::Array, images::Array, labels::Vector; letters::String = "CN")
    m,n = size(images)
    letter_counts = []
    imgs = Array{UInt8}(undef, m*n, 0)
    lbls = []
    indices = []
    for i in eachindex(letters)
        letter = letters[i]        
        # Labels start from zero        
        indx = sum([ifelse(alphabet[j]==letter, j, 0) for j in 1:size(alphabet)[1]])-1   
        indices = append!(indices, indx)   
        res = images[:,:,labels .== indx]
        num = length(res)÷(m*n)
        imgs = hcat(imgs,reshape(res, (m*n, num)))
        letter_counts = append!(letter_counts, num)
        lbl = labels[labels.==indx]
        lbls = append!(lbls, ones(UInt8,length(lbl))*i)
    end
    imgs = reshape(imgs, (m,n,length(imgs)÷(m*n)))
    return imgs, lbls, letter_counts, indices
end

"""
    crossval(imgs::Array, lbls::Vector, letter_count::Vector)

The subset of `imgs` to work with is split into two parts - one is used for learning and one for testing.
"""
function crossval(imgs::Array, lbls::Vector, letter_count::Vector)
    m,n = size(imgs)
    imgs_tst, imgs_trn = Array{UInt8}(undef, m*n, 0), Array{UInt8}(undef, m*n, 0)
    lbls_tst, lbls_trn = [], []
    counts = [x÷2 for x in letter_count]
    curr_indx = 1
    for i in 1:length(letter_count)
        border = curr_indx + counts[i]
        imgs_tst = hcat(imgs_tst, reshape(imgs[:,:,curr_indx:(border-1)],(m*n, counts[i])))
        imgs_trn = hcat(imgs_trn, reshape(imgs[:,:,border:(border + counts[i] - 1)],(m*n, counts[i])))
        lbls_tst = append!(lbls_tst, lbls[curr_indx:(border-1)]) 
        lbls_trn = append!(lbls_trn, lbls[curr_indx:(border-1)]) 
        curr_indx += letter_count[i]
    end
    imgs_trn = reshape(imgs_trn, (m,n,length(imgs_trn)÷(m*n)))
    imgs_tst = reshape(imgs_tst, (m,n,length(imgs_tst)÷(m*n)))
    return imgs_trn, lbls_trn, imgs_tst, lbls_tst
end

# Change lables for binary classification
function flip_lbls(lbls::Vector)
    return [ifelse(x!=1,-1,x) for x in lbls]
end

# Add ones for the bias
add_padding(v::Vector) = append!(v,1)
function add_padding(M::Matrix; position::String = "under")
    if position == "under"
        return vcat(M, ones(Int64, size(M)[2])')
    elseif position == "over"
        return vcat(ones(Int64, size(M)[2])', M)
    else
        throw(ArgumentError("The padding can be added either under or over the matrix"))
        return
    end
end

function prep_data(imgs::Array, lbls::Vector)
    # Create image features
    method1 = LeftRight()
    method2 = TopBottom()
    x_n = compute_measurements(imgs, meas_type = method1)
    y_n = compute_measurements(imgs, meas_type = method2)
    # Create matrices of all the measurments
    X_n_1 = reshape(x_n, 1, size(x_n)[1])
    X_n_2 = vcat(x_n',y_n')
    # Prepare the data
    X_n_1 = add_padding(X_n_1)
    X_n_2 = add_padding(X_n_2)
    lbls_n = flip_lbls(lbls)
    return x_n, X_n_1, X_n_2, lbls_n
end