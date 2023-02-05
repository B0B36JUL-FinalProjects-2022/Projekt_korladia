using Revise
using FinalProject
using Plots

# Load the input data
labels, images = get_data()

# Get all images with the label 7
r = images[:,:,labels .== 0x00]
r[:,:,1]
reverse(r[:,:,1], dims=1)
Plots.heatmap(reverse(r[:,:,1], dims=1))

# Choose two letters from the whole dataset
alphabet = ['A', 'B', 'C', 'D', 'E', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V',
'Y', 'Z']
imgs, lbls, letter_counts = create_work_set(alphabet, images, labels, letters = "CN")
letter_counts

# Split the work set into a train set and a test set (of equal size)
imgs_trn, lbls_trn, imgs_tst, lbls_tst = crossval(imgs, lbls, letter_counts)
size(imgs),size(imgs_trn),size(imgs_tst)

# Create image features
method1 = LeftRight()
method2 = TopBottom()
x_trn = compute_measurements(imgs_trn, meas_type = method1)
y_trn = compute_measurements(imgs_trn, meas_type = method2)
x_tst = compute_measurements(imgs_tst, meas_type = method1)
y_tst = compute_measurements(imgs_tst, meas_type = method2)

# Create matrices of all the measurments
X_trn_2 = vcat(x_trn',y_trn')
X_trn_1 = reshape(x_trn, 1, size(x_trn)[1])
X_tst_2 = vcat(x_tst',y_tst')
X_tst_1 = reshape(x_tst, 1, size(x_tst)[1])

## Logistic regression

# Prepare the data
X_trn_1 = add_padding(X_trn_1, position="under")
X_trn_2 = add_padding(X_trn_2)
lbls_trn = flip_lbls(lbls_trn)
X_tst_1 = add_padding(X_tst_1, position="under")
X_tst_2 = add_padding(X_tst_2)
lbls_tst = flip_lbls(lbls_tst)

# Compute logistic loss and its gradient
X = [1 1 1 ; 1 2 3]
l = [1, -1, -1]
w = [1.5, -0.5]
logistic_loss(X,l,w)
logistic_loss_gradient(X,l,w)

# Find w with gradient descent
w_init = [-7.0,2.0,-8.0]
res = logistic_loss(X_trn_2,lbls_trn,w_init)
w,wt,Et = logistic_loss_gradient_descent(X_trn_2,lbls_trn,w_init)

# Gradient descent progress
xs = 1:size(Et)[1]
plot(xs,Et)

# Classify the images
res_lbl = classification(X_tst_2, w, "logreg")

# Compute the classification error
err = compute_error(res_lbl, lbls_tst)

## Kozinec

# Test on arbitrary data
X = [3 4 5 -1 -1.5 -3 ; 1 0 1 -2 -1 -1.5]
y = [1,1,1,0,0,0]
X_n = add_padding(X)
y_n = flip_lbls(y)
alpha = kozinec(X_n,y_n)
v = perceptron(X_n, y_n)

separ(x::Real, alpha) = (-alpha[3]-alpha[1]*x)/alpha[2]
xlims = extrema(X[1,:]) .+ [-0.1, 0.1]
ylims = extrema(X[2,:]) .+ [-0.1, 0.1]
plot(X[1,:], X[2,:],seriestype=:scatter)
plot!(xlims, x -> separ(x,alpha); label = "Separation", line = (:black,3))

# Train and test on the images
alpha = kozinec(X_trn_1, lbls_trn)
x1 = x_trn[1:15]
x2 = x_trn[16:end]
plot(x1,zeros(15),seriestype=:scatter, color=:blue)
plot!(x2,zeros(15),seriestype=:scatter, color=:yellow)
res_lbl2 = classification(X_tst_1, alpha, "kozinec")
err2 = compute_error(res_lbl2, lbls_tst)
err3 = compute_error(classification(X_trn_1, alpha, "kozinec"), lbls_trn)

v = perceptron(X_trn_1, lbls_trn)
res_lbl2 = classification(X_tst_1, v, "kozinec")
println(lbls_tst)
println(res_lbl2)
err2 = compute_error(res_lbl2, lbls_tst)

X1 = [1 1.5 2 ; 1 0 1]
lbls1 = ones(size(X1)[2])
X2 = [-3 -2 ; 0 -1]
lbls2 = ones(size(X2)[2])*2
X3 = [-2 -1.5 ; 4 5.5]
lbls3 = ones(size(X3)[2])*3
X = hcat(X1,X2,X3)
y = hcat(lbls1', lbls2', lbls3')