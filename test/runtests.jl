using FinalProject
using Test

@testset "FinalProject.jl" begin
    # Write your tests here.
end

@testset "kozinec.jl" begin
    X1 = [3 4 5 -1 -1.5 -3 ; 1 0 1 -2 -1 -1.5; 1 1 1 1 1 1]
    y1 = [1,1,1,-1,-1,-1]
    X2 = [1 3 4 2 5; 1 3 4 2 5; 1 1 1 1 1]
    y2 = [-1, -1, -1, 1,1]
    X3 = [1 0 2; 0 1 1; 1 1 1]
    y3 = [-1, -1, 1]
    res_1 = [3.0, 1.0, 1.0]
    res_2 = Nothing
    res_3 = [1.0, 1.0, -2.0]
    @test kozinec(X1,y1) == res_1
    @test kozinec(X2,y2) == res_2
    @test kozinec(X1,y1,alg=Perceptron()) == res_1
    @test kozinec(X2,y2,alg=Perceptron()) == res_2
    @test kozinec(X3,y3,alg=Perceptron()) == res_3
     
end

@testset "logreg.jl" begin
    X = [1 1 1 ; 1 2 3]
    l = [1, -1, -1]
    w = [1.5, -0.5]
    loss = 0.6601619507527583
    loss_g = reshape([0.28450596994395316, 0.8253257470112381],2,1)
    @test logistic_loss(X,l,w) == loss
    @test logistic_loss_gradient(X,l,w) == loss_g
end

@testset "data_prep.jl" begin
    imgs = [1 2 3; 4 5 6;;; 7 8 9; 0 0 0 ;;; 7 7 7; 1 1 1 ;;; 5 5 5; 4 4 4]
    lbls = [1, 1, 2, 2]
    letter_count = [2, 2]
    imgs1 = [7 8 9; 0 0 0;;; 5 5 5; 4 4 4]   
    imgs2 = [1 2 3; 4 5 6;;; 7 7 7; 1 1 1]
    lbls_n = [1, 2]     
    lbls_flp = [1,1,-1,-1]
    img3 = [1 2 3; 4 5 6]
    img3_u = [1 2 3; 4 5 6; 1 1 1]
    img3_o = [1 1 1; 1 2 3; 4 5 6]
    @test crossval(imgs, lbls, letter_count) == (imgs1,lbls_n,imgs2,lbls_n)
    @test flip_lbls(lbls) == lbls_flp
    @test add_padding(img3) == img3_u
    @test add_padding(img3, position = "over") == img3_o
end
