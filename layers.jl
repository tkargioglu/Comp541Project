using Knet: conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, minibatch, Data

# convolution layer (without pooling)
struct Conv; w; f; p; end
(c::Conv)(x) = (c.f.(conv4(c.w, dropout(x, c.p))))
Conv(w1::Int, w2::Int, cx::Int, cy::Int, f=relu; pdrop=0) = Conv(param(w1,w2,cx,cy), f, pdrop)

# dense layer
struct Dense; w; b; f; p; end
(d::Dense)(x) = (if length(size(x)) == 2 fun=identity; else fun=mat; end; d.f.(d.w * fun(dropout(x,d.p)) .+ d.b)) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int, o::Int, f=relu; pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

#=
struct ConvLayer
    units
    ConvLayer(numunits, w1::Int, w2::Int, cx::Int, cy::Int, f=relu; pdrop=0) = new(layers)
end
(cl::ConvLayer)(x) = cl.units(x)
=#

#=
# test
x = convert(KnetArray{Float32}, ones(10, 10, 1, 1))

c1 = Conv(5, 5, 1, 2)
c2 = Conv(5, 5, 1, 2)

units = [c1; c2]
println(summary(c1(x)))
println(summary(units(x)))
=#

# chain of layers
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)
