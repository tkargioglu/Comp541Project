
"""
glimpse network
gets a glimpse from image and the location of the glimpse 
outputs a feature vector
"""
module glimpseNet
using Base.Iterators: flatten
using IterTools: ncycle, takenth
using Statistics: mean
using MLDatasets: MNIST
import CUDA # functional
import Knet # load, save
using Knet: conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, minibatch, Data

# Random.seed!
atype = KnetArray{Float32}

##########################################################
# hyperparams for the convnet

# matrix for filter sizes of the layers (there are 3 convolution layers)
wsize = Int.(5 * ones(3, 2))
csize = Int.(1 * ones(3, 1))

# out size of the glimpseNet
outsize = 100;

##########################################################

# convolution layer (without pooling)
struct Conv; w; f; p; end
(c::Conv)(x) = (println("start conv layer"); println(size(c.w)); o = c.f.(conv4(c.w, dropout(x, c.p))); println(size(o)); o)
Conv(w1::Int, w2::Int, cx::Int, cy::Int, f=relu; pdrop=0) = Conv(param(w1,w2,cx,cy), f, pdrop)

# dense layer
struct Dense; w; b; f; p; end
(d::Dense)(x) = (println(summary(d.w)); println(summary(d.b));  d.f.(d.w * mat(dropout(x,d.p)) .+ d.b)) # mat reshapes 4-D tensor to 2-D matrix so we can use matmul
Dense(i::Int, o::Int, f=relu; pdrop=0) = Dense(param(o,i), param0(o), f, pdrop)

# chain of layers
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)
(c::Chain)(d::Data) = mean(c(x,y) for (x,y) in d)

##########################################################

# image network 3 conv layers 1 dense layer
layer1 = Conv(wsize[1,1], wsize[1,2], 1, csize[1])
layer2 = Conv(wsize[2,1], wsize[2,2], csize[1], csize[2])
layer3 = Conv(wsize[3,1], wsize[3,2], csize[2], csize[3])
layer4 = Dense(484*484, outsize)

G_image = Chain(layer1, layer2, layer3, layer4)

# location network takes two scalars as coordinates
G_loc = Dense(2, outsize)

glimpse(x, l) = G_image(x) .* G_loc(l)

"""
# test
x = (convert(atype, (reshape(ones(496, 496), (496, 496, 1, 1)))))
l = convert(atype, (reshape(ones(2), (2, 1, 1, 1))))
println(summary(glimpse(x, l)))
"""

end