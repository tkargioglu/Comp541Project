
"""
glimpse network
gets a glimpse from image (1) and the location of the glimpse (2)
outputs a feature vector to feed the lstm
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

# out size of the glimpseNet equal to the size of hidden state of an RNN block
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



###################################################################################################
# util functions to get glimpse from images as described in minh et al 2014
###################################################################################################
"""
returns glimpse consisting of concatenated patches with multiple scales with given size at the given location
"""
function take_glimpse(image, numscales=1, sizepatch=8, location=(0,0), size_to_unit_length=1)
    println("takeglimpse")

    sizeimage = size(image)

    row, col = get_pixel_coordinates(location, sizeimage, size_to_unit_length)

    result = get_downsampled(take_patch(image, sizepatch*1, (row, col)), 1)

    for i=2:numscales
        result = [result get_downsampled(take_patch(image, sizepatch*i, (row, col)), i)]
    end

    result
end

"""
returns a patch from image at given coordinates with given sizes
"""
function take_patch(image, sizepatch, pixelcoordinates)

    s = size(image)
    lefthalf = Int(round(sizepatch / 2))
    righthalf = sizepatch - lefthalf - 1

    r1, c1 = pixelcoordinates .- lefthalf
    r2, c2 = pixelcoordinates .+ righthalf

    println(r1, r2, c1, c2)
    paddings = [max(0,-r1+1), max(0, c2-s[2]), max(0, r2-s[1]), max(0, -c1+1)]
    r1, r2 = [r1, r2] .+ paddings[1]
    c1, c2 = [c1, c2] .+ paddings[4]
    
    println(r1, r2, c1, c2)
    image = pad(image, paddings)

    println(size(image))
    image[r1:r2, c1:c2]    
end

"""
returns the downsampled version of the given image
scale is the downsampling rate, must be a positive integer
"""
function get_downsampled(image, scale)
    if scale == 1
        println("scale is 1")
        println(size(image))
        return image
    end
    println("this is not right")
    s = size(image)
    if length(s) == 2
        return image[1:scale:s[1], 1:scale:s[2]]
    else
        println("ERROR: image dimension is not 2 but: ", length(s))
    end
end

"""
returns the padded version of the 2 dim input image
paddings are top right bottom left paddings respectively
padwith can be used to select the number to pad with 
"""
function pad(image, paddings, padwith::Int=0)

    a, b, c, d = paddings

    println("a b c d ", a, " ", b, " ", c, " ", d)
    if a > 0
        s = size(image)
        image = [ones(Int, a, s[2]) .* padwith ; image]
    end

    if b > 0
        s = size(image)
        image = [image ones(Int, s[1], b) .* padwith]
    end
    
    if c > 0
        s = size(image)
        image = [image; ones(Int, c, s[2]) .* padwith]
    end

    if d > 0
        s = size(image)
        image = [ones(Int, s[1], d) .* padwith image]
    end

    image
end

"""
convert cartesian x and y to pixel indices (row, col) in 2 dimensional array
top-left pixel is (1,1) 
"""
function get_pixel_coordinates(location, image_size, size_to_unit_length::Float16=1)
    x, y = location

    if abs(x) > size_to_unit_length || abs(y) > size_to_unit_length
        println("ERROR: location is out of dimensions")
        return
    end 
    
    half_x, half_y = floor.(image_size./2)
    # center of the image as pixel indices top-left corner being (1, 1)
    center_x, center_y = ceil.(image_size./2)

    col = Int(center_x + round(half_x * x / size_to_unit_length))
    row = Int(center_y - round(half_y * y / size_to_unit_length))

    if col <= 0
        col = 1
    end

    if row <= 0
        row = 1
    end

    row, col
end

# glimpse test
import Pkg
Pkg.add("MLDatasets") 
using MLDatasets: MNIST
train_x, train_y = MNIST.traindata()
test_x,  test_y  = MNIST.testdata()

println(size(train_x))
println(size(train_x[:, :, 1]))
takeglimpse(train_x[:, :, 1])


"""
# test
x = (convert(atype, (reshape(ones(496, 496), (496, 496, 1, 1)))))
l = convert(atype, (reshape(ones(2), (2, 1, 1, 1))))
println(summary(glimpse(x, l)))
"""


end