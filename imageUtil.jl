using Knet
###################################################################################################
# util functions to get glimpse from images as described in minh et al 2014
###################################################################################################

"""
returns glimpses as 4 dim tensor of size (sizepatchxsizepatchxnumscales*channelsizexbatchsize)
    images      4 dimensional (heightxwidthxchannelsizexbatchsize)
    locations   2 dimensional (2xbatchsize)
"""
function take_glimpses(images, locations, sizepatch=8, numscales=1, unit_width_as_pixels=12, atype=KnetArray{Float32})
    
    sizeimage = size(images)[1:2]

    indices = get_pixel_indices(locations, sizeimage, unit_width_as_pixels)

    result = convert(atype, get_downsampled(take_patch(images, sizepatch, indices), 1))

    for i=2:numscales
        result = cat(result, get_downsampled(take_patch(images, sizepatch*i, indices), i), dims=3)
    end

    result
end

#=
# tests
using Knet
using MLDatasets: MNIST
train_x, train_y = convert.(Array{Float64}, MNIST.traindata());
batchsize = 128;
batches = minibatch(train_x, train_y, batchsize);
images = iterate(batches)[1][1];
s = size(images)
images = reshape(images, s[1], s[2], 1, s[3]);
locations = rand(2, batchsize);
glimpses = take_glimpses(images, locations, 8, 2);
println(summary(glimpses))
using ImageView
imshow(convert(Array{Float32}, glimpses[:, :, 1, 44]))
imshow(convert(Array{Float32}, glimpses[:, :, 2, 44]))
imshow(convert(Array{Float32}, images[:,:, 1, 56]))
=#

"""
returns batch of patches (4 dimensional) from 4 dimensional images at given coordinates with given sizes
"""
function take_patch(images, sizepatch, pixelindices)

    channelsize     = size(images)[3]
    batchsize       = size(images)[4]
    result          = convert(KnetArray{Float32}, zeros(sizepatch, sizepatch, channelsize, batchsize))

    s = size(images)[1:2]
    
    lefthalf = Int(round(sizepatch / 2))
    righthalf = sizepatch - lefthalf - 1
        
    for i=1:batchsize
        image   = images[:, :, :, i]
        indices = pixelindices[:, i]

        r1, c1 = indices .- lefthalf
        r2, c2 = indices .+ righthalf

        paddings = [max(0,-r1+1), max(0, c2-s[2]), max(0, r2-s[1]), max(0, -c1+1)]
        r1, r2 = [r1, r2] .+ paddings[1]
        c1, c2 = [c1, c2] .+ paddings[4]
        
        image = pad(image, paddings)

        result[:, :, :, i] = image[r1:r2, c1:c2, :]
    end
    result
end

"""
returns the downsampled version of the given image
    scale is the downsampling rate, must be a positive integer
"""
function get_downsampled(images, scale)

    if scale == 1
        return images
    end

    s = size(images)
    return images[1:scale:s[1], 1:scale:s[2], :, :]
end

#=
# test
images = ones(8, 8, 1, 128);
i = get_downsampled(images, 100);
println(summary(i))
=#

"""
returns the padded version of the 2 dim input image
    paddings are top right bottom left paddings respectively
    padwith can be used to select the number to pad with 
"""
function pad(image, paddings, padwith::Int=0; atype=KnetArray{Float32})

    a, b, c, d = paddings

    if a + b + c + d == 0
        return image
    end

    # image = copy(image)
    
    # make the image 3 dimensional if it is 2
    if length(size(image)) == 2
        image = reshape(image, size(image)..., 1)
    end

    if a > 0
        s = size(image)
        image = cat(convert(atype, ones(Int, a, s[2], s[3])) .* padwith, image, dims=1)
    end

    if b > 0
        s = size(image)
        image = cat(image, convert(atype, ones(Int, s[1], b, s[3])) .* padwith, dims=2)
    end
    
    if c > 0
        s = size(image)
        image = cat(image, convert(atype, ones(Int, c, s[2])) .* padwith, dims=1)
    end

    if d > 0
        s = size(image)
        image = cat(convert(atype, ones(Int, s[1], d)) .* padwith, image, dims=2)
    end

    image
end

#=
# test
image = convert(atype, ones(8, 8));
i = pad(image, [1, 2, 3, 4])
=#

"""
convert cartesian x and y to pixel indices (row, col)
    top-left corner is -1,-1 and the center of the image is 0,0
    loc is 2 dimensional (2xbatchsize)
    returns row;col as 2xbatchsize array
"""
function get_pixel_indices(loc, image_size, unit_width_as_pixels=12; foroffsize="modulus")

    image_size  = vcat(image_size...)

    # half of the image as pixel indices
    half = floor.(image_size./2)
    # center of the image as pixel indices
    center = (ceil.(image_size./2))

    # deviation from center can be half at maximum
    # loc = copy(loc)
    loc = convert(Array{Float32}, loc)
    # 2 distinct approaches to solve off locations below: 1 modulus, 2 clamp 

    if foroffsize == "clamp"
        deviation_from_center = clamp.(round.(loc .* unit_width_as_pixels), -half, half)
    elseif foroffsize == "modulus"
        deviation_from_center = round.(loc .* unit_width_as_pixels) .% half
    else
        println("Error: unknown approach in get_pixel_indices function.")
    end

    indices = floor.(center .+ deviation_from_center)

    s = size(indices)
    ind = zeros(Int, s)
    for i=1:s[1]
        for j=1:s[2]
            if isnan(indices[i, j])
                ind[i, j] = 0
            else
                ind[i, j] = Int(indices[i, j])
            end
        end
    end

    row = ind[2, :]
    col = ind[1, :]

    row .+= (row .== 0)
    col .+= (col .== 0)

    [row col]' # 2xbatchsize after transpose
end

#=
# test
indices = get_pixel_indices(ones(2, 128), [28;28]);
println(summary(indices))
println(summary(col))
=#

"""
deprecated version of the function
"""
function get_pixel_coordinates_v0(location, image_size,  size_to_unit_length::Float64=1.0)

    # location as coordinates
    l1 = location[1]
    l2 = location[2]

    # take modulus, this is a design choice, error might be thrown rather than taking modulus
    location = location .% size_to_unit_length
    
    # pixel indices
    half_x, half_y = floor.(image_size./2)
    # center of the image as pixel indices
    center_x, center_y = ceil.(image_size./2)

    col = Int(center_x + round(half_x * l1 / size_to_unit_length))
    row = Int(center_y + round(half_y * l2 / size_to_unit_length))

    if col <= 0
        col = 1
    end

    if row <= 0
        row = 1
    end

    row, col
end
#=
import Pkg
Pkg.add("MLDatasets") 
using MLDatasets: MNIST
# tests
using Images
image = load("../proj/data/svhn/train/1.png")
image = img = channelview(image)
println(summary(image))
imshow(take_glimpse(image[1,:,:], 5, 300, (0.5,-0.3)))
=#

#=
p = Param(ones(5,5)*15)

p = Param([1 2 3 4 5])
function t(p)
    maximum(p.%4)
end

T = @diff t(p)
g = grad(T, p)
=#
