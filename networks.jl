include("layers.jl")

"""
EmissionNet calculates the next location given the current state
    consists of an fc layer
"""
struct EmissionNet
    d::Dense
end
(e::EmissionNet)(x) = e.d(x)
EmissionNet(i::Int) = EmissionNet(Dense(i, 2))

"""
ClassificationNet returns the softmax output for the labels
    consist of an fc layer and a softmax output layer
"""
struct ClassificationNet
    d::Dense
end
(c::ClassificationNet)(x) = exp.(d(x))
ClassificationNet(i::Int, o::Int) = ClassificationNet(Dense(i, o))

"""
GlimpseNet
    
"""
struct GlimpseNet
    imagechain::Chain
    locationlayer::Dense

    GlimpseNet(weightsize, channelsize, outsize, patchsize) = (
    # calculate image dense layer size
    d = (patchsize - sum(weightsize, dims=1) .+3);
    densein = d[1] * d[2] * channelsize[3];
    
    new(Chain(
        Conv(weightsize[1,1], weightsize[1,2], 1, channelsize[1]),
        Conv(weightsize[2,1], weightsize[2,2], channelsize[1], channelsize[2]),
        Conv(weightsize[3,1], weightsize[3,2], channelsize[2], channelsize[3]),
        Dense(densein, outsize)),   # end of chain (for image)
        Dense(2, outsize))           # for location
    )
   
end
(g::GlimpseNet)(x, l) = g.imagechain(x) .* g.locationlayer(l)

#= function  GlimpseNet(weightsize, channelsize, outsize, patchsize)
    # calculate image dense layer size
    [d1, d2] = (patchsize - sum(weightsize, dims=1) .+3) * channelsize[3]
    densein = d1*d2
    
    return GlimpseNet(Chain(
        Conv(weightsize[1,1], weightsize[1,2], 1, channelsize[1]),
        Conv(weightsize[2,1], weightsize[2,2], channelsize[1], channelsize[2]),
        Conv(weightsize[3,1], weightsize[3,2], channelsize[2], channelsize[3]),
        Dense(densein, outsize)),   # end of chain (for image)
        Dense(2, outsize)           # for location
    )
end =#

wsize = [5 5; 3 3; 3 3]
csize = [64 64 128]
outsize = 100;
patchsize = [10 20]

glimpsenet = GlimpseNet(wsize, csize, outsize, patchsize)

for i=1:1000
    image = rand(10, 20)
    image4 = convert(atype, (reshape(image, (size(image)[1], size(image)[2], 1, 1))))

    loc = rand(Float64, 2)
    loc4 = convert(atype, (reshape(loc, (2, 1, 1, 1))))

    g = glimpsenet(image4, loc4)
end