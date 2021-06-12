include("layers.jl")

"""
EmissionNet calculates the next location given the current state
    consists of an fc layer
    activation function is identity (may be tanh too?)
"""
struct EmissionNet
    d::Dense
    σ
end
(e::EmissionNet)(x) = (μ = e.d(x); loc = randn()*e.σ.+μ; return μ, loc)
EmissionNet(i::Int, sigma=0.03) = EmissionNet(Dense(i, 2, identity), sigma)

"""
ClassificationNet returns the softmax output for the labels
    consist of an fc layer and a softmax output layer
    hidden size is determined by the kind of experiment
"""
struct ClassificationNet
    d::Dense
    o::Dense
end
(c::ClassificationNet)(x) = c.o(c.d(x))
ClassificationNet(i::Int, o::Int) = ClassificationNet(Dense(i, hidden_size), Dense(hidden_size, o))

"""
GlimpseNet used in DRAM model
    ba et al 2015
"""
struct GlimpseNet_DRAM_SVHN
    imagechain::Chain
    locationlayer::Dense

    GlimpseNet_DRAM_SVHN(weightsize, channelsize, outsize, patchsize) = (
    # calculate image dense layer size
    d = (patchsize .- sum(weightsize, dims=1) .+3);
    densein = d[1] * d[2] * channelsize[4];
    
    new(Chain(
        Conv(weightsize[1,1], weightsize[1,2], channelsize[1], channelsize[2]),
        Conv(weightsize[2,1], weightsize[2,2], channelsize[2], channelsize[3]),
        Conv(weightsize[3,1], weightsize[3,2], channelsize[3], channelsize[4]),
        Dense(densein, outsize)),    # end of chain (for image)
        Dense(2, outsize))           # for location
    )
   
end
(g::GlimpseNet_DRAM_SVHN)(x, l) = g.imagechain(x) .* g.locationlayer(l)


"""
GlimpseNet used in DRAM model for MNIST classification
    ba et al 2015
"""
struct GlimpseNet_DRAM_MNIST
    imagelayer::Dense
    locationlayer::Dense

    GlimpseNet_DRAM_MNIST(patchwidth, channelsize, outsize) = (
      
        new(Dense(patchwidth*patchwidth*channelsize, outsize),      # for images
            Dense(2, outsize)                                       # for location
        )
    )
   
end
(g::GlimpseNet_DRAM_MNIST)(x, l) = g.imagelayer(x) .* g.locationlayer(l)


"""
GlimpseNet used in RAM model
    mnih et al 2014
"""
struct GlimpseNet_RAM
    denseglimpse::Dense
    denselocation::Dense
    densecommon::Dense

    GlimpseNet_RAM(patchwidth, channelsize) = (
        new(Dense(patchwidth*patchwidth*channelsize, 128),
            Dense(2, 128),
            Dense(256, 256)
        )
    )
end
(g::GlimpseNet_RAM)(x, l) = g.densecommon(cat(g.denseglimpse(x), g.denselocation(l), dims=1))

"""
Simple ContextNet composed of single dense layer
"""
struct ContextNet_DRAM_v1
    dense::Dense

    ContextNet_DRAM_v1(insize, outsize) = new(Dense(insize, outsize))
end
(c::ContextNet_DRAM_v1)(x) = c.dense(x)

"""
ContextNet used in DRAM model
    ba et al 2015
"""
struct ContextNet_DRAM_v2
    layerChain::Chain

    ContextNet_DRAM_v2(weightsize, channelsize, outsize, imagesize) = (
    # calculate image dense layer size
    d = (imagesize .- sum(weightsize, dims=1) .+3);
    densein = d[1] * d[2] * channelsize[4];
    
    new(Chain(
        Conv(weightsize[1,1], weightsize[1,2], channelsize[1], channelsize[2]),
        Conv(weightsize[2,1], weightsize[2,2], channelsize[2], channelsize[3]),
        Conv(weightsize[3,1], weightsize[3,2], channelsize[3], channelsize[4]),
        Dense(densein, outsize)))
    )
   
end
(c::ContextNet_DRAM_v2)(x, l) = c.chain(x)
