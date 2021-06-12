using Random
using Dates
using MLDatasets: MNIST
using Statistics: mean
using IterTools: ncycle, takenth
using Knet
using Knet: accuracy, zeroone, nll, sgd
using Knet: Knet, AutoGrad, dir, Data, minibatch, Param, @diff, value, params, grad, progress, progress!, KnetArray, load, save

include("networks.jl")
include("imageUtil.jl")

"""
Recurrent Attention Model. Mnih et al. 2014 
"""
struct RAM
    glimpsenet::GlimpseNet_RAM       
    rnn::RNN
    emissionnet::EmissionNet
    classificationnet::ClassificationNet
    baselinenet::Dense
    sigma
end
function RAM(   patchwidth::Int,             # width (and height) of the patches
                glimpselength::Int,          # length of the glimpse feature vector produced by the glimpseNet
                hiddensize::Int,             # length of the hidden state vector of the lstms
                csize,                       # channel sizes of conv layers in glimpse network
                sigma=0.03)
                    
    result = RAM(   GlimpseNet_RAM(patchwidth, csize[1]),
                    RNN(glimpselength, hiddensize),
                    EmissionNet(hiddensize, sigma),
                    ClassificationNet(hiddensize, 10),
                    Dense(hiddensize, 1, identity),
                    sigma)

    result
end
(ram::RAM)(x, deterministic) = ramforward(ram, x, deterministic)

"""
run ram for one episode
"""
function ramforward(ram::RAM, images, deterministic::Bool=false)

    # if there is no channel dimension make them single channel
    s = size(images)
    if length(s) == 3
        images = reshape(images, s[1], s[2], 1, s[3])
    end

    batchsize = size(images)[4]

    # locations at the first glimpse are random
    loc = convert(atype, randn(2, batchsize))
    
    # initializing hidden state of the rnn
    ram.rnn.h = Param(convert(atype, randn(rnn_hidden_size, batchsize, 1)))

    logπs, baselines, locations = [], [], Any[loc]

    r = Any
    # loop over glimpses
    for i=1:numglimpses

        loc4 = reshape(loc, 2, 1, 1, batchsize)

        # get glimpses for the batch as a batch (patchsize x patchsize x numscale x batchsize)
        g = take_glimpses(images, loc, patchwidth, numscales, unit_width_as_pixels)
        
        # get glimpse feature vector (glimpselength)
        glimpsefeature = ram.glimpsenet(g, loc4)

        # hidden state of the lstm (hiddensize)
        r = ram.rnn(glimpsefeature)

        # new location
        μ, loc = ram.emissionnet(r)

        # baseline for this timestep
        baseline = ram.baselinenet(value(r))
        
        σ = ram.sigma
        logπ = -(abs.(loc - μ) .^ 2) / 2σ*σ .- log(σ) .- log(√(2π))
        logπ = sum(logπ, dims=1)

        if deterministic
            loc = μ
        end

        push!(locations, loc)
        push!(baselines, baseline)
        push!(logπs, logπ)
    end

    baseline, logπ = vcat(baselines...), vcat(logπs...)
    
    # scores: numclasses x batchsize
    scores = ram.classificationnet(r)

    return scores, baseline, logπ, locations
end
