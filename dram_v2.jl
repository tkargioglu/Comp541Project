using Statistics
using Knet
using Random
using JLD2
using MLDatasets: MNIST

include("networks.jl")
include("imageUtil.jl")

"""
Deep Recurrent Attention Model version 2.
    Ba et al. 2015    
    Uses a different context network containing conv. layers
"""
struct DRAM_v2
    contextNet
    glimpsenet
    rnnlower::RNN
    rnnupper::RNN
    emissionnet::EmissionNet
    classificationnet::ClassificationNet
    baselinenet::Dense
    σ
end
function DRAM_v2(       patchwidth::Int             = 7,                # width (and height) of the patches
                        numscales::Int              = 2,                # number of scales in glimpses
                        glimpsefeaturelength::Int   = 256,              # length of the glimpse feature vector produced by the glimpseNet
                        rnnhiddensize::Int          = 512,              # length of the hidden state vector of the lstms
                        otherhiddensize::Int        = 256,              # number of hidden layers in cnn and dense layers
                        downsamplingrate::Int       = 4,                # used in generating coarser images for context network
                        inputsize                   = [100, 100],       # size of the input images
                        numclasses::Int             = 19,               # number of output classes
                        sigma                       = 0.03)             # location pick standard deviation

    result = DRAM_v2(   ContextNet_DRAM_v2(wsize, csize, rnnhiddensize, inputsize[1] * inputsize[2] / downsamplingrate),
                        GlimpseNet_DRAM_MNIST(patchwidth, numscales, glimpsefeaturelength),
                        RNN(glimpsefeaturelength, rnnhiddensize),
                        RNN(rnnhiddensize, rnnhiddensize),
                        EmissionNet(otherhiddensize, sigma),
                        ClassificationNet(otherhiddensize, numclasses),
                        Dense(otherhiddensize, 1, identity),
                        sigma)

    result
end
(dram::DRAM_v2)(x)    = dramforward(dram, x)


"""
run for one episode
"""
function dramforward(dram::DRAM_v2, images)

    # if there is no channel dimension make them single channel
    s = size(images)
    if length(s) == 3
        images = reshape(images, s[1], s[2], 1, s[3])
    end

    batchsize = size(images)[4]

    loc = dram.contextNet(get_downsampled(images, downsamplingrate))

    # initializing hidden state of the rnn
    dram.rnnupper.h = Param(convert(atype, randn(rnn_hidden_size, batchsize, 1)))
    dram.rnnlower.h = Param(convert(atype, zeros(rnn_hidden_size, batchsize, 1)))

    dram.rnnupper.c = Param(convert(atype, zeros(rnn_hidden_size, batchsize, 1)))
    dram.rnnlower.c = Param(convert(atype, zeros(rnn_hidden_size, batchsize, 1)))

    logπs, baselines, locations = [], [], Any[loc]

    r1 = Any
    # loop over glimpses
    for i=1:num_glimpses

        loc4 = reshape(loc, 2, 1, 1, batchsize)

        # get glimpses for the batch as a batch (patchsize x patchsize x numscale x batchsize)
        g = take_glimpses(images, loc, patchwidth, numscales, unit_width_as_pixels)
        
        # get glimpse feature vector (glimpselength)
        glimpsefeature = dram.glimpsenet(g, loc4)

        # hidden state of the lstm (hiddensize)
        r1 = dram.rnnlower(glimpsefeature)
        r2 = dram.rnnupper(r1)

        # new location
        μ, loc = dram.emissionnet(r2)

        # baseline for this timestep
        baseline = dram.baselinenet(value(r1))
        
        σ = dram.σ
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
    scores = dram.classificationnet(r1)

    return scores, baseline, logπ, locations
end
