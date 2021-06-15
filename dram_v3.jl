using Statistics
using Knet
using Random
using JLD2

include("networks.jl")
include("imageUtil.jl")
include("util.jl")

"""
Deep Recurrent Attention Model. Ba et al. 2015
    Multi digit classification version
"""
struct DRAM_v3
    contextNet::ContextNet_DRAM_v2
    glimpsenet::GlimpseNet_DRAM_SVHN
    rnnlower::RNN
    rnnupper::RNN
    emissionnet::EmissionNet
    classificationnet::ClassificationNet
    baselinenet::Dense
    σ
end
function DRAM_v3(       patchwidth::Int             = 7,                # width (and height) of the patches
                        numscales::Int              = 2,                # number of scales in glimpses
                        glimpsefeaturelength::Int   = 512,              # length of the glimpse feature vector produced by the glimpseNet
                        rnnhiddensize::Int          = 512,              # length of the hidden state vector of the lstms
                        otherhiddensize::Int        = 1024,              # number of hidden layers in cnn and dense layers
                        downsamplingrate::Int       = 4,              # used in generating coarser images for context network
                        inputsize                   = [64, 64],       # size of the input images
                        numclasses::Int             = 11,            # number of output classes
                        sigma                       = sqrt(0.03))             # location pick standard deviation

                    
    result = DRAM_v3(   ContextNetDRAM_v2(wsize, csize, rnnhiddensize, inputsize[1] * inputsize[2] / downsamplingrate),  
                        GlimpseNet_DRAM_SVHN(patchwidth, numscales, glimpselength),
                        RNN(glimpsefeaturelength, rnnhiddensize),
                        RNN(rnnhiddensize, rnnhiddensize),
                        EmissionNet(otherhiddensize, sigma),
                        ClassificationNet(otherhiddensize,  numclasses),
                        Dense(otherhiddensize, 1, identity),
                        sigma)

    result
end
(dram::DRAM_v3)(x)       = dramforward(dram, x)


"""
run for one episode
    return loss
"""
function dramforward(dram::DRAM_v3, images, deterministic::Bool=false)

    # if there is no channel dimension make them single channel
    s = size(images)
    if length(s) == 3
        images = reshape(images, s[1], s[2], 1, s[3])
    end

    batchsize = size(images)[4]
    
    loc = dram.contextNet(get_downsampled(images, downsamplingrate))

    # initializing hidden state of the rnn
    dram.rnnupper.h = Param(convert(atype, randn(128, batchsize, 1)))
    dram.rnnlower.h = Param(convert(atype, zeros(128, batchsize, 1)))

    dram.rnnupper.c = Param(convert(atype, zeros(128, batchsize, 1)))
    dram.rnnlower.c = Param(convert(atype, zeros(128, batchsize, 1)))

    logπs, baselines, locations = [], [], Any[loc]
    Y = []
    SCORES = []

    r1 = Any
    # loop over max number of digits plus the terminal label
    for i=1:6
        for j=1:num_glimpses_per_digit
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

        # scores: numclasses x batchsize
        scores = dram.classificationnet(r1)
        ŷ = vec(map(i->i[1], argmax(Array(value(scores)), dims=1)))
        if ŷ == terminal_label
            break
        end
        push!(Y, ŷ)
        push!(SCORES, scores)
    end

    baseline, logπ = vcat(baselines...), vcat(logπs...)

    return Y, SCORES, baseline, logπ, locations
end

function getloss(dram::DRAM_v3, x, y::Union{Array{Int64}, Array{UInt8}}, deterministic=false)

    M = 5
    losses = []
    correct = 0
    total = 0
    # monte carlo sampling
    for i =1:M
        Y, SCORES, baseline, logπ, locations = dramforward(dram, x, deterministic)

        r = getreward(Y, y)
        r = reshape(r, 1, length(r))
        R = convert(KnetArray{Float32}, r)
        R̂ = R .- value(baseline)
        loss_action = nllmultidigit(SCORES, y)
        loss_baseline = sum(abs2, baseline .- R) / length(baseline)
        loss_reinforce = mean(sum(-logπ .* R̂, dims=1))
        push!(losses, loss_action + loss_baseline + loss_reinforce)
        correct += sum(r)
        total += length(r)
    end

    loss = sum(losses) / M
    correct = correct / M
    total = total / M

    return loss, correct, total
end
loss(dram::DRAM_v3, x, ygold) = getloss(dram, x,ygold)[1]
loss(dram::DRAM_v3, d::Data) = mean(getloss(dram, x,y)[1] for (x,y) in d)


function getreward(Y, y)
    
    result = zeros(l)
    l = length(y)

    for i=1:l
        k = length(string(y[i]))
        for j = 1:k
            if string(Y[j][i]) == string(y)[j]
                result[i] += 1
            end
        end
    end

    result
end

function train()

    lrate = lr
    dram = DRAM_v3()
    
    bestmodel_path = string("./best.jld2")
    lastmodel_path = string("./last.jld2")
    history = []
    bestacc = 0.0

    loss(x, ygold) = getloss(dram, x, ygold)[1]
    for epoch = 1:maxepoch
        
        @load "./data_preprocessed/svhn.jld2"

        ytrn = convert(Array{Int64}, ytrn)
        ytst = convert(Array{Int64}, ytst)

        dtrn = minibatch(xtrn, ytrn, batchsize; xtype = atype)
        dtst = minibatch(xtst, ytst, batchsize; xtype = atype)

        progress!(nesterov(loss, dtrn, lr=lrate, gamma=0.9))
        
        trn_losses, trn_acc = validate(dram, dtrn)
        tst_losses, tst_acc = validate(dram, dtst)
        printEpochSummary(0, trn_acc, tst_acc)
        push!(history, ([trn_losses..., trn_acc, tst_losses..., tst_acc]))
        
        Knet.save(lastmodel_path, "model", dram, "history", history)

        if tst_acc > bestacc
            bestacc = tst_acc
            Knet.save(bestmodel_path, "model", dram, "history", history)
        end

        lrate = lr_decay * lr
    end
end

function validate(dram::DRAM_v3, data; deterministic=false)
    loss = 0
    ncorrect = ninstances = 0
    for (x,y) in data
        ret = getloss(dram, x, y, deterministic)
        loss += ret[1]
        ncorrect += ret[2]
        ninstances += ret[3]
    end
    loss = loss / length(data)
    loss = [sum(loss), loss...]
    return loss, ncorrect / ninstances
end


"""
return nll
"""
function NLL(scores, y)
    expscores = exp.(scores)
    probabilities = expscores ./ sum(expscores, dims=1)
    answerprobs = (probabilities[y[i]-Int(floor(y[i]/20)),i] for i in 1:length(y))
    mean(-log.(answerprobs))
end

function nllmultidigit(scores, y)

    l = length(y)
    answerprobs = []

    for i=1:l
        k = length(digits(y[i]))

        for j = 1:k
            s = scores[j]
            expscores = exp.(s[:, i])
            probabilities = expscores ./ sum(expscores)
            answerprob = probabilities[digits(y[i])[j] + 10 * (digits(y[i])[j] == 0)]
            push!(answerprobs, answerprob)
        end
    end

    mean(-log.(answerprobs))
end

configurateFor(4)
train()
