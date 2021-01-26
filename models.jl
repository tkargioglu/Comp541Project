using Random
using Dates
using MLDatasets: MNIST
using Statistics: mean
using IterTools: ncycle, takenth
using Knet
using Knet: accuracy, zeroone, nll, sgd
using Knet: Knet, AutoGrad, dir, Data, minibatch, Param, @diff, value, params, grad, progress, progress!, KnetArray, load, save

experiment = 1
include("networks.jl")
include("imageUtil.jl")
include("config.jl")

"""
Recurrent Attention Model. Mnih et al. 2014 
"""
struct RAM
    glimpsenet::GlimpseNetRAM       
    rnn::RNN
    emissionnet::EmissionNet
    classificationnet::ClassificationNet
    baselinenet::Dense
    σ
end
function RAM(   patchwidth::Int,             # width (and height) of the patches
                glimpselength::Int,          # length of the glimpse feature vector produced by the glimpseNet
                hiddensize::Int,             # length of the hidden state vector of the lstms
                batchsize::Int,
                csize,                       # channel sizes of conv layers in glimpse network
                sigma=0.03)
                    
    result = RAM(   GlimpseNetRAM(patchwidth, csize[1]),
                    RNN(glimpselength, hiddensize),
                    EmissionNet(hiddensize, sigma),
                    ClassificationNet(hiddensize, 10),
                    Dense(hiddensize, 1, identity),
                    sigma)

    result
end

(ram::RAM)(x)       = ramforward(ram, x)
# (ram::RAM)(x, y)    = NLL(ram(x), y)
# (ram::RAM)(data::Data) = mean(ram(x,y) for (x,y) in data)
#(ram::RAM)(x, y)        = ramforwardgetloss(ram, x, y)
#(ram::RAM)(data::Data)  = mean(ram(x,y) for (x,y) in data)

"""
run for one episode
    return loss
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
        
        σ = ram.σ
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

function getloss(ram::RAM, x, y::Union{Array{Int64}, Array{UInt8}}, deterministic=false)
    
    etype = eltype(ram)

    M = 8
    losses = []
    correct = 0
    total = 0
    # monte carlo sampling
    for i =1:M
        scores, baseline, logπ, locations = ramforward(ram, x, deterministic)

        ŷ = vec(map(i->i[1], argmax(Array(value(scores)), dims=1)))     
        r = ŷ .== y;
        r = reshape(r, 1, length(r))
        R = convert(KnetArray{Float32}, r)
        R̂ = R .- value(baseline)
        loss_action = NLL(scores, y)
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
loss(ram::RAM, x, ygold) = getloss(ram, x,ygold)[1]
loss(ram::RAM, d::Data) = mean(getloss(ram, x,y)[1] for (x,y) in d)


function train()

    xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
    xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10
    dtrn = minibatch(xtrn, ytrn, batchsize; xtype = atype)
    dtst = minibatch(xtst, ytst, batchsize; xtype = atype)
    ram = RAM(8, 256, 256, batchsize, 2)
    
    bestmodel_path = string("./best.jld2")
    lastmodel_path = string("./last.jld2")
    history = []
    bestacc = 0.0
    
    trn_losses, trn_acc = validate(ram, dtrn)
    tst_losses, tst_acc = validate(ram, dtst)
    println(
        "epoch=$(length(history)) ",
        "trnloss=$(trn_losses), trnacc=$trn_acc, ",
        "tstloss=$(tst_losses), tstacc=$tst_acc")
    push!(history, ([trn_losses..., trn_acc, tst_losses..., tst_acc]))

    loss(x, ygold) = getloss(ram, x, ygold)[1]
    for epoch = 1:100
        progress!(sgd(loss, dtrn))

        trn_losses, trn_acc = validate(ram, dtrn)
        tst_losses, tst_acc = validate(ram, dtst)
        println(
            "epoch=$(length(history)) ",
            "trnloss=$(trn_losses), trnacc=$trn_acc, ",
            "tstloss=$(tst_losses), tstacc=$tst_acc")
        push!(history, ([trn_losses..., trn_acc, tst_losses..., tst_acc]))
        
        Knet.save(lastmodel_path, "model", ram, "history", history)

        if tst_acc > bestacc
            bestacc = tst_acc
            Knet.save(bestmodel_path, "model", ram, "history", history)
        end
    end
end

function validate(ram::RAM, data; deterministic=false)
    loss = 0
    ncorrect = ninstances = 0
    for (x,y) in data
        ret = getloss(ram, x, y, deterministic)
        loss += ret[1]
        ncorrect += ret[2]
        ninstances += ret[3]
    end
    loss = loss / length(data)
    loss = [sum(loss), loss...]
    return loss, ncorrect / ninstances
end

#=
# test
using Knet
using MLDatasets: MNIST

batchsize = 128
# get mnist data
xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10
dtrn = minibatch(xtrn, ytrn, batchsize; xtype = atype)
dtst = minibatch(xtst, ytst, batchsize; xtype = atype)
ram = RAM(8, 256, 256, batchsize, 2)
progress!(sgd(ram, ncycle(dtrn,50)))


ram = RAM(8, 256, 256, batchsize, 2)

ramforwardgetloss(ram, x, y)
ram(dtst)
GC.gc()
Dates.Time(Dates.now())
progress!(sgd(ram, ncycle(dtrn,50)))
Dates.Time(Dates.now())
ram(dtst)
ram(dtrn)

train_data =  convert.(Array{Float64}, MNIST.traindata())
progress(sgd!(ram, data)

ram(images)
=#

"""
return nll
"""
function NLL(scores, y)
    expscores = exp.(scores)
    probabilities = expscores ./ sum(expscores, dims=1)
    answerprobs = (probabilities[y[i],i] for i in 1:length(y))
    mean(-log.(answerprobs))
end

#=
# test
scores = ones(Int, 10)
y = 1
n = NLL(scores, y)
summary(n)
=#
#=
function runOneGlimpseGetLocation(ram::RAM, image, loc)
      
    loc     = convert(atype, loc)
    loc4    = reshape(loc, 2, 1, 1, 1)
        
    # get glimpses for the batch as a batch (patchsize x patchsize x numscale x batchsize)
    g = take_glimpse(image, loc, patchwidth, numscales, imagewidth)
    
    # get glimpse feature vector (glimpselength)
    glimpsefeature = ram.glimpsenet(g, loc4)

    # r1 r2 hidden states of the lstms (hiddensize)
    r1 = ram.rnnlower(glimpsefeature)
    r2 = ram.rnnupper(r1)

    # get location (2)
    ram.emissionnet(r2)
end
=#

#=
function runOneGlimpseGetPrediction(ram::RAM, image, loc)

    loc     = convert(atype, loc)
    loc4    = reshape(loc, 2, 1, 1, 1)
        
    # get glimpses for the batch as a batch (patchsize x patchsize x numscale x batchsize)
    g = take_glimpse(image, loc, patchwidth, numscales, imagewidth)
    
    # get glimpse feature vector (glimpselength x batchsize)
    glimpsefeature = ram.glimpsenet(g, loc4)

    # r1 r2 hidden states of the lstms (hiddensize x batchsize)
    r1 = ram.rnnlower(glimpsefeature)
    r2 = ram.rnnupper(r1)

    # predicted label
    ram.classificationnet(r1)
end
=#

#=
# test
atype           = KnetArray{Float32}            # use gpu arrays
N               = 6                             # number of glimpses per character
imagewidth      = 2.0                           # image width (and height) in terms of cartesian length (not pixels)
patchwidth      = 10                            # width of the patches
numscales       = 2                             # number of scaled glimpses
glimpselength   = 100                           # length of the glimpse feature vector produced by the glimpseNet
hiddensize      = 256                           # length of the hidden state vector of the lstms
batchsize       = 128                           # size of the batches
wsize           = [5 5; 3 3; 3 3]               # filter sizes of conv layers in glimpse network
csize           = [numscales 64 64 128]         # channel sizes of conv layers in glimpse network

using MLDatasets: MNIST
train_x, train_y = convert.(Array{Float64}, MNIST.traindata())
image = convert(atype, train_x[:, :, 1])
loc = rand(Float32, 2)

ram = RAM(patchwidth, glimpselength, hiddensize, wsize, csize)
label, loc = runOneGlimpse(ram, image, loc, atype)

println(label)
println(convert(Array{Float32}, loc))=#











#=
"""
Deep Recurrent Attention Model. Ba et al. 2015 
"""
struct DRAM


end
=#

#=
# test
using Knet
using MLDatasets: MNIST

batchsize = 128
# get mnist data
xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10
dtrn = minibatch(xtrn, ytrn, batchsize; xtype = atype)
dtst = minibatch(xtst, ytst, batchsize; xtype = atype)
ram = RAM(8, 256, 256, batchsize, 2)
progress!(sgd(ram, ncycle(dtrn,50)))

=#

train()
