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
    rnnlower::RNN
    rnnupper::RNN 
    emissionnet::EmissionNet
    classificationnet::ClassificationNet
    baselinenet::Dense
end
function RAM(   patchwidth::Int,             # width (and height) of the patches
                glimpselength::Int,          # length of the glimpse feature vector produced by the glimpseNet
                hiddensize::Int,             # length of the hidden state vector of the lstms
                batchsize::Int,
                csize)                       # channel sizes of conv layers in glimpse network
                    
    result = RAM(   GlimpseNetRAM(patchwidth, csize[1]),
                    RNN(glimpselength, hiddensize),
                    RNN(hiddensize, hiddensize),
                    EmissionNet(hiddensize),
                    ClassificationNet(hiddensize, 10),
                    Dense(hiddensize, 1, identity))

    result
end

(ram::RAM)(x)       = ramforward(ram, x)
# (ram::RAM)(x, y)    = NLL(ram(x), y)
# (ram::RAM)(data::Data) = mean(ram(x,y) for (x,y) in data)
(ram::RAM)(x, y)        = ramforwardgetloss(ram, x, y)
(ram::RAM)(data::Data)  = mean(ram(x,y) for (x,y) in data)

losses = Any[]

str=""
"""
Forward pass of RAM for a given minibatch of images
    used in testing not training
    returns scores
"""
function ramforward(ram::RAM, x)
            
    s = size(x)
    if length(s) == 3
        x = reshape(x, s[1], s[2], 1, s[3])
    end

    batchsize = size(x)[4]

    # locations at the first glimpse are random
    loc = convert(atype, randn(2, batchsize))

    ram.rnnlower.h = Param(convert(atype, randn(rnn_hidden_size, batchsize, 1)))
    ram.rnnupper.h = Param(convert(atype, randn(rnn_hidden_size, batchsize, 1)))

    local r1

    # 8 glimpses for each image
    for i=1:numglimpses

        loc4 = reshape(loc, 2, 1, 1, batchsize)
        locsum = convert(atype, zeros(2, batchsize))

        # multiple markov sampling in each glimpse
        for j=1:num_markov_sampling

            # markov sampling of location
            lm = convert(atype, randn(2, batchsize)) * markov_sigma + loc

            # get glimpses for the batch as a batch (patchsize x patchsize x numscale x batchsize)
            g = take_glimpses(x, lm, patchwidth, numscales, unit_width_as_pixels)
            
            # get glimpse feature vector (glimpselength)
            glimpsefeature = ram.glimpsenet(g, loc4)

            # r1 r2 hidden states of the lstms (hiddensize)
            r1 = ram.rnnlower(glimpsefeature)
            r2 = ram.rnnupper(r1)

            # get location (2)
            locsum += ram.emissionnet(r2)
        end
        loc = locsum ./ num_markov_sampling
    end

    # numclasses x batchsize
    ram.classificationnet(r1)
end

"""
"""
function ramforwardgetloss(ram::RAM, x, y)
    
    s = size(x)
    if length(s) == 3
        x = reshape(x, s[1], s[2], 1, s[3])
    end

    batchsize = size(x)[4]

    # locations at the first glimpse are random
    loc = convert(atype, randn(2, batchsize))

    ram.rnnlower.h = Param(convert(atype, randn(rnn_hidden_size, batchsize, 1)))
    ram.rnnupper.h = Param(convert(atype, randn(rnn_hidden_size, batchsize, 1)))

    local r1, r2, F1, F2, baseline

    baseline=0
    F1=0 
    F2=0

    # multiple glimpses for each image
    for i=1:numglimpses

        loc4 = reshape(loc, 2, 1, 1, batchsize)
        locsum = convert(atype, zeros(2, batchsize))

        # multiple markov sampling in each glimpse
        for j=1:num_markov_sampling

            # markov sampling of location
            lm = convert(atype, randn(2, batchsize)) * markov_sigma + loc

            # get glimpses for the batch as a batch (patchsize x patchsize x numscale x batchsize)
            g = take_glimpses(x, lm, patchwidth, numscales, unit_width_as_pixels)
            
            # get glimpse feature vector (glimpselength)
            glimpsefeature = ram.glimpsenet(g, loc4)

            # r1 r2 hidden states of the lstms (hiddensize)
            r1 = ram.rnnlower(glimpsefeature)
            r2 = ram.rnnupper(r1)

            # get location (2)
            locsum += ram.emissionnet(r2)
            
            F1 += NLL(ram.classificationnet(r1), y) / num_markov_sampling

            # gaussian
            f = exp.(-((lm-loc)/markov_sigma).^2/2)/(markov_sigma*sqrt(2*pi))
            F2 += f[1] * f[2] / num_markov_sampling
        end
        loc = locsum ./ num_markov_sampling
        baseline += ram.baselinenet(r2)[1][1]
    end

    # reward 0 or 1 (batchsize)
    R = zeros(batchsize)
    for k=1:batchsize
        res = ram.classificationnet(r1)
        R[k] = argmax(res[:, k]) == y[k]
    end
    
    # R = (first.(Tuple.(argmax(ram.classificationnet(r1), dims=1))))' .== y

    baseline = baseline/numglimpses

    # loss
    mean(F1 .- (R.-baseline/numglimpses)*F2 .+ sum(abs2.(R.-baseline)))
end

#=
# test
using Knet, MLDatasets: MNIST

batchsize = 128
# get mnist data
xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10
dtrn = minibatch(xtrn, ytrn, batchsize; xtype = atype)
dtst = minibatch(xtst, ytst, batchsize; xtype = atype)
println.(summary.((dtrn,dtst)));
x,y = first(dtst)
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