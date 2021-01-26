using Knet
experiment = 1
"""
Configuration file for all the experiments.
    Hyperparams are set here depending on the experiment. 
"""
# common for all experiments
const atype                       = KnetArray{Float32}
const batchsize                   = 128       # batchsize
const momentum                    = 0.9       # momentum for nesterov
const lr                          = 0.1       # learning rate
const lr_decay                    = 0.97      # lr decays at each epoch exponentially
const num_markov_sampling         = 2         # number of markov samples
const rnn_hidden_size             = 256
if experiment == 1                      # mnist 
    
    const numglimpses             = 3
    const markov_sigma            = 0.03      # standard dev in markov sampling
    const unit_width_as_pixels    = 15        # 1 in cartesian equal to this many pixels
    const patchwidth              = 8
    const numscales               = 2
    const wsize           = [5 5; 3 3; 3 3]               # filter sizes of conv layers in glimpse network
    const csize           = [numscales 64 64 128]         # channel sizes of conv layers in glimpse network
    const hidden_size     = 256
elseif experiment == 2                  # svhn DRAM

    const markov_sigma            = 0.003     # standard dev in markov sampling
    const unit_width_as_pixels    = 12        # 1 in cartesian equal to this many pixels
    const hidden_size     = 1024
else
    throw("Unknown experiment number, couldnt configurate.")
end

println("INFO: Configuration for experiment ", experiment, " is completed.")

import Base.println
println(s::String) = begin
    println(stdout, s)
    flush(stdout)
end
