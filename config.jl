"""
Configuration file for all the experiments.
    Hyperparams are set here depending on the experiment. 
"""
# common for all experiments
atype                       = KnetArray{Float32}
batchsize                   = 128       # batchsize
momentum                    = 0.9       # momentum for nesterov
lr                          = 0.1       # learning rate
lr_decay                    = 0.97      # lr decays at each epoch exponentially
num_markov_sampling         = 2         # number of markov samples
rnn_hidden_size             = 256
if experiment == 1                      # mnist 
    
    numglimpses             = 3
    markov_sigma            = 0.03      # standard dev in markov sampling
    unit_width_as_pixels    = 15        # 1 in cartesian equal to this many pixels
    patchwidth              = 8
    numscales               = 2
    wsize           = [5 5; 3 3; 3 3]               # filter sizes of conv layers in glimpse network
    csize           = [numscales 64 64 128]         # channel sizes of conv layers in glimpse network
    hidden_size     = 256
elseif experiment == 2                  # svhn DRAM

    markov_sigma            = 0.003     # standard dev in markov sampling
    unit_width_as_pixels    = 12        # 1 in cartesian equal to this many pixels
    hidden_size     = 1024
else
    throw("Unknown experiment number, couldnt configurate.")
end

println("INFO: Configuration for experiment ", experiment, " is completed.")
