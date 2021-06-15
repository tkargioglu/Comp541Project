using Knet

"""
Configuration file for all the experiments.
    Hyperparams are set here depending on the experiment.
    
    Experiments:
    1 RAM
    2 DRAM without context on sum of 2 MNIST digits
    3 DRAM with context on sum of 2 MNIST digits
    4 DRAM with context on SVHN recognition
"""

# common for all experiments
const atype                             = KnetArray{Float32}
const batchsize                         = 128       
const maxepoch                          = 1000
const num_monte_carlo                   = 8
const numscales                         = 2

if experiment == 1                      

    const numglimpses                   = 3
    const monte_carlo_sigma             = sqrt(0.03)      
    const unit_width_as_pixels          = 15                            # 1 in cartesian equal to this many pixels
    const patchwidth                    = 8
    const hidden_size                   = 256
    const rnn_hidden_size               = 256                           
elseif experiment == 2                  

    const num_glimpses                  = 5
    const monte_carlo_sigma             = sqrt(0.03)                    # standard dev in mc sampling
    const unit_width_as_pixels          = 12                            # 1 in cartesian equal to this many pixels
    const hidden_size                   = 256    
    const momentum                      = 0.9                           # momentum for nesterov
    const lr                            = 0.1                           # learning rate
    const lr_decay                      = 0.97                          # lr decays at each epoch exponentially
    const rnn_hidden_size               = 512                           # number of lstm units
    const wsize                         = [5 5; 3 3; 3 3]               # filter sizes of conv layers in glimpse network
    const csize                         = [numscales 64 64 128]         # channel sizes of conv layers in glimpse network
    const input_size                    = [100, 100]
    const numclasses                    = 19
elseif experiment == 3                  

    const num_glimpses                  = 5
    const monte_carlo_sigma             = sqrt(0.03)                    # standard dev in mc sampling
    const unit_width_as_pixels          = 12                            # 1 in cartesian equal to this many pixels
    const hidden_size                   = 256
    const downsamplingrate              = 4    
    const momentum                      = 0.9                           # momentum for nesterov
    const lr                            = 0.1                           # learning rate
    const lr_decay                      = 0.97                          # lr decays at each epoch exponentially
    const rnn_hidden_size               = 512                           # number of lstm units
    const wsize                         = [5 5; 3 3; 3 3]               # filter sizes of conv layers in glimpse network
    const csize                         = [numscales 64 64 128]         # channel sizes of conv layers in glimpse network
    const input_size                    = [100, 100]
    const numclasses                    = 19
elseif experiment == 4                  

    const monte_carlo_sigma             = 0.03                          # standard dev in mc sampling
    const unit_width_as_pixels          = 20                            # 1 in cartesian equal to this many pixels
    const hidden_size                   = 1024
    const downsamplingrate              = 4
    const momentum                      = 0.9                           # momentum for nesterov
    const lr                            = 0.1                           # learning rate
    const lr_decay                      = 0.97                          # lr decays at each epoch exponentially
    const rnn_hidden_size               = 512                           # number of lstm units
    const input_size                    = [64, 64]
    const wsize                         = [5 5; 3 3; 3 3]               # filter sizes of conv layers in glimpse network
    const csize                         = [numscales 64 64 128]         # channel sizes of conv layers in glimpse network
    const num_glimpses_per_digit        = 3
    const terminal_label                = -1   
else
    println("Experiment no: ", experiment)
    throw("Unknown experiment number, couldnt configurate.")
end

println("INFO: Configuration for experiment ", experiment, " is completed.")

import Base.println

println(s::String) = begin
    println(stdout, s)
    flush(stdout)
end
