include("engine.jl")

using JLD2

function ram_main()
    configurateFor(1)

    if (!configurated)
        throw("No configuration is made, aborting...")
    end
    
    xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
    xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10
    
    data = [xtrn, ytrn, xtst, ytst]
    ram = RAM(patchwidth, hidden_size, hidden_size, numscales, monte_carlo_sigma)
    
    train(ram, data)    
end

function dram_v1_main()
    configurateFor(2)

    if (!configurated)
        throw("No configuration is made, aborting...")
    end

    @load "./modified_mnist.jld2"

    data = [xtrn, ytrn, xtst, ytst]
    dram = DRAM_v1(patchwidth, numscales, hidden_size, rnn_hidden_size, hidden_size, input_size, numclasses, monte_carlo_sigma)
    
    train(dram, data)    
end

function dram_v2_main()
    configurateFor(3)

    if (!configurated)
        throw("No configuration is made, aborting...")
    end

    @load "./modified_mnist.jld2"

    data = [xtrn, ytrn, xtst, ytst]
    dram = DRAM_v2(patchwidth, numscales, hidden_size, rnn_hidden_size, hidden_size, input_size, numclasses, monte_carlo_sigma)
    
    train(dram, data)    
end

function dram_v3_main()
    configurateFor(4)

    if (!configurated)
        throw("No configuration is made, aborting...")
    end

    @load "./svhn.jld2"

    data = [xtrn, ytrn, xtst, ytst]
    dram = DRAM_v2(patchwidth, numscales, hidden_size, rnn_hidden_size, hidden_size, input_size, numclasses, monte_carlo_sigma)
    
    train(dram, data)    
end

ram_main()
# dram_v1_main()
# dram_v2_main()
