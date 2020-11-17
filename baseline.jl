module Model

    using MAT
    #using Knet.minibatch
    using Images
    
    batchsize = 32


    function predict(img)
        # random predict function for baseline model

        # randomly determining number of digits
        r = rand()

        if r < 0.2
            numlength = 1
        elseif r < 0.5
            numlength = 2
        elseif r < 0.8
            numlength = 3
        elseif r < 0.95
            numlength = 4
        else
            numlength = 5
        end

        # randomly determining the result
        r = rand()
        result = floor(r * 10^numlength)  

        Int(result)
    end

    function get_true_label(data)
        
        len = length(data)
        result = 0
        for i in 1:len
            result += data[i] % 10 * 10^(len - i)
        end

        Int(result)
    end

    function test()
        
        println("Testing starts...")
        
        # reading test data
        testdatadir = "./data/svhn/test/"

        matdir = "./data/svhn/test/digitStruct.mat"
        filein = matopen(matdir)
        println("Data was read from ", matdir)
        
        digitStruct = read(filein)["digitStruct"]
        data = digitStruct["bbox"]

        len = length(data)
        println("There are ", len, " elements in test data.")

        # testing loop
        accurate = 0
        for i = 1:len
            img = load(testdatadir * string(i) * ".png")
            img = channelview(img)
            if predict(img) == get_true_label(data[i]["label"])
                accurate += 1
            end
        end

        println("Number of accurate predictions: ", accurate)
        println("Accuracy: ", accurate/len)
        close(filein)

        println("Testing ends.")
    end

    function train()
        println("Training starts...")
        
        # reading training data
        traindatadir = "./data/svhn/train/digitStruct.mat"
        filein = matopen(traindatadir)
        println("Data was read from ", traindatadir)
        
        digitStruct = read(filein)["digitStruct"]
        data = digitStruct["bbox"]

        len = length(data)
        println("There are ", len, " elements in train data.")
        
        iters = len / batchsize
        println(iters)

        for i = 1:iters
            for j = 1:batchsize
                img = load("./data/svhn/train/" * string(Int((i-1)*batchsize+j)) * ".png")
                # calculate the gradient
            end
            # update the model after each minibatch
        end
        
        close(filein)

        println("Training ends.")

    end

    train()
    test()

end