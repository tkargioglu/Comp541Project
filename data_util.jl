using MAT
using Images
using ImageView
using Statistics
using JLD2

"""
preprocess svhn images according to Goodfellow et al. 2013
    images are grayscaled
    cropped to numbers only
    resized to 64x64
"""
function preprocess_svhn(inputdir, outputdir)

    matdir = string(inputdir, "digitStruct.mat")
    filein = matopen(matdir)
    println("Data was read from ", matdir)

    digitStruct = read(filein)["digitStruct"]
    data = digitStruct["bbox"]

    len = length(data)
    println("There are ", len, " elements in data.")

    # loop over all images
    for i=1:len
        img = load(inputdir * string(i) * ".png")
        img = colorview(Gray, img)
        img = mean(img, dims=1)
        img = permutedims(img, (2, 3, 1))
        img = reshape(img, size(img)[1], size(img)[2])
        img = convert(Array{Float32}, img)
        
        left = Int(minimum(data[i]["left"]))
        right = Int(maximum(data[i]["left"] + data[i]["width"]))
        top = Int(minimum(data[i]["top"]))
        bottom = Int(maximum(data[i]["top"] + data[i]["height"]))
        
        left = left < 1 ? 1 : left
        top = top < 1 ? 1 : top
        right = right > size(img)[2] ? size(img)[2] : right
        bottom = bottom > size(img)[1] ? size(img)[1] : bottom

        img = img[top:bottom, left:right]
        img = imresize(img, 64, 64)
        save(outputdir *  string(i) * ".png", colorview(Gray, img))
        println("Image " * string(i) * " is saved.")
    end
end


using MLDatasets: MNIST
using Knet

"""
Generate images containing 2 digits randomly picked from MNIST
"""
function preprocess_mnist(outputdir, outdim=100)

    # xtrn: 28x28x60000, ytrn: 60000
    xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
    # xtst: 28x28x10000, ytst: 10000
    xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10

    sizefactor = 1
    result_xtrn = zeros(Float32, outdim, outdim, Int(60000*sizefactor))
    result_ytrn = zeros(Float32, Int(60000*sizefactor))

    result_xtst = zeros(Float32, outdim, outdim, Int(10000*sizefactor))
    result_ytst = zeros(Float32, Int(10000*sizefactor))

    for i = 1:Int(70000*sizefactor)

        # random locations for the first digit
        r1, c1 = rand(1:outdim-28+1, 2)

        println("r1 and c1 " * string(r1) * " " * string(c1))
        # random locations for the second digit
        r2 = r1
        c2 = c1

        # prevent intersection
        while true
            if (r2 >= r1 + 28 || r2 <= r1 - 28) && (c2 >= c1 + 28 || c2 <= c1 - 28)
                break;
            end
            r2, c2 = rand(1:outdim-28+1, 2)
        end

        r = zeros(outdim, outdim)
        
        # train set
        if i <= Int(60000*sizefactor)
            # random mnist picks
            a, b = rand(1:Int(60000*sizefactor), 2)

            r[r1:r1+27, c1:c1+27] = xtrn[:, :, a]'
            r[r2:r2+27, c2:c2+27] = xtrn[:, :, b]'

            result_xtrn[:, :, i] = r
            result_ytrn[i] = ytrn[a] + ytrn[b]
        # test set
        else
            # random mnist picks
            a, b = rand(1:Int(10000*sizefactor), 2)

            r[r1:r1+27, c1:c1+27] = xtst[:, :, a]'
            r[r2:r2+27, c2:c2+27] = xtst[:, :, b]'

            result_xtst[:, :, i-Int(60000*sizefactor)] = r
            result_ytst[i-Int(60000*sizefactor)] = ytst[a] + ytst[b]
        end

        #if i%10 == 0
            println("image " * string(i) * " done")
        #end
    end

    xtrn = result_xtrn
    xtst = result_xtst
    ytrn = result_ytrn
    ytst = result_ytst

    @save outputdir xtrn ytrn xtst ytst
end

"""
Generate images containing 2 digits randomly picked from MNIST
"""
function preprocess_mnist_2(outputdir; outdim=100, crop=6)

    # xtrn: 28x28x60000, ytrn: 60000
    xtrn,ytrn = MNIST.traindata(Float32); ytrn[ytrn.==0] .= 10
    # xtst: 28x28x10000, ytst: 10000
    xtst,ytst = MNIST.testdata(Float32);  ytst[ytst.==0] .= 10

    sizefactor = 1
    result_xtrn = zeros(Float32, outdim, outdim, Int(60000*sizefactor))
    result_ytrn = zeros(Float32, Int(60000*sizefactor))

    result_xtst = zeros(Float32, outdim, outdim, Int(10000*sizefactor))
    result_ytst = zeros(Float32, Int(10000*sizefactor))

    imsize = 28 - crop

    for i = 1:Int(70000*sizefactor)

        t = rand(1:6)

        if t == 1
            r1, c1 = 1, 1
        elseif t == 2
            r1, c1 = Int(outdim/2-imsize/2), 1
        elseif t == 3
            r1, c1 = Int(outdim-(imsize-1)), 1
        elseif t == 4
            r1, c1 = Int(outdim-(imsize-1)), Int(outdim-(imsize-1))
        elseif t == 5
            r1, c1 = Int(outdim/2-imsize/2), Int(outdim-(imsize-1))
        else
            r1, c1 = 1, Int(outdim-(imsize-1))
        end

        r2, c2 = r1, c1
        
        if t <= 3
            t2 = rand(4:6)
        else
            t2 = rand(1:3)
        end

        t = t2
        if t == 1
            r1, c1 = 1, 1
        elseif t == 2
            r1, c1 = Int(outdim/2-imsize/2), 1
        elseif t == 3
            r1, c1 = Int(outdim-(imsize-1)), 1
        elseif t == 4
            r1, c1 = Int(outdim-(imsize-1)), Int(outdim-(imsize-1))
        elseif t == 5
            r1, c1 = Int(outdim/2-imsize/2), Int(outdim-(imsize-1))
        else
            r1, c1 = 1, Int(outdim-(imsize-1))
        end

        r = zeros(outdim, outdim)
        
        # train set
        if i <= Int(60000*sizefactor)
            # random mnist picks
            a, b = rand(1:Int(60000*sizefactor), 2)

            r[r1:r1+(imsize-1), c1:c1+(imsize-1)] = xtrn[:, :, a][Int(crop/2+1):Int(imsize+crop/2), Int(crop/2+1):Int(imsize+crop/2)]'
            r[r2:r2+(imsize-1), c2:c2+(imsize-1)] = xtrn[:, :, b][Int(crop/2+1):Int(imsize+crop/2), Int(crop/2+1):Int(imsize+crop/2)]'

            result_xtrn[:, :, i] = r
            result_ytrn[i] = ytrn[a] + ytrn[b]
        # test set
        else
            # random mnist picks
            a, b = rand(1:Int(10000*sizefactor), 2)

            r[r1:r1+(imsize-1), c1:c1+(imsize-1)] = xtst[:, :, a][Int(crop/2+1):Int(imsize+crop/2), Int(crop/2+1):Int(imsize+crop/2)]'
            r[r2:r2+(imsize-1), c2:c2+(imsize-1)] = xtst[:, :, b][Int(crop/2+1):Int(imsize+crop/2), Int(crop/2+1):Int(imsize+crop/2)]'

            result_xtst[:, :, i-Int(60000*sizefactor)] = r
            result_ytst[i-Int(60000*sizefactor)] = ytst[a] + ytst[b]
        end

        if i%10 == 0
            println("image " * string(i) * " done")
        end
    end

    xtrn = result_xtrn
    xtst = result_xtst
    ytrn = result_ytrn
    ytst = result_ytst

    @save outputdir xtrn ytrn xtst ytst
end

"""
Save SVHN data in easily usable format
    Get SVHN data from .mat file into jld2 files
"""
function svhnjld(datadir::String, imagedir::String, outputdir::String)
    xtrn = zeros(64, 64, 1, 33402)
    ytrn = zeros(33402)
    
    xtst = zeros(64, 64, 1, 13068)
    ytst = zeros(13068)

    matdir = string(datadir, "/train/digitStruct.mat");
    filein = matopen(matdir);
    println("Data was read from ", matdir)

    digitStruct = read(filein)["digitStruct"]

    data = digitStruct["bbox"]

    for i=1:33402
        img = load(imagedir * "/train/" * string(i) * ".png")
        img = convert(Array{Float32}, img)
        xtrn[:, :, 1, i] = img

        s = length(data[i]["label"])

        y = 0
        for j=1:s
            y += 10^(s-j) * data[i]["label"][j]
        end

        ytrn[i] = y

        if i%10 == 0
            println("image " * string(i) * " was read.")
        end
    end

    matdir = string(datadir, "/test/digitStruct.mat");
    filein = matopen(matdir);
    println("Data was read from ", matdir)

    digitStruct = read(filein)["digitStruct"]

    data = digitStruct["bbox"]

    for i=1:13068
        img = load(imagedir * "/test/" * string(i) * ".png")
        img = convert(Array{Float32}, img)
        xtst[:, :, 1, i] = img

        s = length(data[i]["label"])

        y = 0
        for j=1:s
            y += 10^(s-j) * data[i]["label"][j]
        end

        ytst[i] = y

        if i%10 == 0
            println("image " * string(i) * " was read.")
        end
    end

    @save outputdir xtrn ytrn xtst ytst
end


svhnjld("../proj/data/svhn/", "./data_preprocessed/", "./data_preprocessed/svhn.jld2")

#=
# preprocess train data
preprocess_svhn("../proj/data/svhn/train/", "./data_preprocessed/train/" )

# preprocess test data (probably will not be used)
preprocess_svhn("../proj/data/svhn/test/", "./data_preprocessed/test/" )

# preprocess extra data
preprocess_svhn("../proj/data/svhn/extra/", "./data_preprocessed/extra/" )
=#
