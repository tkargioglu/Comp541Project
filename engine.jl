include("ram.jl")
include("util.jl")

function train(model, data)

    dtrn = minibatch(data[1], data[2], batchsize; xtype = atype)
    dtst = minibatch(data[3], data[4], batchsize; xtype = atype)
    
    bestmodel_path = string("./best.jld2")
    lastmodel_path = string("./last.jld2")
    history = []
    bestacc = 0.0
    
    trn_loss, trn_acc = validate(model, dtrn)
    tst_loss, tst_acc = validate(model, dtst)
    
    printEpochSummary(0, trn_acc, tst_acc)

    push!(history, ([trn_loss..., trn_acc, tst_loss..., tst_acc]))

    loss(x, ygold) = getLoss(model, x, ygold)[1]
    for epoch = 1:maxepoch
        progress!(sgd(loss, dtrn))

        trn_losses, trn_acc = validate(model, dtrn)
        tst_losses, tst_acc = validate(model, dtst)

        printEpochSummary(epoch, trn_acc, tst_acc)
        
        push!(history, ([trn_losses..., trn_acc, tst_losses..., tst_acc]))
        
        Knet.save(lastmodel_path, "model", model, "history", history)

        if tst_acc > bestacc
            bestacc = tst_acc
            Knet.save(bestmodel_path, "model", model, "history", history)
        end
    end
end

function validate(model, data; deterministic=false)
    loss = 0
    ncorrect = ninstances = 0
    for (x,y) in data
        ret = getLoss(model, x, y, deterministic)
        loss += ret[1]
        ncorrect += ret[2]
        ninstances += ret[3]
    end
    loss = loss / length(data)
    loss = [sum(loss), loss...]
    return loss, ncorrect / ninstances
end

function getLoss(model, x, y::Union{Array{Int64}, Array{UInt8}}, deterministic=false)

    M = num_monte_carlo
    losses = []
    correct = 0
    total = 0

    for i =1:M
        scores, baseline, logπ, locations = model(x, deterministic)

        ŷ = vec(map(i->i[1], argmax(Array(value(scores)), dims=1)))     
        r = ŷ .== y;
        r = reshape(r, 1, length(r))
        R = convert(atype, r)
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
loss(model, x, ygold) = getLoss(model, x,ygold)[1]
loss(model, d::Data) = mean(getLoss(model, x,y)[1] for (x,y) in d)
