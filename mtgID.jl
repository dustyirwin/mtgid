"""
Construct and train a convolutional network to identify an mtg card.
Current mtg set count is 219. 01/16/2018
"""

using Images: load, channelview, float64
using HTTP: download
using Flux

function trainModel(model, loss, opt, mtgData, setCode, cardNum, epochs)
    setHot = getHot(setVec, setCode, cardNum)[1]
    cardImg = getImg(mtgData, setCode, cardNum)
    cardName = mtgData[setCode]["cards"][cardNum]["name"]
    print("Training model on $cardName in $setCode...")
    for i = 1:epochs
        @time Flux.train!(
            loss,
            [(cardImg, setHot)],
            opt,
            cb = Flux.throttle(() -> @show(loss(cardImg, setHot)), 5)
        )
    end
end

function testModel(model, mtgData, setCode, cardNum) #test set prediction model on card image
    setHot = getHot(setVec, setCode, cardNum)[1]
    cardImg = getImg(mtgData, setCode, cardNum)
    cardName = getName(mtgData, setCode, cardNum)

    print("Guessing set for $cardName in $setCode...")
    prediction = model(cardImg)

    if Flux.argmax(prediction) == Flux.argmax(setHot)
        print("...correct answer! ($setCode)\n")
    else
        wrong = setVec[Flux.argmax(prediction)]
        print("...wrong answer. ($wrong) :(\n")
    end
end

function getName(mtgData, setCode, cardNum)
    cardName = mtgData[setCode]["cards"][cardNum]["name"]
    return cardName
end

function getHot(setVec, setCode, cardNum) #create set-length vector & one-hot vector
    setHot = Float64.(Flux.onehot(setCode, setVec))
    cardVec = 1:length(mtgData[setCode]["cards"]);
    cardHot = Float64.(Flux.onehot(cardNum,cardVec));
    return setHot, cardHot
end

function getImg(mtgData, setCode, cardNum)
    #download card image from gatherer and display
    multiverseId = mtgData[setCode]["cards"][cardNum]["multiverseid"];
    cardName = mtgData[setCode]["cards"][cardNum]["name"];
    imgUrl = "http://gatherer.wizards.com/Handlers/Image.ashx?multiverseId=$multiverseId&type=card";
    try
        mkdir("./setimgs/$setCode");
        HTTP.download(imgUrl,"./setimgs/$setCode/$cardName.jpg");
    catch e
        #print("$e\n")
        HTTP.download(imgUrl,"./setimgs/$setCode/$cardName.jpg");
    end
    imgRGB = load("./setimgs/$setCode/$cardName.jpg")

    #convert imgRGB to A = {Float64}[H,W,C,N]
    imgCH = float64.(channelview(imgRGB));
    H, W = length(imgCH[1,:,1]), length(imgCH[1,1,:]);

    R = [ imgCH[1,h,w] for h=1:H,w=1:W ];
    G = [ imgCH[2,h,w] for h=1:H,w=1:W ];
    B = [ imgCH[3,h,w] for h=1:H,w=1:W ];

    cardImg = zeros(H, W, 3, 1);
    cardImg[:,:,1,1] = R;
    cardImg[:,:,2,1] = G;
    cardImg[:,:,3,1] = B;

    return cardImg
end

function saveModel(model, modelName) #save model to file
    open("$modelName.jls", "w") do out
        serialize(out, model)
        print("Model saved to file.")
    end
end

function loadModel(modelName) #load model from file
    open("$modelName.jls", "r") do in
        m = deserialize(in)
        return m
    end
end

function getSetVec(mtgData)
    setList = collect(keys(mtgData))
    setVec = [ (setList[i], mtgData[setList[i]]["releaseDate"]) for i = 1:length(setList) ]
    sort!(setVec, by = setVec -> setVec[2])
    setVec = [ setVec[i][1] for i = 1:length(setVec) ]
    return setVec
end

"""
setm = Chain(
    Conv2D((5,5),3=>100,swish),
    Conv2D((2,2),100=>100,stride=2),
    Conv2D((5,5),100=>200,swish),
    Conv2D((2,2),200=>200,stride=2),
    Conv2D((5,5),200=>300,swish),
    Conv2D((2,2),300=>300,stride=2),
    Conv2D((5,5),300=>200,swish),
    Conv2D((2,2),200=>200,stride=2),
    Conv2D((5,5),200=>100,swish),
    Conv2D((2,2),100=>100,stride=2),
    vec,
    Dense(1500,219,swish),
    Dropout(0.25),
    softmax,
    )
"""
#load model, optimizer, loss function and mtg data
pwd()
cd("/home/dusty/Documents/mtgid")
setm = loadModel("setm")

setopt = ADAM(params(setm); β1=0.9, β2=0.999, ϵ=1e-08, decay=0.0005);
setloss(x, y) = Flux.crossentropy(setm(x), y);
mtgData = Dict()
open("AllSets.json", "r") do f
    global mtgData = JSON.parse(f)
end
setVec = getSetVec(mtgData)

#train model to predict the set on n random cards from k random sets
for k = 1:1
    j = rand(1:length(mtgData))
    for n = 1:5
        i = rand(1:length(mtgData[setVec[j]]["cards"]))
        trainModel(setm, setloss, setopt, mtgData, setVec[j], i, 5)
    end
    testModel(setm, mtgData, setVec[j], rand(1:length(mtgData[setVec[j]]["cards"])))
    saveModel(setm, "setm")
end
testModel(setm, mtgData, setVec[6], rand(1:length(mtgData[setVec[6]]["cards"])))

mtgData[setVec[25]]
for _ = 1:2
    trainModel(setm, setloss, setopt, mtgData, setVec[25], 2, 5)
    testModel(setm, mtgData, setVec[25], rand(1:length(mtgData[setVec[25]]["cards"])))
end
