require 'nn'
require 'cunn'
require 'optim'
local mnist = require 'mnist';
local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);


criterion = nn.CrossEntropyCriterion():cuda()
batchSize = 96

function forwardNet(data, labels)

    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    local numBatches = 0
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    return avgError
end


epochs = 100
testError = torch.Tensor(epochs)

function load_model(modelfile)

	model = nn.Sequential()
	model:apply(function(l) l:reset() end)
	model = torch.load(modelfile)
	for e = 1, epochs do
		testError[e] = forwardNet(testData, testLabels)
	end
	return model, testError[epochs]

end

upload_model, upload_error = load_model("model.out")
print(upload_model)
print(upload_error)