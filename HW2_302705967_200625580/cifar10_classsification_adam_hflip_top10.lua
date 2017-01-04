--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 

mkdir t5
cd t5/
git clone https://github.com/soumith/cifar.torch.git
cd cifar.torch/
th Cifar10BinToTensor.lua

]]

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'xlua'

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1, bs do
        if flip_mask[i] == 1 then 
			image.hflip(input[i], input[i]) 
			--flip = torch.uniform()
			--if flip<0.33 then image.hflip(input[i], input[i]) 
			--elseif flip<0.66 then image.vflip(input[i], input[i]) 
			--else image.scale(input[i], input[i])
			--end
		end
      end
    end
    self.output:set(input:cuda())
    return self.output
  end
end

local function cast(t)
   if opt.type == 'cuda' then
      require 'cunn'
      return t:cuda()
   elseif opt.type == 'float' then
      return t:float()
   elseif opt.type == 'cl' then
      require 'clnn'
      return t:cl()
   else
      error('Unknown type '..opt.type)
   end
end

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')


local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)



local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



--  ****************************************************************
--  Define our neural network
--  ****************************************************************



local model = nn.Sequential()
model:add(nn.BatchFlip():float())
model:add(cudnn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(cudnn.SpatialMaxPooling(2,2)) 
model:add(cudnn.ReLU(true))                     
model:add(nn.SpatialBatchNormalization(32,1e-3))  
model:add(cudnn.SpatialConvolution(32, 64, 3, 3))
model:add(cudnn.SpatialMaxPooling(2,2))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.3))
model:add(nn.SpatialBatchNormalization(64,1e-3))
model:add(cudnn.SpatialConvolution(64, 32, 1, 1))
model:add(cudnn.SpatialMaxPooling(2,2))
model:add(cudnn.ReLU(true))
model:add(nn.Dropout(0.2))

--[[
--withou FC last layer, but a conv layer that decreas number of parameter to half 

model:add(cudnn.SpatialConvolution(32, 10, 3, 3))
model:add(nn.View(10))
]]

--original:

model:add(nn.View(32*3*3):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
model:add(nn.Linear(32*3*3, 89))
model:add(cudnn.ReLU(true))
--model:add(nn.Dropout(0.4)
model:add(nn.Linear(89, #classes))

model:add(nn.LogSoftMax())                    


model:cuda()
criterion = nn.ClassNLLCriterion():cuda()
--criterion = nn.CrossEntropyCriterion():cuda()
--criterion = nn.MSECriterion():cuda()

w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'
--
local batchSize = 128

optimState = {
    learningRate = 0.003,
	beta1 = 0.9,
	beta2 = 0.999,
	epsilon = 1e-2
}


function forwardNet(data,labels, train)
	timer = torch.Timer()
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize)--:cuda()
		local yt = labels:narrow(1, i, batchSize)--:cuda()
        local y = model:forward(x)
		
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    print(timer:time().real .. ' seconds')
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

---------------------------------------------------------------------

epochs = 800
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

tim = 10
besttest = 1.0

require 'gnuplot'
for t = 1, tim do
	
	for e = 1, epochs do
		trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
		trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
		testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
	end
	print('Training error: ' .. trainError[epochs], 'Training Loss: ' .. trainLoss[epochs], 'Training Acc.: ' .. 1 - trainError[epochs])
	print('Test error: ' .. testError[epochs], 'Test Loss: ' .. testLoss[epochs], 'Test Acc.: ' .. 1 - testError[epochs])
	if testError[epochs] < besttest then
		besttest = testError[epochs]
		torch.save("model_adam_hflip.out", model)
		print('Best test error: ' .. besttest)
		
		local range = torch.range(1, epochs)
		gnuplot.pngfigure('Loss_adam_hflip.png')
		gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
		gnuplot.xlabel('epochs')
		gnuplot.ylabel('Loss')
		gnuplot.plotflush()

		local range = torch.range(1, epochs)
		gnuplot.pngfigure('Error_adam_hflip.png')
		gnuplot.plot({'trainError',trainError},{'testError',testError})
		gnuplot.xlabel('epochs')
		gnuplot.ylabel('Error')
		gnuplot.plotflush()

	end
	model:apply(function(l) l:reset() end)
end



plotError(trainError, testError, 'Classification Error')


--  ****************************************************************
--  Network predictions
--  ****************************************************************

--
model:evaluate()   --turn off dropout

print(classes[testLabels[10]])
print(testData[10]:size())
saveTensorAsGrid(testData[10],'testImg10.jpg')
local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 

-- assigned a probability to each classes
for i=1,predicted:size(2) do
    print(classes[i],predicted[1][i])
end



