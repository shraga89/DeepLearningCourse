require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'xlua'
require 'optim'

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



criterion = nn.ClassNLLCriterion():cuda()
batchSize = 128

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


epochs = 1
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

upload_model, upload_error = load_model("model_adam_hflip.out")
print(upload_model)
print(upload_error)