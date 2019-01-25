%Get training images

flower_ds = imageDatastore('Flowers','IncludeSubfolders',true,'LabelSource','foldernames');
[trainImgs,testImgs] = splitEachLabel(flower_ds,0.6);
numClasses = numel(categories(flower_ds.Labels));


%Create a network by modifying AlexNet

net = alexnet;
layers = net.Layers;
layers(end-2) = fullyConnectedLayer(numClasses);
layers(end) = classificationLayer;
 

%Set training algorithm options

options = trainingOptions('sgdm','InitialLearnRate', 0.003,'Momentum',0.83);
 

%Perform training

[flowernet,info] = trainNetwork(trainImgs, layers, options);
 

%Use trained network to classify test images

plot(info.TrainingLoss)

flwrPreds = classify(flowernet,testImgs);

flwrActual = testImgs.Labels;% find actual label in testimg

numCorrect = nnz(flwrPreds == flwrActual)% Count non-zero elements in an array. return the number of the matched elements in two array

fracCorrect = numCorrect / numel(flwrPreds) % the correction persentation

confusionchart(flwrActual,flwrPreds)

%-------------------------------------
net = flowernet

inlayer = ly(1)
outlayer = ly(end)
categorynames = outlayer.ClassNames

sz = inlayer.InputSize

imds = imageDatastore('file*.jpg')
auds = augmentedImageDatastore([227 227], imds)

fname = auds.Files

img = readimage(auds,1)% read first image in datastore

[preds,scores] = classify(net,auds)

maxnum = max(scores,[],2)% return max number every row
for i = 4
    score = scores(i,:)
    thresh = median(score + std(score))
    highscores = score > 0.02
    bar(score(highscores))
    
    
    xticks(1:length(score(highscores)))
    xticklabels(categorynames(highscores))
    xtickangle(60)
    
end
