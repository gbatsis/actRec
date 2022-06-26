import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from decord import VideoReader
from sklearn.metrics import f1_score, classification_report, confusion_matrix


from torch.utils.tensorboard import SummaryWriter

'''
'''
class DFEDataset(torch.utils.data.Dataset):
    def __init__(self,dataDF,CONFIGURATION):
        self.dataDF = dataDF
        self.CONFIGURATION = CONFIGURATION
        self.transform =  transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([CONFIGURATION["width"],CONFIGURATION["height"]]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.dataDF)

    def __getitem__(self, index):
        
        node = self.dataDF.iloc[index]
        videoPath = node.Path
        vr = VideoReader(videoPath)
        framesArray = list()

        for i in range(node.SegmentStart,node.SegmentStart+self.CONFIGURATION["maxFrames"]):
            frame = vr[i].asnumpy()
            frame = self.transform(frame)
            framesArray.append(frame)

        X = torch.stack(framesArray, dim=0)

        return X,dict(node)

    '''
    '''
    def augmentor(self,image):
        w = image.shape[1]
        h = image.shape[0]
        M = cv2.getRotationMatrix2D((w/2,h/2), 30, 1.0)
        image = cv2.warpAffine(image,M,(w,h))
        return image

'''
'''

class DFExtractor(nn.Module):
    def __init__(self):
        super(DFExtractor, self).__init__()

        ptModel = models.resnet50(pretrained=True)
        for param in ptModel.parameters():
            param.requires_grad = False
        layers = list(ptModel.children())[:-1]
        self.featureExtractor = nn.Sequential(*layers)


    def forward(self, x):
        featuresSeq = list()
        for s in range(0,x.size(1)):
            with torch.no_grad():
                frameFeatures = self.featureExtractor(x[:, s, :, :, :]) 
                print(frameFeatures.shape)
                frameFeatures = frameFeatures.view(frameFeatures.size(0), -1)
            featuresSeq.append(frameFeatures)

        featuresSeq = torch.stack(featuresSeq, dim=0).transpose_(0, 1)

        return featuresSeq

'''
'''
class DMDataset(torch.utils.data.Dataset):
    def __init__(self,dataDF,CONFIGURATION):
        self.dataDF = dataDF
        self.CONFIGURATION = CONFIGURATION
        self.transform =  transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([CONFIGURATION["width"],CONFIGURATION["height"]]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.dataDF)

    def __getitem__(self, index):
        node = self.dataDF.iloc[index]
    
        featuresPath = os.path.join(self.CONFIGURATION["DVFDir"],"{}.pt".format(node.Name))      
        features = torch.load(featuresPath,map_location=torch.device('cpu'))
        X = features

        '''
        node = self.dataDF.iloc[index]
        videoPath = node.Path
        vr = VideoReader(videoPath)
        framesArray = list()

        for i in range(node.SegmentStart,node.SegmentStart+self.CONFIGURATION["maxFrames"]):
            frame = vr[i].asnumpy()
            #if node.Augmentation != "None":
            #    self.augmentor(frame)
            frame = self.transform(frame)
            framesArray.append(frame)

        X = torch.stack(framesArray, dim=0)

        '''
        y = torch.LongTensor([node.Label])  

        return X,y

'''
    Val Loss: 1.373 - Val F1: 0.97
'''
class ApA(nn.Module):
    def __init__(self,noClasses,lstmLayers=1,lstmHidden=1024):
        super(ApA, self).__init__()
        resnet = models.resnet50(pretrained=True)

        self.lstm = nn.LSTM(
            resnet.fc.in_features,
            lstmHidden,
            lstmLayers,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstmHidden, lstmHidden//2),
            nn.BatchNorm1d(lstmHidden//2, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(lstmHidden//2, noClasses),
        )

    def forward(self, x):
        x , _= self.lstm(x,None)
        x = x[:, -1]
        x = self.classifier(x)
        return x

'''
    Vl = 0.37, F1 = 0.97
'''
class ApB(nn.Module):
    def __init__(self,noClasses,lstmLayers=1,feHidden=1024,lstmHidden=1024):
        super(ApB, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feFinal = self.feFinal = nn.Sequential(
            nn.Linear(resnet.fc.in_features, feHidden),
            nn.BatchNorm1d(feHidden, momentum=0.01),
            )

        self.lstm = nn.LSTM(
            feHidden,
            lstmHidden,
            lstmLayers,
            batch_first=True,
            bidirectional=False,
        )

        self.classifier = nn.Sequential(
            nn.Linear(lstmHidden, lstmHidden//2),
            nn.BatchNorm1d(lstmHidden//2, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(lstmHidden//2, noClasses),
        )

    def forward(self, x):
        batchSize, noFrames, noFeatures = x.shape
        x = x.view(batchSize * noFrames, noFeatures)

        x = x.view(x.size(0), -1)
        x = self.feFinal(x)

        x = x.view(batchSize, noFrames, -1)

        x , _= self.lstm(x,None)
        x = x[:, -1]
        x = self.classifier(x)
        return x


class ApCB(nn.Module):
    def __init__(self,noclasses,feHidden = 1024,visEmb=512,lstmLayers=2,hiddenDim=1024,clfHidden = 512):
        super(ApCB, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feFinal = nn.Sequential(
            nn.Linear(resnet.fc.in_features, feHidden),
            nn.BatchNorm1d(feHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feHidden,visEmb)
            
        )
        self.lstm = nn.LSTM(
            visEmb,
            hiddenDim,
            lstmLayers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hiddenDim, hiddenDim),
            nn.BatchNorm1d(hiddenDim, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hiddenDim,clfHidden),
            nn.BatchNorm1d(clfHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(clfHidden, noclasses),
        )

    def forward(self, x):
        batchSize, noFrames, noFeatures = x.shape
        x = x.view(batchSize * noFrames, noFeatures)
        x = x.view(x.size(0), -1)
        x = self.feFinal(x)
        x = x.view(batchSize,noFrames, -1)
        x , _= self.lstm(x,None)
        x = x[:, -1]
        x = self.output_layers(x)
        return x


'''
'''
class ApC(nn.Module):
    def __init__(self,noclasses,feHidden = 1024,visEmb=512,lstmLayers=2,hiddenDim=1024,clfHidden = 512):
        super(ApC, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feFinal = nn.Sequential(
            nn.Linear(resnet.fc.in_features, feHidden),
            nn.BatchNorm1d(feHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feHidden,visEmb)
            
        )
        self.lstm = nn.LSTM(
            visEmb,
            hiddenDim,
            lstmLayers,
            batch_first=True,
            bidirectional=False,
        )
        self.output_layers = nn.Sequential(
            nn.Linear(hiddenDim, hiddenDim),
            nn.BatchNorm1d(hiddenDim, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hiddenDim,clfHidden),
            nn.BatchNorm1d(clfHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(clfHidden, noclasses),
        )

    def forward(self, x):
        batchSize, noFrames, noFeatures = x.shape
        x = x.view(batchSize * noFrames, noFeatures)
        x = x.view(x.size(0), -1)
        x = self.feFinal(x)
        x = x.view(batchSize,noFrames, -1)
        x , _= self.lstm(x,None)
        x = x[:, -1]
        x = self.output_layers(x)
        return x

'''
'''
class ApD(nn.Module):
    def __init__(self,noclasses,feHidden = 1024,visEmb=256,lstmLayers=3,hiddenDim=1024,clfHidden = 512):
        super(ApD, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feFinal = nn.Sequential(
            nn.Linear(resnet.fc.in_features, feHidden),
            nn.BatchNorm1d(feHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feHidden, feHidden//2),
            nn.BatchNorm1d(feHidden//2, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(feHidden//2,visEmb)
            
        )
        self.lstm = nn.LSTM(
            visEmb,
            hiddenDim,
            lstmLayers,
            batch_first=True,
            bidirectional=True,
        )
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hiddenDim, hiddenDim),
            nn.BatchNorm1d(hiddenDim, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hiddenDim,clfHidden),
            nn.BatchNorm1d(clfHidden, momentum=0.01),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(clfHidden, noclasses),
        )

    def forward(self, x):
        batchSize, noFrames, noFeatures = x.shape
        x = x.view(batchSize * noFrames, noFeatures)
        x = x.view(x.size(0), -1)
        x = self.feFinal(x)
        x = x.view(batchSize,noFrames, -1)
        x , _= self.lstm(x,None)
        x = x[:, -1]
        x = self.output_layers(x)
        return x

'''
'''
def epochTrain(model,device,trainDataGen,optimizer,report=50):
    model.train()

    epochLoss = 0
    dataCounter = 0
    
    yTrueList = list()
    yPredList = list()

    for batchIndex, (X,y) in enumerate(trainDataGen):
        dataCounter += X.size(0)

        X = X.to(device)
        y = y.to(device).view(-1, )

        optimizer.zero_grad()

        output = model(X)

        loss = F.cross_entropy(output, y)

        loss.backward()
        optimizer.step()

        epochLoss += loss.item() * X.size(0)

        pred = output.max(1, keepdim=True)[1]  

        yTrueList.extend(y)
        yPredList.extend(pred)

        if (batchIndex + 1) % report == 0:
            print("[INFO]       {}/{} samples have passed...".format(dataCounter,len(trainDataGen.dataset)))
    
    epochLoss = epochLoss/len(trainDataGen)
    
    yTrue = torch.stack(yTrueList, dim=0)
    yPred = torch.stack(yPredList, dim=0)
    score = f1_score(yTrue.cpu().data.squeeze().numpy(), yPred.cpu().data.squeeze().numpy(),average='macro')

    return epochLoss, score

'''
'''
def epochVal(model, device,valDataGen):
    model.eval()

    epochLoss = 0
    yTrueList = list()
    yPredList = list()
    
    with torch.no_grad():
        for X, y in valDataGen:
            X = X.to(device)
            y = y.to(device).view(-1, )

            output = model(X)

            loss = F.cross_entropy(output, y)
            
            epochLoss += loss.item() * X.size(0)
            
            pred = output.max(1, keepdim=True)[1]  

            yTrueList.extend(y)
            yPredList.extend(pred)
            
    yTrue = torch.stack(yTrueList, dim=0)
    yPred = torch.stack(yPredList, dim=0)

    score = f1_score(yTrue.cpu().data.squeeze().numpy(), yPred.cpu().data.squeeze().numpy(),average='macro')

    epochLoss = epochLoss/len(valDataGen)
    return epochLoss, score

'''
'''
def developDeepModel(de,modelName,CONFIGURATION):
    print("[INFO] Model ---> {}".format(modelName))
    modelPath = os.path.join(CONFIGURATION["modelsPath"],modelName)
    os.makedirs(modelPath,exist_ok=True)
    writerPath = os.path.join(modelPath,"runs")
    trainWriter = SummaryWriter(os.path.join(writerPath,"train"))
    valWriter = SummaryWriter(os.path.join(writerPath,"val"))

    trainDF = de.videoSegments[de.videoSegments["Portion"] == "Training"].sample(frac=1).reset_index(drop=True)
    
    valDF = de.videoSegments[de.videoSegments["Portion"] == "Validation"].sample(frac=1).reset_index(drop=True)
    
    trainingSet = DMDataset(trainDF,CONFIGURATION)
    trainDataGen = torch.utils.data.DataLoader(trainingSet,**CONFIGURATION["datasetParams"])

    valSet = DMDataset(valDF,CONFIGURATION)
    valDataGen = torch.utils.data.DataLoader(valSet,**CONFIGURATION["datasetParams"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if modelName == "ApA":
        model = ApA(de.classes)
    elif modelName == "ApB":
        model = ApB(de.classes)
    elif modelName == "ApC":
        model = ApC(de.classes)
    elif modelName == "ApCB":
        model = ApCB(de.classes)
    elif modelName == "ApD":
        model = ApD(de.classes)
    else:
        print("WRONG MODEL!")
        return
    
    
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainLosses = list()
    trainScores = list()
    valLosses = list()
    valScores = list()

    sinceLastBest = 1
    minLoss = 999999

    for epoch in range(1,CONFIGURATION["maxEpochs"]+1):
        trainLoss, trainScore = epochTrain(model,device,trainDataGen,optimizer,report=50)
        valLoss, valScore = epochVal(model,device,valDataGen)

        print("[INFO]       Epoch {} ---> Training Loss = {:.4} - Training Accuracy {:.4} -Validation Loss = {:.4} - Validation Accuracy = {:.4}".format(epoch,trainLoss,trainScore,valLoss,valScore))
        
        trainLosses.append(trainLoss)
        trainScores.append(trainScore)
        valLosses.append(valLoss)
        valScores.append(valScore)

        trainWriter.add_scalar("Loss",trainLoss,epoch)
        trainWriter.add_scalar("Accuracy",trainScore,epoch)
        valWriter.add_scalar("Loss",valLoss,epoch)
        valWriter.add_scalar("Accuracy",valScore,epoch)

        sinceLastBest += 1
        
        if valLoss < minLoss:
            print("[INFO]       Model saved!")
            torch.save(model.state_dict(), os.path.join(modelPath,"Model_{}.pt".format(modelName)))
            sinceLastBest = 1
            minLoss = valLoss

        if sinceLastBest > CONFIGURATION["patience"]:
            break
    
    print("[INFO]   Model trained!")

    trainWriter.flush()
    valWriter.flush()

'''
'''
def deepFeaturesExtraction(de,CONFIGURATION,report=50):
    print("[INFO]   Deep Features Extraction...")

    devDataset = de.videoSegments

    params = {
            "batch_size": 16,
            "num_workers": 2
            }
 
    devSet = DFEDataset(devDataset,CONFIGURATION)
    devGen = torch.utils.data.DataLoader(devSet,**params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = DFExtractor()
    extractor = extractor.to(device)

    dataCounter = 0

    for batchIndex, (X,info) in enumerate(devGen):
        dataCounter += X.size(0)

        X = X.to(device)

        featuresSeq = extractor(X)
        for inB in range(featuresSeq.shape[0]):
            seq = featuresSeq[inB,:,:]
            featuresPath = os.path.join(CONFIGURATION["DVFDir"],"{}.pt".format(info["Name"][inB]))
            print("[INFO]   Saving tensor to: {}".format(featuresPath))
            torch.save(seq,featuresPath)
            
        if (batchIndex + 1) % report == 0:
            print("[INFO]       {}/{} samples have passed...".format(dataCounter,len(devGen.dataset)))

'''
'''
def deployModel(de,modelName,CONFIGURATION,mode="Val"):

    if modelName == "ApA":
        model = ApA(de.classes)
    elif modelName == "ApB":
        model = ApB(de.classes)
    elif modelName == "ApC":
        model = ApC(de.classes)
    elif modelName == "ApCB":
        model = ApCB(de.classes)
    elif modelName == "ApD":
        model = ApD(de.classes)
    else:
        print("WRONG MODEL!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelPath = os.path.join(CONFIGURATION["modelsPath"],modelName)
    modelState = os.path.join(modelPath,"Model_{}.pt".format(modelName))
    model = model.to(device)
    model.load_state_dict(torch.load(modelState))

    if mode != "ind":
        if mode == "Val":
            valDF = de.videoSegments[de.videoSegments["Portion"] == "Validation"].sample(frac=1).reset_index(drop=True)
        else:
            valDF = de.videoSegments[de.videoSegments["Portion"] == "Test"].sample(frac=1).reset_index(drop=True)
        
        valSet = DMDataset(valDF,CONFIGURATION)        
        valDataGen = torch.utils.data.DataLoader(valSet,**CONFIGURATION["datasetParams"])

        model.eval()

        epochLoss = 0
        yTrueList = list()
        yPredList = list()
        
        with torch.no_grad():
            for X, y in valDataGen:
                X = X.to(device)
                y = y.to(device).view(-1, )

                output = model(X)

                loss = F.cross_entropy(output, y)
                
                epochLoss += loss.item() * X.size(0)
                
                pred = output.max(1, keepdim=True)[1]  

                yTrueList.extend(y)
                yPredList.extend(pred)
                
        yTrue = torch.stack(yTrueList, dim=0)
        yPred = torch.stack(yPredList, dim=0)

        score = f1_score(yTrue.cpu().data.squeeze().numpy(), yPred.cpu().data.squeeze().numpy(),average='macro')

        epochLoss = epochLoss/len(valDataGen)
        
        if mode == "Val":
            print(score,epochLoss)
        else:
            print(classification_report(y_pred=yPred.cpu().data.squeeze().numpy(),y_true=yTrue.cpu().data.squeeze().numpy()))
            print(confusion_matrix(y_pred=yPred.cpu().data.squeeze().numpy(),y_true=yTrue.cpu().data.squeeze().numpy()))
    
    
