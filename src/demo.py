import os
import numpy as np
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as ptModels

from decord import VideoReader
from dataHandler import DataExplorer
from models import ApC

'''
    python -W ignore src/demo.py
'''

'''
'''
CONFIGURATION = {
    "datasetPath" : "E:\\AI\\DL\\UCF101",
    "infoDFName":"datasetInfo.csv",
    "segmentsInfoDFName":"segmentsInfo.csv",
    "dataRE" : "**/*.avi",
    "maxFrames" : 20,
    "maxEpochs" : 100,
    "datasetParams" : {"batch_size": 8,
          "num_workers": 2,
          "shuffle":False
          },
    "subset":{
        0:"PlayingGuitar",
        1:"RockClimbingIndoor",
        2:"SoccerJuggling",
        3:"BandMarching"
    },
    
    "splitMode":"01",
    "DVFDir":"E:\\AI\\DL\\UCF101\\DeepVisualFeatures",
    "width" : 224,
    "height" : 224,
    "modelsPath":"Models",
    "patience":4,

    "devModels":["ApA","ApB","ApC","ApCB","ApD"]
}

def main():
    transform =  transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize([CONFIGURATION["width"],CONFIGURATION["height"]]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    de = DataExplorer(CONFIGURATION)
    dfInfo = de.datasetInfo[de.datasetInfo["Portion"] == "Test"]
    paths = dfInfo["Path"].values
    print("Choose one from the following paths:")
    print(paths)

    vidPath = input("Give path: ")
    
    video = cv2.VideoCapture(vidPath)
    noFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    info = list()

    segments = divmod(noFrames,CONFIGURATION["maxFrames"])
    if segments[0] >= 1:
        segIdx = np.arange(0,segments[0])*CONFIGURATION["maxFrames"]
        segIdx = segIdx.tolist()

    if len(segIdx) > 0:
        name = "testVid"
        for idx in segIdx:
            videoInfo = {
                "Path":vidPath,
                "NoFrames":noFrames,
                "Name":"{}__{}".format(name,idx),
                "SegmentStart":idx,
                "Activity":vidPath.split("/")[-2],
            }
            info.append(videoInfo)

    dataInfo = pd.DataFrame.from_dict(info)
    vr = VideoReader(vidPath)
    
    ptModel = ptModels.resnet50(pretrained=True)
    for param in ptModel.parameters():
        param.requires_grad = False
    layers = list(ptModel.children())[:-1]
    featureExtractor = nn.Sequential(*layers)    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelPath = os.path.join(CONFIGURATION["modelsPath"],"ApC")
    modelState = os.path.join(modelPath,"Model_{}.pt".format("ApC"))
    model = ApC(de.classes)
    model = model.to(device)
    model.load_state_dict(torch.load(modelState))

    segments = list()
    posts = list()
    for node in dataInfo.itertuples():
        framesArray = list()
        for i in range(node.SegmentStart,node.SegmentStart+CONFIGURATION["maxFrames"]):
            frame = vr[i].asnumpy()
            frame = transform(frame)
            framesArray.append(frame)

            X = torch.stack(framesArray, dim=0)

        with torch.no_grad():
            features = featureExtractor(X)
            features = features.view(features.size(0),-1)

        features = features.to(device)
        features = features.view(1,features.size(0),features.size(1))
        model.eval()
        with torch.no_grad():
            output = model(features)
            posteriors = F.softmax(output)

        posts.extend(posteriors)

    posts = torch.stack(posts, dim=0)
    posts = posts.cpu().data.squeeze().numpy()

    dataInfo["C0"] = posts[:,0]
    dataInfo["C1"] = posts[:,1]
    dataInfo["C2"] = posts[:,2]
    dataInfo["C3"] = posts[:,3]
    dataInfo["C0"] = sum(dataInfo["C0"])/len(dataInfo)
    dataInfo["C1"] = sum(dataInfo["C1"])/len(dataInfo)
    dataInfo["C2"] = sum(dataInfo["C2"])/len(dataInfo)
    dataInfo["C3"] = sum(dataInfo["C3"])/len(dataInfo)

    posteriors = dataInfo.iloc[0,-4:].values
    yPred = np.argmax(posteriors)
    
    print(CONFIGURATION["subset"][yPred])
    #posteriors = np.sum(posteriors,axis=0)/len(dataInfo)
    #predID = np.argmax(posteriors)
    #print(posteriors)
    #print(posteriors.shape)
    #print()
    #print()
    #print()
    #print(posteriors)
    #print(predID)
    #print("[INFO]:  Action in this video: {}".format(CONFIGURATION["subset"][predID]))
  

    '''
        segments.append(X)
    
    X = torch.stack(segments,dim=0)

    noSegs, noFrames, c, w, h = X.shape
    
    X = X.view(noSegs*noFrames,c, w, h)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ptModel = ptModels.resnet50(pretrained=True)
    for param in ptModel.parameters():
        param.requires_grad = False
    layers = list(ptModel.children())[:-1]
    featureExtractor = nn.Sequential(*layers)

    with torch.no_grad():
        features = featureExtractor(X)
        features = features.view(noSegs, noFrames, features.shape[1])
        print(features.shape)

    modelPath = os.path.join(CONFIGURATION["modelsPath"],"ApC")
    modelState = os.path.join(modelPath,"Model_{}.pt".format("ApC"))
    model = ApC(de.classes)
    model = model.to(device)
    model.load_state_dict(torch.load(modelState))

    features = features.to(device)


    model.eval()
    with torch.no_grad():
        output = model(features)
        posteriors = F.softmax(output)
    
    posteriors = posteriors.cpu().data.squeeze().numpy()
    
    dataInfo["C0"] = posteriors[:,0]
    dataInfo["C1"] = posteriors[:,1]
    dataInfo["C2"] = posteriors[:,2]
    dataInfo["C3"] = posteriors[:,3]
    dataInfo["C0"] = sum(dataInfo["C0"])/len(dataInfo)
    dataInfo["C1"] = sum(dataInfo["C1"])/len(dataInfo)
    dataInfo["C2"] = sum(dataInfo["C2"])/len(dataInfo)
    dataInfo["C3"] = sum(dataInfo["C3"])/len(dataInfo)

    posteriors = dataInfo.iloc[0,-4:].values
    yPred = np.argmax(posteriors)
    
    print(CONFIGURATION["subset"][yPred])
    #posteriors = np.sum(posteriors,axis=0)/len(dataInfo)
    #predID = np.argmax(posteriors)
    #print(posteriors)
    #print(posteriors.shape)
    #print()
    #print()
    #print()
    #print(posteriors)
    #print(predID)
    #print("[INFO]:  Action in this video: {}".format(CONFIGURATION["subset"][predID]))
    '''


if __name__ == "__main__":
    main()