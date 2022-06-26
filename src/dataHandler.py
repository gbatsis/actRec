import os
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split

'''
    Implementation of data explorer:
    > Generates information such as the assignment of event label, dataset splitting etc.
    > Extracts information about video segments. 
'''
class DataExplorer:
    def __init__(self,CONFIGURATION):
        self.CONFIGURATION = CONFIGURATION
        self.datasetInfo = self.retrieveDatasetInfo()
        self.videoSegments = self.retrieveSegmentsInfo()
        self.classes = len(self.CONFIGURATION["subset"])

    '''
    '''
    def retrieveDatasetInfo(self):
        if os.path.isfile(os.path.join(self.CONFIGURATION["datasetPath"],self.CONFIGURATION["infoDFName"])):
            return pd.read_csv(os.path.join(self.CONFIGURATION["datasetPath"],self.CONFIGURATION["infoDFName"]),index_col=0)
        else:
            return self.generateDatasetInfo()

    '''
    '''
    def getCategory(self,x):
        return x.split("/")[-2]

    '''
    '''
    def passAugmentationInfo(self,trainDF):
        methods = ["RR"]

        augDFs = list()
        for m in methods:
            augDF = trainDF.copy()
            augDF["Augmentation"] = m
            augDFs.append(augDF)
        
        noAug = trainDF.copy()
        noAug["Augmentation"] = "None"
        augDFs.append(noAug)

        finalDF = pd.concat(augDFs)
        return finalDF.sample(frac=1).reset_index(drop=True)

    '''
    '''
    def findInit(self,x):
        seq = x.split("/")[-1].split("_")[:-1]
        return "_".join(seq)

    '''
    '''
    def getName(self,x):
        return x.split("/")[-1].split(".")[0].replace(" ","")
    
    '''
    '''
    def generateDatasetInfo(self):
        listOfTrainTxt = sorted(list(Path(self.CONFIGURATION["datasetPath"]).glob("**/trainlist{}.txt".format(self.CONFIGURATION["splitMode"]))))    
        listOfTestTxt = sorted(list(Path(self.CONFIGURATION["datasetPath"]).glob("**/testlist{}.txt".format(self.CONFIGURATION["splitMode"]))))
        
        trainPaths = pd.read_csv(listOfTrainTxt[0], sep=" ", header=None)
        trainPaths = trainPaths.iloc[:,0].to_frame()
        trainPaths.columns = ["Path"]
        
        testPaths = pd.read_csv(listOfTestTxt[0], sep=" ", header=None)
        testPaths = testPaths.iloc[:,0].to_frame()
        testPaths.columns = ["Path"]

        trainPaths["InitSequence"] = trainPaths["Path"].apply(lambda x: self.findInit(x))
        trainPaths["Activity"] = trainPaths["Path"].apply(lambda x: self.getCategory(x))
        
        trainData = trainPaths[["InitSequence","Activity"]].copy().groupby('InitSequence', as_index=False).last()

        testPaths["InitSequence"] = testPaths["Path"].apply(lambda x: self.findInit(x))
        testPaths["Activity"] = testPaths["Path"].apply(lambda x: self.getCategory(x))
        
        testData = testPaths[["InitSequence","Activity"]].copy().groupby('InitSequence', as_index=False).last()
        trainData = trainData[trainData["Activity"].isin(list(self.CONFIGURATION["subset"].values()))]
        testData = testData[testData["Activity"].isin(list(self.CONFIGURATION["subset"].values()))]
       
        trainData, valData = train_test_split(trainData, test_size=0.1,stratify=trainData["Activity"])

        trainDF = trainPaths[trainPaths["InitSequence"].isin(trainData["InitSequence"].values)]
        valDF = trainPaths[trainPaths["InitSequence"].isin(valData["InitSequence"].values)]
        
        testPaths["InitSequence"] = testPaths["Path"].apply(lambda x: self.findInit(x))
        testPaths["Activity"] = testPaths["Path"].apply(lambda x: self.getCategory(x))

        trainDF["Portion"] = "Training"
        valDF["Portion"] = "Validation"
        testPaths["Portion"] = "Test"

        datasetDF = pd.concat([trainDF,valDF,testPaths]).reset_index(drop=True)
        videoDir = os.path.join(self.CONFIGURATION["datasetPath"],"UCF-101/")
        datasetDF["Path"] = videoDir+datasetDF["Path"]
        datasetDF = datasetDF[datasetDF["Activity"].isin(list(self.CONFIGURATION["subset"].values()))]
        datasetDF["Name"] = datasetDF["Path"].apply(lambda x: self.getName(x))
        datasetDF.to_csv(os.path.join(self.CONFIGURATION["datasetPath"],self.CONFIGURATION["infoDFName"]))
        return datasetDF
       
    '''
    '''
    def retrieveSegmentsInfo(self):
        if os.path.isfile(os.path.join(self.CONFIGURATION["datasetPath"],self.CONFIGURATION["segmentsInfoDFName"])):
            return pd.read_csv(os.path.join(self.CONFIGURATION["datasetPath"],self.CONFIGURATION["segmentsInfoDFName"]),index_col=0)
        else:
            return self.generateSegmentsInfo()
    
    '''
    '''
    def getLabel(self,x):
        return list(self.CONFIGURATION["subset"].keys())[list(self.CONFIGURATION["subset"].values()).index(x)]


    '''
    '''
    def generateSegmentsInfo(self):
        print("[INFO]   Generating information about video segments...")
        info = list()
        for node in self.datasetInfo.itertuples():
            video = cv2.VideoCapture(node.Path)
            noFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            segments = divmod(noFrames,self.CONFIGURATION["maxFrames"])
            if segments[0] >= 1:
                segIdx = np.arange(0,segments[0])*self.CONFIGURATION["maxFrames"]
                segIdx = segIdx.tolist()

            if len(segIdx) > 0:
                name = node.Name
                for idx in segIdx:
                    videoInfo = {
                        "Path":node.Path,
                        "NoFrames":noFrames,
                        "Name":"{}__{}".format(name,idx),
                        "SegmentStart":idx,
                        "Portion":node.Portion,
                        "Activity":node.Activity,
                    }
                    info.append(videoInfo)

        dataInfo = pd.DataFrame.from_dict(info)
        
        dataInfo["Label"] = dataInfo["Activity"].apply(lambda x: self.getLabel(x))
        
        dataInfo.to_csv(os.path.join(self.CONFIGURATION["datasetPath"],self.CONFIGURATION["segmentsInfoDFName"]))
        
        return dataInfo 