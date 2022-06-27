import os
from dataHandler import DataExplorer
from models import developBaseline, deepFeaturesExtraction, developDeepModel, deployModel

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
    de = DataExplorer(CONFIGURATION)
    if not os.path.isdir(CONFIGURATION["DVFDir"]):
        os.makedirs(CONFIGURATION["DVFDir"],exist_ok=True)
        deepFeaturesExtraction(de,CONFIGURATION)

    developBaseline(de,CONFIGURATION)
    for m in CONFIGURATION["devModels"]:
        if not os.path.isdir(os.path.join(CONFIGURATION["modelsPath"],m)):
            developDeepModel(de,m,CONFIGURATION)
    #        # Actualy re-validation
        deployModel(de,m,CONFIGURATION)

    # Predict test data using the best model. 
    deployModel(de,"ApC",CONFIGURATION,"Test")


if __name__ == "__main__":
    main()