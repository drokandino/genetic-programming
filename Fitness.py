import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
#from Projekt import GeneticProgramming
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef

#data = pd.read_csv("avpdb.csv")
#data = shuffle(data, random_state=5)

### DEMO of classifier
# X = [[0, 0],
#      [1, 1]
#      ]

# y = [0, 1]

# classifier = svm.SVC()
# classifier.fit(X, y)

# print(classifier.predict([[1, 1], 
#                           [0, 0]]))


def quality_prediction(data):
       
       #global data, dataTrain, dataTest
       
       #rowCount = len(data.index)

       #Podaci nasumicno izmjesani, kako ne bi bili sortrani po nekoj od kolona
       #random_state - Determines random number generation for shuffling the data.
       #Pass an int for reproducible results across multiple function calls.
       
       
       #Kako bi se izbjegla nebalansiranost podataka (npr. 95% pozitivni i 5% negativni brojevi)
       #Provedena je stratificirana podjela podataka
       # dataTrain, dataTest = train_test_split(data, stratify=data["label"],  test_size=0.2, random_state=42)
       # dataValidate, dataTest = train_test_split(dataTest, stratify=dataTest["label"], test_size=0.5, random_state=20)
       #
       dataTrain, dataValidate = train_test_split(data, stratify=data["label"],  test_size=0.11, random_state=42)

       #Pareto 80/20
       #Testirano sa 70/30, 80/20 je bolje       
       # dataTrain = data[:int(rowCount*0.8)]
       # dataTest = data[int(rowCount*0.8):]

       #Prepare train data
       dataTrainX = dataTrain.drop(["label", "sequence"], axis='columns')
       dataTrainY = dataTrain["label"]

       # Support vector machine model
       #classifierSVC = svm.SVC()
       #classifierSVC.fit(dataTrainX, dataTrainY)

       # MLP model
       #classifierMLP = MLPClassifier(hidden_layer_sizes = (11, 11, 11), max_iter = 800)
       #classifierMLP.fit(dataTrainX, dataTrainY)

       #Random forest model
       #200 stabala
       classifierRndForest = RandomForestClassifier(n_estimators = 200)
       classifierRndForest.fit(dataTrainX, dataTrainY)

       # print(classifierRndForest.feature_importances_)

       #Prepare test data
       dataValidateX = dataValidate.drop(["label", "sequence"], axis='columns')
       dataValidateY = dataValidate["label"]

       #Predict with test data
       #predicted_SVC_Y = classifierSVC.predict(dataValidateX)
       #predicted_MLP_Y = classifierMLP.predict(dataValidateX)
       predicted_RndForest_Y = classifierRndForest.predict(dataValidateX)

       #Accuracy of SVC model
       #tocnePredikcije = 0
       #for predicted, real in zip(predicted_SVC_Y, dataValidateY):
           #if predicted == real:
               #tocnePredikcije += 1
       #predictionAccuracySVC = tocnePredikcije / len(dataValidateY)

       #Inicijalno testirano koristeći točnost, ali nepraktično kod nebalansiranih skupova
       #tocnePredikcije = 0
       #for predicted, real in zip(predicted_RndForest_Y, dataValidateY):
           #if predicted == real:
               #tocnePredikcije += 1
       #predictionAccuracyRndForest = tocnePredikcije / len(dataValidateY)
       
       #Accuracy of Random Forest model  
       #Matthews correlation coefficient, [-1, +1], compatible with classes of different sizes 
       #Testirano sa 100, 200, 500 stabala, 200 kompromis između točnosti i brzine TBD
       #mcc između 0.5 i 0.7
       mcc = matthews_corrcoef(dataValidateY, predicted_RndForest_Y)

       #Accuracy of MLP (neural network) model        
       #tocnePredikcije = 0
       #for predicted, real in zip(predicted_MLP_Y, dataValidateY):
           #if predicted == real:
               #tocnePredikcije += 1
       #predictionAccuracyMLP = tocnePredikcije / len(dataValidateY)
       
       #predictionAccuracyAVG = (predictionAccuracyMLP + predictionAccuracyRndForest + predictionAccuracySVC) / 3

       return mcc, classifierRndForest.feature_importances_, classifierRndForest


def fitnessTestData(data, classifierRndForest):

       # Prepare test data
       dataValidateX = data.drop(["label", "sequence"], axis='columns')
       dataValidateY = data["label"]

       # Predict with test data
       predicted_RndForest_Y = classifierRndForest.predict(dataValidateX)

       mcc = matthews_corrcoef(dataValidateY, predicted_RndForest_Y)

       return mcc


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
