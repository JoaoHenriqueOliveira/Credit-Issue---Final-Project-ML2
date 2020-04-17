#Here we will implement the code to find the better parameters to the best Classifiers found
#in model.py 
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV

##################################################################################################

##Classifiers
from sklearn.ensemble import AdaBoostClassifier #begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted
from sklearn.ensemble import BaggingClassifier #Bagging classifier fits base classifiers each on random subsets of the original dataset and aggregate their individual predictions
from sklearn.ensemble import ExtraTreesClassifier #Extremely Random Trees: This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import GradientBoostingClassifier #GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier #Classifier implementing the k-nearest neighbors vote.
from sklearn.ensemble import VotingClassifier

def load_dataset(name):
    print(f"Loading dataset: {name}")
    
    df = pd.read_csv(name)
    df.fillna(0, inplace=True)
    variable_names = list(df)
    print(f"Variable Names: {variable_names}")
    
    X = df.drop("Y", axis = 1)
    y = df["Y"]
    
    return X, y
    
def prune():
    filename = "../data/data_scaled.csv"
    X, y = load_dataset(filename)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    
    clf = [
            [AdaBoostClassifier(), "AdaBoostClassifier"],
            [BaggingClassifier(), "BaggingClassifier"],
            [ExtraTreesClassifier(), "ExtraTreesClassifier"],
            [GradientBoostingClassifier(), "GradientBoostClassifier"],
            [DecisionTreeClassifier(), "DecisionTreeClassifier"],
            [SGDClassifier(), "SGDClassifier"],
            [KNeighborsClassifier(), "KNeighborsClassifier"]
        ]
    
    results = {}
    for elem in clf:
        name = elem[1]
        results[name] = []
        
    print("SDGClassifier")
    hyperT = dict(loss = ["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],penalty = ["l2", "l1", "elasticnet"], alpha = [float(10 ** i)/10000 for i in range(4)], learning_rate = ["constant", "optimal", "invscaling", "adaptive"], eta0 = [float(10 ** i)/10000 for i in range(2)], average = [i for i in range(7,16)]) #early_stopping = ["True", "False"],
    gridT = GridSearchCV(SGDClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["SGDClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("*************************************************")
    
    print("GradientBoosClassifier")
    hyperT = dict(learning_rate = [float(10 ** i)/100 for i in range(4)], n_estimators =[100*i for i in range(1,3)]) #min_samples_split = [i for i in range(1,4)],max_depth = [i for i in range(1,6)], verbose=[i for i in range(3),  min_samples_leaf=[i for i in range(1,6)] criterion = ["friedman_mse", "friedman_mae"]
    gridT = GridSearchCV(GradientBoostingClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["GradientBoostClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("***********************************************")
    
    print("KNeighborsClassifier")
    hyperT = dict(n_neighbors = [i for i in range(3,6)], weights = ["uniform", "distance"], algorithm = ["auto", "ball_tree", "kd_tree", "brute"], p = [i for i in range(1,4)])
    gridT = GridSearchCV(KNeighborsClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["KNeighborsClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("*************************************************")
    
    print("DecisionTreeClassifier")
    hyperT = dict(criterion = ["gini","entropy"],max_depth = [None]+[i for i in range(1,6)], max_features = ["auto", "sqrt","log2"]) #,  min_samples_leaf=[i for i in range(1,6)], , min_samples_split = [i for i in range(1,6)]
    gridT = GridSearchCV(DecisionTreeClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["DecisionTreeClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("*************************************************")
    
    print("AdaBoostClassifier")
    hyperT = dict(n_estimators =[5*i for i in range(5,16)], learning_rate = [float(10 ** i)/100 for i in range(4)])
    gridT = GridSearchCV(AdaBoostClassifier(), hyperT, cv = 3, scoring='f1' )
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["AdaBoostClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    
    print("**************************************")
    
   
    print("BaggingClassifier")
    hyperT = dict(n_estimators =[5*i for i in range(5,50)],  bootstrap =  ["True", "False"],bootstrap_features=["True", "False"]) #max_samples = [i for i in range(1,6)], max_features = [i for i in range(1,6)],
    gridT = GridSearchCV(BaggingClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["BaggingClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("***************************************")
    
    print("ExtraTreesClassifier")
    hyperT = dict(n_estimators =[5*i for i in range(10,25)],criterion = ["gini", "entropy"], bootstrap =  ["True", "False"]) #max_depth = [None]+[i for i in range(1,6)],, verbose=[i for i in range(3) min_samples_split = [i for i in range(1,6)], min_samples_leaf=[i for i in range(1,6)],
    gridT = GridSearchCV(ExtraTreesClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    results["ExtraTreesClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    
    print("********************************************")
    
    
    f = open("prune.txt", "w")
    f.write("GridCV Results: ")
    
    for classifier in results:
        f.write(f"{classifier}: {results[classifier]}")
        
    f.close()   
    
    pass

if __name__ == "__main__":
    prune()
    pass
