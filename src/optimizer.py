#Here we will implement the code to find the better parameters to the best Classifiers found
#in model.py 
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

##################################################################################################

##Classifiers
from sklearn.ensemble import AdaBoostClassifier #begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted
from sklearn.ensemble import BaggingClassifier #Bagging classifier fits base classifiers each on random subsets of the original dataset and aggregate their individual predictions
from sklearn.ensemble import ExtraTreesClassifier #Extremely Random Trees: This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import GradientBoostingClassifier #GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
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
    
def prune(scale = True, pca = False, under = False, over = False):
       
    filename = "../data/df_cluster.csv"
    
    X, y = load_dataset(filename)
    if scale:
        scaled_features = StandardScaler().fit_transform(X.values)
        X = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
    
    if pca:
        n_comp = 13
        columns = []
        for i in range(n_comp):
            columns.append("pca" +str(i+1))
            
        scaled_features = MinMaxScaler().fit_transform(X.values)
        scaled_features_df = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
        pca = PCA(n_components = n_comp)
        pca.fit(scaled_features_df)
        X = pca.transform(scaled_features_df)
        X = pd.DataFrame(X)
        X.columns = columns
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    if under:
        rus = RandomUnderSampler(random_state=0)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    if over:
        ros = RandomOverSampler(random_state=0)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        
    clf = [
            [AdaBoostClassifier(), "AdaBoostClassifier"],
            [BaggingClassifier(), "BaggingClassifier"],
            [ExtraTreesClassifier(), "ExtraTreesClassifier"],
            [GradientBoostingClassifier(), "GradientBoostClassifier"],
            [DecisionTreeClassifier(), "DecisionTreeClassifier"],
            [RandomForestClassifier(), "RandomForestClassifier"],
            [KNeighborsClassifier(), "KNeighborsClassifier"]
        ]
    
    results = {}
    for elem in clf:
        name = elem[1]
        results[name] = []
        
    print("GradientBoosClassifier")
    hyperT = dict(n_estimators =[10 ** i for i in range(2,5)]) #min_samples_split = [i for i in range(1,4)],max_depth = [i for i in range(1,6)], verbose=[i for i in range(3),  min_samples_leaf=[i for i in range(1,6)] criterion = ["friedman_mse", "friedman_mae"] learning_rate = [float(10 ** i)/100 for i in range(6)],
    gridT = GridSearchCV(GradientBoostingClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["GradientBoostClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("***********************************************")
    
    print("RandomForestClassifier")
    hyperT = dict(n_estimators =[i for i in range(100,700,10)], criterion = ["gini", "entropy"], bootstrap = ["True", "False"])
    gridT = GridSearchCV(RandomForestClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["RandomForestClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("************************************************")
        
    print("KNeighborsClassifier")
    hyperT = dict(n_neighbors = [i for i in range(3,8)], weights = ["uniform", "distance"], algorithm = ["auto", "ball_tree", "kd_tree", "brute"], p = [i for i in range(1,4)])
    gridT = GridSearchCV(KNeighborsClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["KNeighborsClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("*************************************************")
    
    print("DecisionTreeClassifier")
    hyperT = dict(criterion = ["gini","entropy"], max_depth = [None]+[i for i in range(1,6)], max_features = ["auto", "sqrt","log2"]) #,  min_samples_leaf=[i for i in range(1,6)], , min_samples_split = [i for i in range(1,6)]
    gridT = GridSearchCV(DecisionTreeClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["DecisionTreeClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("*************************************************")
    
    print("AdaBoostClassifier")
    hyperT = dict(n_estimators =[i for i in range(50,300, 100)], learning_rate = [float(10 ** i)/100 for i in range(4)])
    gridT = GridSearchCV(AdaBoostClassifier(), hyperT, cv = 3, scoring='f1' )
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["AdaBoostClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    
    print("**************************************")
    
   
    print("BaggingClassifier")
    hyperT = dict(n_estimators =[i for i in range(90,600,100)],  bootstrap =  ["True", "False"],bootstrap_features=["True", "False"]) #max_samples = [i for i in range(1,6)], max_features = [i for i in range(1,6)],
    gridT = GridSearchCV(BaggingClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["BaggingClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    print("***************************************")
    
    print("ExtraTreesClassifier")
    hyperT = dict(n_estimators =[i for i in range(100,1000,100)], max_depth = [None]+[i for i in range(1,6)], criterion = ["gini", "entropy"], bootstrap =  ["True", "False"]) #,, verbose=[i for i in range(3) min_samples_split = [i for i in range(1,6)], min_samples_leaf=[i for i in range(1,6)],
    gridT = GridSearchCV(ExtraTreesClassifier(), hyperT, cv = 3, scoring='f1')
    bestT = gridT.fit(X_train, y_train)
    y_pred = bestT.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(gridT.best_params_)
    results["ExtraTreesClassifier"].append([gridT.best_params_ , f1])
    print(f1)
    
    print("********************************************")
    
    if scale:
        f = open("test_prune_scale_under_over.txt", "w")
    else:
        f = open("pca.txt", "w")
        
    f.write("GridCV Results: \n")
    
    for classifier in results:
        f.write(f"{classifier}: {results[classifier]}\n")
        
    f.close()   
    
    pass

if __name__ == "__main__":
    #prune(scale=True, under=False, over = True)
    prune(pca=True,over=True)
    pass
