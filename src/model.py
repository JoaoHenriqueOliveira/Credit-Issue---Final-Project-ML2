# util
import pandas as pd

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict, cross_val_score

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


##############################################################################################

##Classifiers
from sklearn.ensemble import AdaBoostClassifier #begins by fitting a classifier on the original dataset and then fits additional copies of the classifier on the same dataset but where the weights of incorrectly classified instances are adjusted
from sklearn.ensemble import BaggingClassifier #Bagging classifier fits base classifiers each on random subsets of the original dataset and aggregate their individual predictions
from sklearn.ensemble import ExtraTreesClassifier #Extremely Random Trees: This class implements a meta estimator that fits a number of randomized decision trees (a.k.a. extra-trees) on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting
from sklearn.ensemble import GradientBoostingClassifier #GB builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions
from sklearn.ensemble import RandomForestClassifier # random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import RidgeClassifier #Classifier using Ridge regression. This classifier first converts the target values into {-1,1} and then treats the problem as a regression task
from sklearn.linear_model import LogisticRegression #Logistic Regression (aka logit, MaxEnt) classifier.
from sklearn.linear_model import LogisticRegressionCV #Logistic Regression CV (aka logit, MaxEnt) classifier.
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier #Classifier implementing the k-nearest neighbors vote.
from sklearn.neighbors import RadiusNeighborsClassifier #Classifier implementing a vote among neighbors within a given radius

from sklearn.svm import SVC #Support Vector Classification
 
from pygam import LogisticGAM, s, f #interesting stuff

from sklearn.ensemble import VotingClassifier #Voting for the best classifiers

def load_dataset(name):
    print(f"Loading dataset: {name}")
    
    df = pd.read_csv(name)
    variable_names = list(df)
    aux = []
    for elem in variable_names:
        if elem != "Y":
            aux.append(elem)
            
    print(f"Variable Names: {variable_names}")
    
    df.fillna(0, inplace=True)
    
    X, y = df[aux], df["Y"]
    
    return X, y

def model(scale = False, pca = False, over = False, under = False):
    clf = [
        [AdaBoostClassifier(), "AdaBoostClassifier"],
        [BaggingClassifier(), "BaggingClassifier"],
        [ExtraTreesClassifier(), "ExtraTreesClassifier"],
        [GradientBoostingClassifier(), "GradientBoostClassifier"],
        [RandomForestClassifier(), "RandomForestClassifier"],
        
        [DecisionTreeClassifier(), "DecisionTreeClassifier"],
        #[MultinomialNB(), "MultinomialNB"],
        [RidgeClassifier(), "RidgeClassifier"],
        [LogisticRegression(), "LogisticRegression"],
        [LogisticRegressionCV(), "LogisticRegressionCV"],
        [SGDClassifier(), "SGDClassifier"],
        [KNeighborsClassifier(), "KNeighborsClassifier"],
        [SVC(), "SVC"]
    ]
    
    performance_train = {}
    performance_test = {}
    performance_cv = {}
    
    filename = "../data/df_cluster.csv"
    X, y = load_dataset(filename)
    
    if scale:        
        scaled_features = StandardScaler().fit_transform(X.values)
        X = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
    
    if pca:
        n_comp = 20
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
                 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    
    if over:
        ros = RandomOverSampler(sampling_strategy = 'minority')
        X_train, y_train = ros.fit_resample(X_train, y_train)
    if under:
        rus = RandomUnderSampler(sampling_strategy = 'majority')
        X_train, y_train = rus.fit_resample(X_train, y_train)
    
    for classifier, clf_name in clf: performance_train[clf_name] = []
    for classifier, clf_name in clf: performance_test[clf_name] = []
    for classifier, clf_name in clf: performance_cv[clf_name] = []
    
    #fit the models above using default hyperparameters
    for elem in clf: #Use each classifier in clf
        classifier = elem[0]
        classifier_name = elem[1]
        print(classifier_name)
        
        try:    
            classifier.fit(X_train, y_train)
            
            y_hat = classifier.predict(X_train)
            #Train Scores:
            f1_train = f1_score(y_train, y_hat)
            accuracy_train = accuracy_score(y_train, y_hat)
            precision_train = precision_score(y_train, y_hat)
            recall_train = recall_score(y_train, y_hat)
            #Print train Scores
            print(f"Train scores: \tf1-score: {f1_train}\tAccuracy: {accuracy_train}\tPrecision: {precision_train}\tRecall: {recall_train}")
            #Sava train scors for comparison
            performance_train[classifier_name].append(f1_train)
            performance_train[classifier_name].append(accuracy_train)
            performance_train[classifier_name].append(precision_train)
            performance_train[classifier_name].append(recall_train)
            
            y_pred = classifier.predict(X_test)
            #Test scores
            f1_test = f1_score(y_test, y_pred)
            accuracy_test = accuracy_score(y_test, y_pred)
            precision_test = precision_score(y_test, y_pred)
            recall_test = recall_score(y_test, y_pred)  
            #Print test scores          
            print(f"Test scores: \tf1-score: {f1_test}\tAccuracy: {accuracy_test}\tPrecision: {precision_test}\tRecall: {recall_test}")
            #Save test scores
            performance_test[classifier_name].append(f1_test)
            performance_test[classifier_name].append(accuracy_test)
            performance_test[classifier_name].append(precision_test)
            performance_test[classifier_name].append(recall_test)
            
            #Cross validation
            y_cv = cross_val_predict(classifier, X, y, cv = 3)
            #CV scores
            f1_cv = f1_score(y, y_cv)
            accuracy_cv = accuracy_score(y, y_cv)
            precision_cv = precision_score(y, y_cv)
            recall_cv = recall_score(y, y_cv)
            #Print CV scores
            print(f"CV scores: \tf1-score: {f1_cv}\tAccuracy: {accuracy_cv}\tPrecision: {precision_cv}\tRecall: {recall_cv}")
            #Save CV scores
            performance_cv[classifier_name].append(f1_cv)
            performance_cv[classifier_name].append(accuracy_cv)
            performance_cv[classifier_name].append(precision_cv)
            performance_cv[classifier_name].append(recall_cv)

            print("\n**********************************************************************")
        except ImportError:
            print("Classifier \"" + classifier_name + "failed.")
                    
    #Write the final summary in summary.txt
    f = open("summary_cluster.txt", "w")
    
    f.write("Train results:\n")
    for classifier in performance_train:
        f.write(f"{classifier} -> F1: {performance_train[classifier][0]} \tAccuracy: {performance_train[classifier][1]}\tPrecision: {performance_train[classifier][2]}\tRecall: {performance_train[classifier][3]} \n")            
    
    f.write("\nTest results:\n")
    for classifier in performance_test:
        f.write(f"{classifier} -> F1: {performance_test[classifier][0]} \tAccuracy: {performance_test[classifier][1]}\tPrecision: {performance_test[classifier][2]}\tRecall: {performance_test[classifier][3]} \n")    
    
    f.write("\nCV results:\n")
    for classifier in performance_cv:
        f.write(f"{classifier} -> F1: {performance_cv[classifier][0]} \tAccuracy: {performance_cv[classifier][1]}\tPrecision: {performance_cv[classifier][2]}\tRecall: {performance_cv[classifier][3]} \n")    
    
    f.close()
    pass


if __name__ == "__main__":
    model(scale = True, over = True)
    pass