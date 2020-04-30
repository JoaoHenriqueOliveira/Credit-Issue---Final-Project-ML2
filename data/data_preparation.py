import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import featuretools as ft
first = True

def details_correction(): #deal with missing dates and correct commas
    data = pd.read_csv("CreditTraining.csv")
    
    nai = data["Net_Annual_Income"] #net annual income = nai
    aux = []
    for nbr in nai: #the numbers with "," cannot be transformed to float so we replace the ","
        if isinstance(nbr, str):
            nbr = nbr.replace(",", "")
        aux.append(nbr)
        
    close = data["Prod_Closed_Date"] #we split here the day, mont and year for this attribute
    day, month, year = [], [], []
    for date in close:
        date = str(date)
        if "nan" == date: #missing dates will be 0/0/0
            day.append(0)
            month.append(0)
            year.append(0)
        else:
            date = date.split("/")
            day.append(date[1])
            month.append(date[0])
            year.append(date[2])
            
    data.drop("Prod_Closed_Date", axis = 1, inplace=True)
    data["Day(Prod_Closed_Date)"] = day 
    data["Month(Prod_Closed_Date)"] = month
    data["Year(Prod_Closed_Date)"] = year
    
    data.drop("Net_Annual_Income", axis = 1, inplace = True)
    data["Net_Annual_Income"] = aux
    
    data.fillna(0, inplace=True)
    data.to_csv("CreditTraining.csv", encoding = 'utf-8',  index = False)
    first = False
    pass

def hot_encoding(): #performs hot encoding (only categorical attributes)
    data = pd.read_csv("CreditTraining.csv")
    cat = ["Customer_Type", "P_Client","Educational_Level", "Marital_Status", 
           "Prod_Sub_Category", "Source","Type_Of_Residence","Prod_Category"]
    
    for col in cat:
        one_hot = pd.get_dummies(data[[col]])
        data = data.drop(col, axis = 1)
        data = data.join(one_hot)
    
    data.to_csv("data_final.csv", encoding = 'utf-8',  index = False)
    pass

def DeepFeatureSynthesis(): #Splits date, fins day of the week, and transforms all numerical values to floats
    data = pd.read_csv("data_final.csv")
    
    customers_df = data[["Id_Customer", "BirthDate", "Customer_Open_Date", 
                         "Number_Of_Dependant", "Years_At_Residence", "Net_Annual_Income",
                         "Years_At_Business", "Prod_Decision_Date", "Nb_Of_Products"]]
                        
    entity = {"customers": (customers_df, "Id_Customer")}
    feature_matrix_customers, _ = ft.dfs(entities=entity, target_entity="customers")

    feature_matrix_customers.to_csv("dfs.csv", encoding = 'utf-8',  index = False)
    pass

def merge(): #merge the treated numerical data with hot encoding
    dfs = pd.read_csv("dfs.csv")
    data = pd.read_csv("data_final.csv")
    numerical = ["Id_Customer", "BirthDate", "Customer_Open_Date", 
                 "Number_Of_Dependant", "Years_At_Residence", "Net_Annual_Income", 
                 "Years_At_Business","Prod_Decision_Date", "Nb_Of_Products"]
    
    for col in numerical:
        data = data.drop(col, axis = 1)
        
    data = data.join(dfs)
    data.to_csv("data_final.csv", encoding = 'utf-8',  index = False)
    pass

def feature_creation(pca = False, cluster = False, n_comp = 5):
    data = pd.read_csv("data_final.csv")
    data["Age(Prod_Decision_Date)"] = data["YEAR(Prod_Decision_Date)"] - data["YEAR(BirthDate)"] #calculate age at decision date
    data["Gap"] = data["YEAR(Prod_Decision_Date)"] - data["YEAR(Customer_Open_Date)"] #gap time between Decision and Open date

    data.to_csv("data_final.csv", encoding = 'utf-8',  index = False)
    
    if pca: #saves the pca in the original dimension of the original data space
        data_pca = pd.read_csv("data2_no_label_cluster.csv")
        data_pca.fillna(0, inplace = True)
    
        scaled_features = StandardScaler().fit_transform(data_pca.values)
        scaled_features_df = pd.DataFrame(scaled_features, index = data_pca.index, columns = data_pca.columns)
    
        mu = np.mean(scaled_features_df, axis = 0)
            
        pca = PCA(n_components = n_comp)
        pca.fit(scaled_features_df)
        
        Xhat = np.dot(pca.transform(scaled_features_df)[:,:n_comp], pca.components_[:n_comp,:])
        Xhat += mu
        Xhat = Xhat.T
        
        dict = {}    
        name = ''
        for i in range(len(Xhat)):
            name = "pca " + str(i)
            dict[name] = list(Xhat[i,])
            name = ''
        df = pd.DataFrame.from_dict(dict)
        print(pca.explained_variance_ratio_)
        df.to_csv("pca_no_label.csv")  
          
    if cluster: #performs cluster in data and save the group as a new feature
        data_cluster = pd.read_csv("data_final.csv")
        label = data_cluster["Y"]
        data_cluster = data_cluster.drop(['Y'], axis = 1)
        
        scaled_features = StandardScaler().fit_transform(data_cluster.values)
        scaled_features_df = pd.DataFrame(scaled_features, index = data_cluster.index, columns = data_cluster.columns)
        
        #K Means -> k = 2
        kmeans = KMeans(n_clusters=2, random_state=0).fit(scaled_features_df)
        group_kmeans = kmeans.labels_
        data_cluster["KMeans 2"] = group_kmeans
        print(group_kmeans)
        #K Means -> k = 3
        kmeans = KMeans(n_clusters=3, random_state=0).fit(scaled_features_df)
        group_kmeans = kmeans.labels_
        data_cluster["KMeans 3"] = group_kmeans
        print(group_kmeans)
        
        #Agglomerative Clustering -> n_clusters = 2, affinity = 'cosine', linkage = 'average'
        agg_clustering = AgglomerativeClustering(n_clusters = 2, affinity = 'cosine', linkage = 'average').fit(scaled_features_df)
        group_agg = agg_clustering.labels_
        data_cluster["Agglomerative Clustering Cosine"] = group_agg
        print(group_agg)
        #Agglomerative Clustering -> n_clusters=2, affinity='euclidean', linkage='ward'
        agg_clustering = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit(scaled_features_df)
        group_agg = agg_clustering.labels_
        data_cluster["Agglomerative Clustering Euclidean"] = group_agg
        print(group_agg)
       
        #DBSCAN -> eps = 3, min_samples = 2
        dbscan = DBSCAN(eps = 3, min_samples = 2).fit(scaled_features_df)
        group_dbscan = dbscan.labels_ 
        data_cluster["DBSCAN eps = 3"] = group_dbscan
        print(group_dbscan)
        
        data_cluster = data_cluster.join(label)
        data_cluster.to_csv("data_final.csv", encoding = 'utf-8',  index = False)        
    pass

if __name__ == "__main__":
    if first:
        details_correction()
        
    hot_encoding()
    DeepFeatureSynthesis()
    merge()
    feature_creation(cluster = True)
    pass