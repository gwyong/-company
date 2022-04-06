class RiSE:
    def __init__(self, columns_x, column_y="Risk Rating", dbConnection=True):
        self.columns_x = columns_x # a list of X headers for training
        self.column_y  = column_y # a Y header for training (type string)
        self.dbConnection = dbConnection # if connected, neomCDE data will be loaded directly.
        
    def DataLoad(self):
        import pymongo
        import pandas as pd
        from pymongo import MongoClient
        from pymongo.cursor import CursorType
        
        if self.dbConnection == True:
            myClient = pymongo.MongoClient("mongodb://localhost:27017/")
            cursor   = myClient["database"]["collection"].find({"boardId": "boardId"})
            df       = pd.DataFrame(list(cursor))
            
            nameConvert_dic = {"oldName":"newName"}
            targetCols = [key for key, val in nameConvert_dic.items()]
            df = df[targetCols].rename(columns=nameConvert_dic)
            df = df[self.columns_x+[self.column_y]]
        
        else:
            df = pd.read_csv('./localFile.csv')
            df = df[self.columns_x + [self.column_y]]
        
        print("Loaded data completely.")
        return df
    
    def PreProcessing(self, df, targetCols, tol=5):
        import pandas as pd
        import numpy as np
        
        print("Entered PreProcessing Stage!")
        print("Initial Dataframe Size:", df.shape)
        
        for col in targetCols:
            removedCategories = [key for key, val in df[col].value_counts().items() if val < tol]
            df = df.loc[~df[col].isin(removedCategories)]
        
        df = df.dropna() # removing missing values
        print("Final Dataframe Size:", df.shape)
        return df
    
    def nlpVectorizing(self, df, targetCols, lang="en"):
        import pandas as pd
        import pickle
        import time
        from tqdm import tqdm
        from sentence_transformers import SentenceTransformer
        
        if lang == "en":
            model = SentenceTransformer('all-mpnet-base-v2')
        elif lang == "ko":
            model = SentenceTransformer('jhgan/ko-sroberta-multitask')
        
        vectorDics = {}
        for targetCol in targetCols:
            vectorDics[targetCol]=[]

        for targetCol in targetCols:
            for i in tqdm(range(len(df)), desc=targetCol+" is Vectorizing...", mininterval=0.5):
                time.sleep(0.1)
                sentences = df[targetCol].iloc[i]
                embeddings = model.encode([sentences]) # size (1, 768)
                vectorDics[targetCol].append(embeddings)
        
        nlpCols = []
        for targetCol in targetCols:
            df["vec "+targetCol] = vectorDics[targetCol]
            pos = self.columns_x.index(targetCol)
            self.columns_x[pos] = ("vec "+targetCol)
            nlpCols.append("vec "+targetCol)
        
        pickle.dump(model, open("./savedModels/nlp_"+lang+"_model.pkl", 'wb')) # nlpModel
        print("Sentence Embedding is completed.")
        return df, model, nlpCols

    def trainTestSplit(self, df, testRatio=0.2, randomSeed=0):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(df, test_size=testRatio, random_state=randomSeed)
        return train, test
        
    def Encoding(self, train, test, categoryCols, nlpCols):
        import pandas as pd
        import numpy as np
        import sklearn
        import joblib
        from sklearn.preprocessing import OneHotEncoder
        
        OneHotDics = {}
        
        OHE = OneHotEncoder(sparse=False, handle_unknown="ignore")
        train_x = OHE.fit_transform(np.array(train[categoryCols[0]]).reshape(-1,1))
        test_x  = OHE.transform(np.array(test[categoryCols[0]]).reshape(-1,1))
        
        joblib.dump(OHE, "./savedEncoders/OHE_"+categoryCols[0]+".pkl")
        OneHotDics[categoryCols[0]] = OHE
        
        for col in columns_x:
            if col != categoryCols[0] and col in categoryCols:
                OHE = OneHotEncoder(sparse=False, handle_unknown="ignore")
                train_x_temp = OHE.fit_transform(np.array(train[col]).reshape(-1,1))
                test_x_temp  = OHE.transform(np.array(test[col]).reshape(-1,1))
                joblib.dump(OHE, "./savedEncoders/OHE_"+col+".pkl")
                OneHotDics[col] = OHE
                train_x = np.concatenate((train_x, train_x_temp), axis=1)
                test_x  = np.concatenate((test_x, test_x_temp), axis=1)
            elif col != categoryCols[0] and col in nlpCols:
                train_x = np.concatenate((train_x, np.array(list(train[col].values)).reshape(len(train),-1)), axis=1)
                test_x  = np.concatenate((test_x, np.array(list(test[col].values)).reshape(len(test),-1)), axis=1)
            elif col != categoryCols[0]:
                train_x = np.concatenate((train_x, train[col].to_numpy().reshape(len(train),-1)), axis=1)
                test_x  = np.concatenate((test_x, test[col].to_numpy().reshape(len(test),-1)), axis=1)
            
            
        train_y = train[self.column_y]
        test_y = test[self.column_y]
        
        print("shape of train_x: ", train_x.shape)
        print("shape of test_x : ", test_x.shape)
        print("Encoding is finished.")
        return train_x, train_y, test_x, test_y, OneHotDics
    
    def RiSE(self, train_x, train_y, test_x, test_y, method="SVR"):
        import pandas as pd
        import numpy as np
        import pickle
        
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVR
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.cross_decomposition import PLSRegression as  PLS
        from sklearn.kernel_ridge import KernelRidge as KR
        from sklearn.neural_network import MLPRegressor

        from sklearn.metrics import mean_squared_error
        print()
        print("="*30)
        print("RiSE is based on "+method+".")
        
        if method == "LR":
            model = LinearRegression()
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
        elif method == "SVR":
            model = SVR(kernel='rbf', C=50)
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
        elif method == "RF":
            model = RandomForestRegressor(max_depth=5, random_state=0)
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
        elif method == "MLP":
            model = MLPRegressor(random_state=0, hidden_layer_sizes=(300,3),
                   learning_rate="constant", max_iter=500).fit(train_x, train_y)
            pred_y = model.predict(test_x)
        elif method == "PLS":
            model = PLS()
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
        elif method == "KR":
            model = KR()
            model.fit(train_x, train_y)
            pred_y = model.predict(test_x)
        else:
            print("Error!")
            return
        
        pickle.dump(model, open("./savedModels/model_"+method+".pkl", 'wb'))
        print("Mean squared error: %.2f" % mean_squared_error(test_y, pred_y, squared=False))
        return model