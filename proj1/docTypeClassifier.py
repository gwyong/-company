class docTypeClassifier:
    def __init__(self, inputFile, outputFolder):
        import os, glob

        self.inputFile    = inputFile
        self.outputFolder = outputFolder

    def Preprocessing(self):
        import time
        import re
        import pickle
        import pandas as pd
        import numpy as np

        from tqdm import tqdm
        from sklearn.preprocessing import LabelEncoder
        from sentence_transformers import SentenceTransformer

        df = pd.read_csv(self.inputFile)["파일명", "문서유형"].dropna()

        def cleanText(string):
            pattern = "[-=+%^,#&_/\?:^.@*\"※~ㆍ0-9!』‘|\(\)\[\]`\'…》\”\“\’·]"
            text = re.sub(pattern, " ", string)
            texts = text.split()
            text = " ".join(texts)
            return text
        df["파일명"] = df["파일명"].apply(cleanText)

        model_NLP = SentenceTransformer('jhgan/ko-sroberta-multitask')
        nlpEmbeddings = []
        for i in tqdm(range(len(df)), desc="NLP Vectorizing...", mininterval=0.5):
            time.sleep(0.001)
            sentences = df["파일명"].iloc[i]
            embeddings = model_NLP.encode([sentences]) # size (1, 768)
            nlpEmbeddings.append(embeddings)
            
        df["파일명 vec"] = nlpEmbeddings

        encoder_y = LabelEncoder()
        df["문서유형 enc"] = encoder_y.fit_transform(df["문서유형"])
        self.encoder_y = encoder_y

        fileName = self.outputFolder + "/df_preprocessed.pkl"
        df.to_pickle(fileName)
        # df = pd.read_pickle(fileName)

        return df

    def TrainTestSplit(self, df, testSize=0.2, randomSeed=0):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        train, test = train_test_split(df, test_size=testSize, random_state=randomSeed)

        def DFtoVEC(series):
            return np.array(list(series.values)).reshape(len(series), -1)

        train_x = DFtoVEC(train["파일명 vec"])
        train_y = train["문서유형 enc"]
        test_x  = DFtoVEC(test["파일명 vec"])
        test_y  = test["문서유형 enc"]

        self.train_x = train_x
        self.train_y = train_y
        self.test_x  = test_x
        self.test_y  = test_y

        print("shape of train_x: ", train_x.shape)
        print("shape of test_x : ", test_x.shape)

        return train, test

    def Classify(self, train, test, model="RF"):
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier

        if model == "RF":
            model = RandomForestClassifier(n_estimators=500, n_jobs=8)
        elif model == "SVM":
            model = SVC(verbose=True)
        else:
            model = MLPClassifier()

        model.fit(self.train_x, self.train_y)
        self.model = model
        return model

    def TestReport(self, test):
        import pandas as pd
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        pred_y = self.model.predict(self.test_x)

        print(classification_report(self.test_y, pred_y))

        test["예측 문서유형 enc"] = pred_y
        test["예측 문서유형"] = self.encoder_y.inverse_transform(pred_y)
        test = test["파일명", "문서유형", "예측 문서유형"]
        fileName = self.outputFolder + "/결과보고.csv"
        test.to_csv(fileName)
        return