class SheetAnalysis:
    def __init__(self, folderPath, filePath, saveFileName,
                targetCol,
                minSentenceLength, minWordLength, minCutDeg,
                patterns, deleteWords,
                nGram,
                targetArea="total"):
        
        self.folderPath   = folderPath
        self.filePath     = filePath
        self.saveFileName = saveFileName
        self.targetCol    = targetCol
        self.targetArea   = targetArea
        
        self.minSentenceLength = minSentenceLength
        self.minWordLength     = minWordLength
        self.minCutDeg         = minCutDeg
        self.patterns          = patterns
        self.deleteWords       = deleteWords
        self.nGram             = nGram
        
    def countNull(self):        
        import pandas as pd
        
        df = pd.read_csv(self.folderPath+"/"+self.filePath, encoding='utf-16', sep='\t')
        
        nullDic = {}
        for key, val in df.isnull().sum().items():
            nullDic[key] = val
        pd.DataFrame(nullDic, index=["numNull"]).to_excel(self.folderPath+"/"+self.saveFileName,
                                                         header=True,
                                                         index=True)
        print("Generated an excel file including the number of missing values.")
        
    def MakeWordGraph(self):
        import re
        import nltk, nltk.data
        import pandas as pd
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt

        from itertools import combinations
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        nltk.download('stopwords')
        nltk.download('punkt')

        df = pd.read_csv(folderPath+"/"+filePath, encoding='utf-16', sep='\t')
        if self.targetArea == "total":
            targetData = df[self.targetCol].dropna()
        else:
            df = df.loc[df['regions']==self.targetArea]
            targetData = df[self.targetCol].dropna()
        
        stopWords = stopwords.words('english')
        tokenizer = nltk.data.load('tokenizers/punkt/PY3/english.pickle')
        sentences = [tokenizer.tokenize(contents) for contents in targetData]
        sentences = sum(sentences, [])
        updatedSentences = []

        # nodes refer to a set of meaningful words.
        nodes = []
        for sentence in sentences:
            for pattern in self.patterns:
                sentence = re.sub(pattern, "", sentence)
            sentence = sentence.lower()
            words    = [word for word in word_tokenize(sentence) if word not in stopWords and word not in self.deleteWords and len(word)>self.minWordLength]
            if len(words) > 1:
                nodes += words
                sentence = " ".join(words)
                if len(sentence) > self.minSentenceLength:
                    updatedSentences.append(sentence.lower())
        
        G = nx.Graph()
        nodes = list(set(nodes))

        edges_list = []
        edges_dict = {}
        
        def UpdateEdgesListDict(edges, edges_list=edges_list, edges_dict=edges_dict, G=G):
            for edge in edges:
                if edge not in edges_list:
                    edges_list.append(edge)
                    edges_dict[edge] = 1
                    G.add_edge(edge[0], edge[1], weight=1)
                else:
                    edges_dict[edge] += 1
                    G.add_edge(edge[0], edge[1], weight=edges_dict[edge])

        for sentence in updatedSentences:
            words = sentence.split()
            if len(words) < self.nGram+1:
                edges = list(combinations(words,2))
                UpdateEdgesListDict(edges)
            else:
                for i in range(len(words)-self.nGram):
                    for j in range(1, self.nGram+1):
                        edge = (words[i], words[i+j])
                        if edge not in edges_list:
                            edges_list.append(edge)
                            edges_dict[edge] = 1
                            G.add_edge(edge[0], edge[1], weight=1)
                        else:
                            edges_dict[edge] += 1
                            G.add_edge(edge[0], edge[1], weight=edges_dict[edge])

        degNodes    = list(G.degree())
        degNodes.sort(key=lambda x:x[1])
        removeNodes = [degInfo[0] for degInfo in degNodes if degInfo[1] < self.minCutDeg]
        G.remove_nodes_from(removeNodes)
        
        return G
    
    def VisualizeG(self, G, fontSize, figW, figH, edgeColor='mediumseagreen'):
        import numpy as np
        import networkx as nx
        import matplotlib.pyplot as plt
        from  matplotlib.colors import LinearSegmentedColormap

        colors    = ["darkgreen", "green", "palegreen", "yellow", "lightcoral", "red", "darkred"]
        values    = [0,.15,.4,.5,0.6,.9,1.]
        cmaplist  = list(zip(values, colors))
        cmap      = LinearSegmentedColormap.from_list('rg', cmaplist, N=256)
        
        plt.rc('font', size=fontSize)
        plt.figure(figsize=(figW,figH))
        plt.margins(x=0.01, y=0.01)
        
        pos = nx.kamada_kawai_layout(G)

        nx.draw_networkx_nodes(G, pos,
                               node_color =list(np.array([nodeInfo[1] for nodeInfo in G.degree()])),
                               node_size  =list(np.array([nodeInfo[1] for nodeInfo in G.degree()])*10),
                               cmap=cmap)

        nx.draw_networkx_edges(G, pos, edge_color=edgeColor,
                               width=list(np.array([edge[2] for edge in G.edges(data="weight", default=1)])*2))

        nx.draw_networkx_labels(G, pos,
                                font_family='Times New Roman',
                                font_color='black',
                                font_size=fontSize,
                                font_weight='bold')
        plt.axis('off')
        plt.show()


# # Parameter Selection

# In[155]:


folderPath   = 
filePath     = 
saveFileName = 

# targetCol refers to a target column for analyzing word-Graph.
targetCol = 

# targetArea refers to a region for analyzing word-Graph. (default="total")
targetArea = 

# the length of a minimum meaningful sentence and word.
minSentenceLength = 3
minWordLength     = 3

# patterns are string patterns which will be discarded.
patterns = ["<p>", "</p>", "\\n[1-9]", "\\n", "\\r",
           "[-=+#/?:^$@*\"※~&%ㆍ!』|…》\(\).,]", # special character
           "[0-9]+"] # number

# deleteWords is a set of unnecessary words. This will be updated later.
deleteWords = ["must", "would", "might"]

# nGram is a value which will be used for how many words are connected to a target word.
nGram = 2

# the value which will be used for cutting nodes based on their degree centrality.
minCutDeg = 10


## Main Implementation

SA = SheetAnalysis(folderPath, filePath, saveFileName,
                   targetCol,
                   minSentenceLength, minWordLength, minCutDeg,
                   patterns, deleteWords,
                   nGram,
                   targetArea=targetArea)

# Counting null values
SA.countNull()

# Making a word-graph
G = SA.MakeWordGraph()

# font size
fontDict = {"small":10, "medium":13, "large":25}
fontSize = fontDict["large"]

# figure size
figW, figH = 30, 30

# edge color
edgeColor ='silver'

# Visualizing graph
SA.VisualizeG(G, fontSize, figW, figH, edgeColor=edgeColor)
