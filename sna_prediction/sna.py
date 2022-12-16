import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import random
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

class SNA():
    """  Class for link prediction """
    def __init__(self):
        pass


    def prepare_data(self, edges):
        """
        Function which creates grapsh from existing edges, and reates edges that doesn't exist
        to make the data balanced for future fitting

        Parameters
        ----------
        edges : pandas dataframe with two columns
            

        Returns
        -------
        self.G : graph of existing edges
        self.all_edges : pandas dataframe existing and non-existing edges
        """

        # renameing coulmn names
        edges.set_axis(["User_1", "User_2"], axis='columns', inplace=True)

        # dropping duplicate rows
        edges.drop_duplicates(inplace=True)

        # dropping rows where User_1 = User_2
        edges.drop(edges[edges['User_1']==edges['User_2']].index, inplace=True)

        # shuffle the data
        edges = edges.sample(frac=1, random_state=1, ignore_index=True)

        # creating a graph
        G = nx.from_pandas_edgelist(edges, "User_1", "User_2", create_using = nx.Graph())

        # number of edges
        edge_num = len(G.edges)
        # number of nodes
        node_num = len(G.nodes)

        # creating a dictionary which key is a tuple of two nodes and values are 
        # 1 or 0, depending if they are connected or not respectively
        edge_dict = dict()

        for index, row in edges.iterrows():
            edge_dict[(row["User_1"], row["User_2"])] = 1
            
        missing_edges = set()
        while len(missing_edges) < edge_num:
            # creating random start and end nodes 
            st_node = random.randint(1, node_num)
            en_node = random.randint(1, node_num)
            
            # assigning value 0 to edge, if it doesn't exist in our dictionary
            # else it is 1, as it takes the value from dictionary
            
            # for undirected Graph
            if edge_dict.get((st_node, en_node)) or edge_dict.get((en_node, st_node)):
                edge = 1
            else:
                edge = 0
                
            
            # checking if there is no edge, and it is not a self loop
            if edge == 0 and  st_node != en_node:
                try:
                    # add points to the missing ones if the distance is greater than 2
                    if nx.shortest_path_length(G, source=st_node, target=en_node) > 2:
                        missing_edges.add((st_node, en_node))
                # if there is no path from st_node to en_node
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    missing_edges.add((st_node, en_node))

        # creating a datafram with edges that doesn't exist
        no_edges = pd.DataFrame(list(missing_edges), columns=["User_1", "User_2"])
        
        # creating a graph from train dataset with edges
        self.G = nx.from_pandas_edgelist(edges, "User_1", "User_2", create_using = nx.Graph())

        # index column shows the connection
        # 1: there is an edge, 0: there is no edge
        edges['index'] = 1
        no_edges['index'] = 0

        # combining train and test data for existing and non existing edges
        # x's are dataframes
        all_edges = pd.concat([edges, no_edges], ignore_index=True)


        # shuffling the dataset
        self.all_edges = all_edges.sample(frac=1, random_state=5, ignore_index=True)

        return self

    # calculating shortes path without considering the connection between nodes
    def shortest_path(self, start, end):
        """
        Calculates shortest path between start and end nodes, if they are in our graph,
        else, it return 0 

        Parameters
        ----------
        start : integer
            
        end : integer
            

        Returns
        -------
        path length : integer
        """
        
        try:
            if self.G.has_edge(start, end):
                # removing existing edge 
                self.G.remove_edge(start, end)
                path = nx.shortest_path_length(self.G, start, end)
            else:
                path = nx.shortest_path_length(self.G, start, end)
            
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0


    def common_neighbours(self, start, end):
        """
        Calculates number of common neighbors of two nodes if they in our graph,
        else, it return -1

        Parameters
        ----------
        start : integer
            
        end : integer
            

        Returns
        -------
        number of common neighbors : integer
        """

        try:
            com_n = len(sorted(nx.common_neighbors(self.G, start, end)))
            return com_n
        except nx.NetworkXError:
            return -1

    
    def jaccard_coef(self, start, end):
        """
        Calculates the Jaccard coefficient of two nodes
        returns -1 if nodes are not in our graph

        Parameters
        ----------
        start : integer
            
        end : integer
            

        Returns
        -------
        Jaccard coefficient : float
        """

        if self.G.has_node(start) and self.G.has_node(end):
            coef = sorted(nx.jaccard_coefficient(self.G, [(start, end)]))[0][2]
            return coef
        else:
            return -1

    def reso_alloc(self, start, end):
        """
        Calculates Resource Allocation index of two nodes
        returns -1 if nodes are not in our graph

        Parameters
        ----------
        start : integer
            
        end : integer
            

        Returns
        -------
        Resource Allocation index : float
        """

        if self.G.has_node(start) and self.G.has_node(end):
            coef = sorted(nx.resource_allocation_index(self.G, [(start, end)]))[0][2]
            return coef
        else:
            return -1

    def adamic_adar(self, start, end):
        """
        Calculates Adamic Adar index of two nodes
        returns -1 if nodes are not in our graph

        Parameters
        ----------
        start : integer
            
        end : integer
            

        Returns
        -------
        Adamic Adar index : float
        """

        if self.G.has_node(start) and self.G.has_node(end):
            coef = sorted(nx.adamic_adar_index(self.G, [(start, end)]))[0][2]
            return coef
        else:
            return -1


    def pref_attach(self, start, end):
        """
        Calculates Preferential Attachment of two nodes
        returns -1 if nodes are not in our graph

        Parameters
        ----------
        start : integer
            
        end : integer
            

        Returns
        -------
        Preferential Attachment : float
        """

        if self.G.has_node(start) and self.G.has_node(end):
            coef = sorted(nx.preferential_attachment(self.G, [(start, end)]))[0][2]
            return coef
        else:
            return -1


    def page_rank(self):
        """ Calculates pagerank for each node in the graph
        and also the mean of them"""

        # Page Rank
        self.pr = nx.pagerank(self.G)
        self.pr_mean = sum(self.pr.values())/len(self.pr)
        return self

    def calculate_feature(self, df):
        """
        Function which calls the other functions to calculate all the 
        features for each existing and non-existing edge

        Parameters
        ----------
        df : pandas dataframe
            

        Returns
        -------
        pandas dataframe with new coulmns
        """

        # Shortest Path
        df['shortes_path'] = df.apply(lambda row: self.shortest_path(row['User_1'], row['User_2']), axis=1)
        
        # Common Neighbors
        df['comm_neigh'] = df.apply(lambda row: self.common_neighbours(row['User_1'], row['User_2']), axis=1)

        # Jaccard Coefficient
        df['Jac_coef'] = df.apply(lambda row: self.jaccard_coef(row['User_1'], row['User_2']), axis=1)

        # Resource Allocation
        df['Res_alloc'] = df.apply(lambda row: self.reso_alloc(row['User_1'], row['User_2']), axis=1)

        # Adamic Adar Index
        df['Adam_adar'] = df.apply(lambda row: self.adamic_adar(row['User_1'], row['User_2']), axis=1)

        # Preferencial Attachment
        df['Pref_att'] = df.apply(lambda row: self.pref_attach(row['User_1'], row['User_2']), axis=1)

        # Page ranks
        self.page_rank()
        df['page_rank_1'] = df['User_1'].apply(lambda x: self.pr.get(x, self.pr_mean))
        df['page_rank_2'] = df['User_2'].apply(lambda x: self.pr.get(x, self.pr_mean))
        
        return df

    def x_y_split(self, edges):
        """
        Divides data into two parts for fitting in right way

        Parameters
        ----------
        edges : pandas dataframe
            

        Returns
        -------
        x and y as dataframes
        """

        # preparing data
        self.prepare_data(edges)

        # calculating features
        edges = self.calculate_feature(self.all_edges)

        # x and y 
        self.y = edges['index']
        self.x = edges.drop(['User_1', 'User_2', 'index'], axis = 1)

        return self

    def fit(self, edges):
        """
        Fits data to random Forest Classifier

        Parameters
        ----------
        edges : pandas dataframe
            

        Returns
        -------

        """
        
        # splitting data to x and y parts
        self.x_y_split(edges)

        # initializing classfier
        self.rfc = RandomForestClassifier(max_depth=15,
                            min_samples_leaf=30,
                            min_samples_split=100,
                            n_jobs=-1,
                            random_state=75)

        # fitting the data
        self.rfc.fit(self.x, self.y)
        return self

    def predict(self, edges):
        """

        Parameters
        ----------
        edges : pandas dataframe
            

        Returns
        -------
        array of predictions
        """

        # calculating features
        edges = self.calculate_feature(edges)

        # dropping node numbers to keep only features
        edges.drop(['User_1', 'User_2'], axis = 1, inplace=True)

        # predicting the new edges
        pred = self.rfc.predict(edges)

        return pred



