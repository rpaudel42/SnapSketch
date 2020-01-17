# ******************************************************************************
# sketching.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 10/19/19   Paudel     Initial version,
# ******************************************************************************

import random
from random import randint
import math
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from graph_utils import GraphUtils
import numpy as np
from anomaly_detection import AnomalyDetection
from time import time
import multiprocessing as mp
from sklearn import metrics
import node2vec


class Sketch():
    win_shingles = {}
    win_sketch = []
    disc_shingle = {}
    min_entropy = 0

    def __init__(self):

        #
        # perms = [(randint(0, max_val), randint(0, max_val)) for i in range(self.N)]
        # print("Perms: ", perms)
        pass

    def get_cost(self, sh, curr_g):
        nodes = sh.split('-')
        cost = 0
        for i in range(0, len(nodes)-1):
            cost += curr_g.get_edge_data(nodes[i], nodes[i + 1])[0]['weight']
            # data = curr_g.get_edge_data(nodes[i], nodes[i + 1])
            # cost += data[0]['weight']
        return cost

    def generate_shingles(self, walk_path, k, curr_g):
        shingles = {}
        sh_cost = {}
        for n_walk in walk_path:
            i = 0
            # sh_id = [n_walk[i:i + k] for i in range(len(n_walk) - k + 1)]
            # sh_id = ['_'.join(sh) for sh in sh_id]
            while (i < len(n_walk)-k+1):
                sh_label = curr_g.node[n_walk[i]]['label'] #n_walk[i]
                sh_id = n_walk[i]
                for j in range(1, k):
                    sh_label = str(sh_label) + '-' + str(curr_g.node[n_walk[i + j]]['label'])
                    sh_id = str(sh_id) + '-' + str(n_walk[i+j])

                cost = self.get_cost(sh_id, curr_g)

                if sh_label not in shingles:
                    shingles[sh_label] = 1
                else:
                    freq = shingles[sh_label]
                    shingles[sh_label] = freq + 1

                if sh_label not in sh_cost:
                    sh_cost[sh_label] = cost

                i += 1
        return shingles, sh_cost

    def draw_graph(self, G, nodes):
        # nx.draw(nx_G)  # networkx draw()
        # plt.draw()
        # plt.show()

        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw_networkx_nodes(G, pos)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_labels(G, pos, nodes, font_size=8)
        plt.axis('off')
        plt.savefig("g.png")  # save as png
        plt.show()

    def initialize_hash(self, D, p):
        self.shingle_hash = np.random.choice([0, 1], [0, D+1], p=[1 - p, p])

    def hashing(self, shingles, D, p):
        sh = [[k] for k in shingles]
        # print("New Shingles: ", sh)
        hash = np.random.choice([0, 1], [len(sh), D], p=[1 - p, p])
        hash = np.concatenate((sh, hash), axis=1)
        # print("New Hash: ", hash)
        # shingle_hash = np.concatenate((self.shingle_hash, hash), axis=0)
        # print("Final Hash: ", len(self.shingle_hash), self.shingle_hash)
        return hash

    def sketch_graph(self, shingle_hash, d_shingle, sh_cost, D):
        sketch_vec = np.empty((0, D), int)
        for sh in d_shingle:
            sh_hash = np.array((shingle_hash[shingle_hash[:, 0] == sh])[:, 1:], dtype=int)*sh_cost[sh]
            # print("hash k: ", sh, sh_hash)
            sketch_vec = np.concatenate((sketch_vec, sh_hash), axis=0)
        sketch_vec = np.sum(sketch_vec, axis=0).tolist()
        # print("Sketch Vec: ", sketch_vec)
        return sketch_vec

    def plot_anomaly_score(self, anomaly_score, true_anomalies, sh_vector, parameters):
        anomaly_detection = AnomalyDetection()

        # print("Anomaly Score \n: ", anomaly_score.tolist())
        score = pd.Series([s for s in anomaly_score])

        sorted_d1 = score.sort_values(ascending=False)
        pred_rrcf = score > score.quantile(0.90)
        # print(metrics.classification_report(true_anomalies, pred_rrcf))
        N = 140  # 404
        for k in [100, 200, 300, 400]:
            top_k_real = anomaly_detection.get_top_k_anomalies(sh_vector, k)
            top_k_disp = anomaly_detection.get_top_k_displacement(sorted_d1, k)
            anomaly_detection.get_top_k_performance(top_k_real, top_k_disp, true_anomalies, pred_rrcf, N)

        real = []
        for index, row in sh_vector.iterrows():
            real.append(int(row['anom_count']) * 0.0007)

        plt.figure(1)
        plt.plot(real, color='blue', linewidth=0.5)
        plt.plot(score, color='red', linewidth=1)
        plt.savefig("results/images/vast_"+parameters+'.pdf')
        # plt.show()

    def shingle_sketch(self, graphs, args):
        '''

        :param graphs:
        :param args:
        :return:
        '''

        Ds = [64] #[16, 32, 128, 512, 1024] # [16] #
        # D = args.sketch_size
        ks = [16, 32, 64, 128, 256, 512, 1024] # [128, 256, 512, 1024] # [16] #
        # k = args.disc_shingle
        ps = [0.2] #[0.1, 0.2, 0.3] #, 0.4, 0.5] # [0.1] #
        # p = args.prob
        ns =  [3]#[2, 3, 4, 5, 6]
        # n = args.n_shingle
        ls = [50] #[50, 100, 200, 300, 400, 500] #[50] #
        # l = args.walk_len

        results = []
        parameters = ''

        graph_ids = [id for id in graphs]
        # print(len(graph_ids))
        # graph_ids = graph_ids[:25]
        index = 0

        # '''
        # sketch_list = []

        # self.initialize_hash(D, p)

        print("\n\nStart Sketching ....")
        for l in ls: #for i in range(0,10): # for l in ls:
            for n in ns: # if 1 == 1 #
                graph_shingles = {}
                shingles_cost = {}
                for g in tqdm(range(0, len(graph_ids))):  #starting from the begining
                    index += 1

                    gu = GraphUtils()
                    nx_G = gu.create_graph(graphs[graph_ids[g]])


                    G = node2vec.Graph(nx_G, args.directed, p=10, q=5)
                    G.preprocess_transition_probs()
                    # sketch_start = time()
                    walk_path = G.simulate_walks(l)

                    # nodes = list(G.G.nodes())
                    # random.shuffle(nodes)
                    # sketch_start = time()
                    # pool = mp.Pool(mp.cpu_count())
                    # walk_path =[pool.apply(G.node2vec_walk, args =(walk_len, node)) for node in nodes]
                    # pool.close()

                    # sketch_end = time()
                    # print("Random Walk Time: ", sketch_end-sketch_start)

                    shingles, sh_cost = self.generate_shingles(walk_path, n, nx_G)
                    sorted_sh = sorted(shingles.items(), key=lambda kv: kv[1], reverse=True)

                    graph_shingles[int(graph_ids[g])] = shingles
                    shingles_cost[int(graph_ids[g])] = sh_cost

                print("\nFinish Random Walk .... l = " + str(l) + " n = " + str(n))
                for D in Ds: # if 1 == 1: #
                    for k in ks: # if 1 == 1: #
                        for p in ps: # if 1 == 1: #
                            sketch_list = []
                            # self.initialize_hash(D, p)
                            for g in tqdm(range(0, len(graph_ids))):
                                shingles = graph_shingles[int(graph_ids[g])]
                                # shingles = shingles_cost[int(graph_ids[g])]

                                sorted_sh = sorted(shingles.items(), key=lambda kv: kv[1], reverse=True)
                                d_shingle = [s[0] for s in sorted_sh[:k]]

                                # print("Disc Shingle:", d_shingle)
                                # new_shingle = np.setdiff1d(d_shingle, self.shingle_hash[:, 0].tolist())

                                # if len(new_shingle) > 0:
                                #     self.hashing(new_shingle, D, p)
                                shingle_hash = self.hashing(d_shingle, D, p)
                                sh_cost = shingles_cost[int(graph_ids[g])]
                                Vg = self.sketch_graph(shingle_hash, d_shingle, sh_cost, D)
                                # print("Vector: ", Vg)
                                sketch_list.append(
                                    [graph_ids[g], Vg, graphs[graph_ids[g]]['label'], graphs[graph_ids[g]]['anom_count']])

                            sh_vector = pd.DataFrame(sketch_list, columns=['graphid', 'sketch', 'anomaly', 'anom_count'])
                            sketch_file = 'sketches/'+ args.filename +'/costdim_' + str(D) + '_n_'+ str(n) + '_k_'+ str(k) + 'p_' + str(p) + '_l_' + str(l) + 'p10q5.csv'
                            sh_vector.to_csv(sketch_file)

                            anomaly_detection = AnomalyDetection()
                            # sh_vector = anomaly_detection.read_sketch(sketch_file)
                            # # print(sh_vector)
                            # # sh_vector.loc[sh_vector['anom_count'] >= 50, 'anomaly'] = 1
                            #
                            parameters = "l=" + str(l) + " n=" + str(n) + " d=" + str(D) + " k=" + str(k) + " p=" + str(p)

                            true_anomalies = np.array(sh_vector['anomaly'])
                            a, p, r, f = anomaly_detection.anomaly_detection(sh_vector, args)
                            # results.append([parameters, a, p, r, f])

                            anomaly_score = anomaly_detection.robust_random_cut(sh_vector)
                            #
                            print("\nParameters: ", parameters)
                            print("\nAnomaly Score: ", anomaly_score.tolist())
                            self.plot_anomaly_score(anomaly_score, true_anomalies, sh_vector, parameters)

                            # return sketch_file

        result_vector = pd.DataFrame(results, columns=['parameters', 'accuracy', 'precision', 'recall', 'f1-score'])
        result_file = 'results/'+ args.filename +'.csv'
        result_vector.to_csv(result_file)