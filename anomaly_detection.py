# ******************************************************************************
# anomaly_detection.py
#
# Date      Name       Description
# ========  =========  ========================================================
# 6/12/19   Paudel     Initial version,
# ******************************************************************************

# anomaly detection based on Isolation Forest

from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, gradient_boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict, GridSearchCV, train_test_split
from sklearn.cluster import DBSCAN, KMeans, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from pyod.models.knn import KNN

import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import matplotlib as mat
from matplotlib.cm import get_cmap
from mpl_toolkits.mplot3d import Axes3D
from itertools import cycle
import glob

import statistics


#anomaly detection based on Robust Random Cut Forest
import rrcf
#
# from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import random
from daal.algorithms.svm import training, prediction


class AnomalyDetection:

    def __init__(self):
        # print("\n\n------ Start Anomaly Detection ---- ")
        pass

    def read_csv_file(self, csv_file):
        tcp = pd.DataFrame(index=[], columns=[])
        tcp1 = pd.read_csv(csv_file)
        tcp = tcp.append(tcp1, ignore_index=True)
        tcp = tcp.iloc[:, [0, 1, 2, 3]]
        tcp.columns = ['source', 'destination', 'anomaly', 'time_past']
        return tcp

    def read_sketch_baseline(self, file_name):
        sketches = pd.DataFrame(index=[], columns=[])
        sketch = pd.read_csv(file_name, converters={"sketch": lambda x: x.strip("[]").split(", ")})
        sketches = sketches.append(sketch, ignore_index=True)
        sketches = sketches.iloc[:, [1, 2, 3]]
        # print(sketches)
        # sketches = sketches.iloc[:, [0, 1, 2]]
        sketches.columns = ['graphid', 'sketch', 'anomaly']
        sketches['sketch'] = sketches['sketch'].apply(lambda x: pd.to_numeric(x, errors='ignore', downcast='float'))
        sketches['graphid'] = sketches['graphid'].astype(int)
        return sketches

    def read_sketch(self, file_name):
        sketches = pd.DataFrame(index=[], columns=[])
        sketch = pd.read_csv(file_name, converters={"sketch": lambda x: x.strip("[]").split(", ")})
        sketches = sketches.append(sketch, ignore_index=True)
        sketches = sketches.iloc[:, [1, 2, 3, 4]]
        sketches.columns = ['graphid', 'sketch', 'anomaly', 'anom_count']
        sketches['sketch'] = sketches['sketch'].apply(lambda x: pd.to_numeric(x, errors='ignore', downcast='float'))
        sketches['graphid'] = sketches['graphid'].astype(int)
        return sketches

    def get_top_k_displacement(self, sorted_disp, k):
        top_k_index = []
        for index, value in sorted_disp.items():
            if len(top_k_index) < k:
                top_k_index.append(index)
        return top_k_index

    def get_top_k_anomalies(self, sketch_vector, k):
        top_k_index = []
        i = 0
        for index, row in sketch_vector.nlargest(k, 'anom_count').iterrows():
            # print("Top: ", i, " ", row['graphid'], " index ", index, "anom_count: ", row['anom_count'])
            if row['anomaly'] == 1:
                top_k_index.append(index)
            i += 1
        return top_k_index

    def get_top_k_performance(self, top_k_real, top_k_disp,  true_anomalies, predicted_anomalies, N):
        print("\n--- Performance on (K = ", len(top_k_real), " ) ")
        true_positive = []
        false_positive = []
        for index in top_k_real: #top_k_disp:  # top_k_real
            if true_anomalies[index] == True and predicted_anomalies[index] == True:
                true_positive.append(index)
            else:
                false_positive.append(index)

        # true_positive.sort()
        # false_positive.sort()
        # print("True Positive: ",  len(true_positive), true_positive)
        # print("False Positive: ", len(false_positive), false_positive)
        print("Precision: ", len(true_positive)/len(top_k_disp))
        print("Recall: ", len(true_positive)/N)

        # k_true = []
        # k_predicted = []
        #
        # for index in top_k_real:
        #     k_true.append(true_anomalies[index])
        #     k_predicted.append(predicted_anomalies[index])
        # print("Top K True Anomalies: ", top_k_real)
        # # target_names = ['Normal', 'Anomaly']
        # print(metrics.classification_report(k_true, k_predicted))
        # p = metrics.precision_score(k_true, k_predicted)
        # r = metrics.recall_score(k_true, k_predicted)
        # f = metrics.f1_score(k_true, k_predicted)
        # return p, r, f

    def knn_detector(self, sketch_vector):
        clf_name = 'KNN'
        clf = KNN()
        clf.fit(X_train)

        # get the prediction label and outlier scores of the training data
        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores

    def rrcf_shingle(self, sketch_vector):
        sketch = sketch_vector['sketch'].tolist()
        # pca =PCA(n_components=50)
        # sketch = pca.fit_transform(sketch)
        # print(sketch)
        # sketch = preprocessing.scale(sketch)
        # print(sketch)
        num_trees = 50
        shingle_size = 3  # args.win_size
        tree_size = 256

        points = rrcf.shingle(sketch, size=shingle_size)
        # points = np.vstack([point for point in points])
        print("Points : ", points)
        n = points.shape[0]
        print("N: ", points.shape, n)
        sample_size_range = (n // tree_size, tree_size)
        print("Sample Size Range: ", sample_size_range)

        forest = []
        while len(forest) < num_trees:
            ixs = np.random.choice(n, size=sample_size_range,
                                   replace=False)
            trees = [rrcf.RCTree(points[ix], index_labels=ix)
                     for ix in ixs]
            forest.extend(trees)

        avg_codisp = pd.Series(0.0, index=np.arange(n))
        index = np.zeros(n)

        for tree in forest:
            codisp = pd.Series({leaf: tree.codisp(leaf)
                                for leaf in tree.leaves})
            avg_codisp[codisp.index] += codisp
            np.add.at(index, codisp.index.values, 1)

        avg_codisp /= index
        avg_codisp.index = sketch_vector.iloc[(shingle_size - 1):].index

    def robust_random_cut(self, sketch_vector):
        # Set tree parameters
        sketch = sketch_vector['sketch'].tolist()
        # pca =PCA(n_components=50)
        # sketch = pca.fit_transform(sketch)
        # print(sketch)
        # sketch = preprocessing.scale(sketch)
        # print(sketch)
        num_trees = 1000
        shingle_size = 24  #args.win_size
        tree_size = 24
        print("\nShingle Size= ", shingle_size, " Tree Size = ", tree_size, " Num Tree: ", num_trees)

        # Create a forest of empty trees
        forest = []
        for _ in range(num_trees):
            tree = rrcf.RCTree()
            forest.append(tree)

        # Use the "shingle" generator to create rolling window
        points = rrcf.shingle(sketch, size=shingle_size)

        # Create a dict to store anomaly score of each point
        avg_codisp = {}
        # For each shingle...
        for index, point in enumerate(points):
            # For each tree in the forest...
            if index % 50 == 0:
                print("Graph: ", index)
            for tree in forest:
                # If tree is above permitted size...
                if len(tree.leaves) > tree_size:
                    # Drop the oldest point (FIFO)
                    tree.forget_point(index - tree_size)
                # Insert the new point into the tree
                tree.insert_point(point, index=index)
                # Compute codisp on the new point...
                new_codisp = tree.codisp(index)
                # And take the average over all trees
                if not index in avg_codisp:
                    avg_codisp[index] = 0
                avg_codisp[index] += new_codisp / num_trees

        disp = pd.Series([avg_codisp[s] for s in avg_codisp])
        return disp

    def print_result(self, y_test, y_pred):
        target_names = ['Normal', 'Anomaly']
        print(metrics.classification_report(y_test, y_pred, target_names=target_names))
        print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
        # print("Confusion Matrix: \n", metrics.confusion_matrix(y_test, y_pred))

    def plot_2d_scatter(self, X, y_pred):
        # plt.subplot(221)
        pca = PCA(n_components=2).fit(X)
        pca_2d = pca.transform(X)
        # print(pca_3d)
        plt.figure(figsize=(6, 6))
        plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y_pred)
        plt.title("2D Scatter Plot")
        plt.show()

    def plot_3d_scatter(self, X, y_pred):
        pca = PCA(n_components=3).fit(X)
        pca_3d = pca.transform(X)
        colormap = get_cmap('viridis')
        colors = [rgb2hex(colormap(col)) for col in np.arange(0, 1.01, 1 / (2 - 1))]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax = Axes3D(fig)
        ax.scatter(pca_3d[:, 0], pca_3d[:, 1], pca_3d[:, 2], c=y_pred, s=50, cmap=mat.colors.ListedColormap(colors))
        plt.title('3D Scatter Plot')
        plt.show()

    def run_knn(self, sketch_vector):
        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        # X = sketch_vector['sketch'].tolist()
        knn = KNeighborsClassifier(n_neighbors=10)
        y_pred = cross_val_predict(knn, X, sketch_vector['anomaly'].tolist(), cv=5)
        return y_pred

    def run_nn(self, sketch_vector):
        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        # X = sketch_vector['sketch'].tolist()
        nn = MLPClassifier(alpha=1, max_iter=1000)
        y_pred = cross_val_predict(nn, X, sketch_vector['anomaly'].tolist(), cv=5)
        return y_pred

    def run_kmean(self, sketch_vector):
        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        X = sketch_vector['sketch'].tolist()
        y_pred = KMeans(n_clusters=1, random_state=10).fit_predict(X)
        # return  y_pred
        print("True: ", sketch_vector['anomaly'].tolist())
        print("Predicted: ",y_pred)
        print(" \n -- K Mean Results")
        self.print_result(sketch_vector['anomaly'].tolist(), y_pred)
        self.plot_2d_scatter(X, y_pred)
        # self.plot_3d_scatter(X, y_pred_rf)
        return  metrics.accuracy_score(sketch_vector['anomaly'].tolist(), y_pred), y_pred

    def run_random_forest(self, sketch_vector):
        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        # X = sketch_vector['sketch'].tolist()

        # estimator = {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 50, 'bootstrap': False}
        random_forest_classifier = RandomForestClassifier(n_estimators= 200, min_samples_split= 2, min_samples_leaf= 1, max_features='sqrt', max_depth= 50, bootstrap=False)
        y_pred_rf = cross_val_predict(random_forest_classifier, X, sketch_vector['anomaly'].tolist(), cv=5)
        return y_pred_rf
        # print(" \n -- Random Forest Results")
        # self.print_result(sketch_vector['anomaly'].tolist(), y_pred_rf)
        # return metrics.accuracy_score(sketch_vector['anomaly'].tolist(), y_pred_rf), y_pred_rf

    def run_decision_tree(self, sketch_vector):
        # print(sketch_vector['sketch'])
        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        # X = sketch_vector['sketch'].tolist()
        decision_tree_classifier = DecisionTreeClassifier()
        y_pred = cross_val_predict(decision_tree_classifier, X, sketch_vector['anomaly'].tolist(), cv=5)
        return  y_pred
        # print(" \n -- Decision Tree Results")
        # # print("Predicted: ", y_pred)
        # self.print_result(sketch_vector['anomaly'].tolist(), y_pred)
        # return metrics.accuracy_score(sketch_vector['anomaly'].tolist(), y_pred), y_pred

    def svm_parameter_tuning(self, train_X, train_y, nfold):
        print("\n\n Parameter Tuning ... ")
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1], 'kernel': ['rbf', 'linear']}
        grid_search = GridSearchCV(svm.SVC(), param_grid, cv=nfold, verbose=1)
        grid_search.fit(train_X, train_y)
        grid_search.best_params_
        return grid_search.best_params_, grid_search.best_estimator_

    def run_gradient_bosting(self, sketch_vector):
        gdc = gradient_boosting.GradientBoostingClassifier(n_estimators=200, max_depth=4,learning_rate=0.01);

        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        # X = sketch_vector['sketch'].tolist()
        y_pred_gdc = cross_val_predict(gdc, X, sketch_vector['anomaly'].tolist(), cv=5)
        return y_pred_gdc
        # print(" \n -- Gradient Boosting Results")
        # self.print_result(sketch_vector['anomaly'].tolist(), y_pred_gdc)
        # # self.plot_2d_scatter(X, y_pred_rf)
        # # self.plot_3d_scatter(X, y_pred_rf)
        # return metrics.accuracy_score(sketch_vector['anomaly'].tolist(), y_pred_gdc), y_pred_gdc

    def run_svm(self, sketch_vector):
        X = preprocessing.scale(sketch_vector['sketch'].tolist())
        # X = sketch_vector['sketch']
        # print(X)
        # param, estimator = self.svm_parameter_tuning(X, sketch_vector['anomaly'].tolist(), 10)
        # print("Best Estimator: ", estimator)
        estimator = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=0.01, kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)
        y_pred = cross_val_predict(estimator, X, sketch_vector['anomaly'].tolist(), cv=10)
        return y_pred

    def print_aggregate_result(self, results):
        # print("Results: ", results)
        a = []
        p = []
        r = []
        f = []
        for array in results:
            a.append(array[0])
            p.append(array[1])
            r.append(array[2])
            f.append(array[3])

        print("Accuracy: ", round(statistics.mean(a)*100, 2), '(+/- ', round(statistics.stdev(a)*100, 2), ')')
        print("Precision: ", round(statistics.mean(p)*100, 2), '(+/- ', round(statistics.stdev(p)*100, 2), ')')
        print("Recall: ", round(statistics.mean(r)*100, 2), '(+/- ', round(statistics.stdev(r)*100, 2), ')')
        print("F1 Score: ", round(statistics.mean(f)*100, 2), '(+/- ', round(statistics.stdev(f)*100, 2), ')')


    def print_topk_result(self, results):
        # print("Results: ", results)
        # results = {0: {100: [1.0, 0.96, 0.9795918367346939], 200: [1.0, 0.945, 0.9717223650385605], 300: [1.0, 0.92, 0.9583333333333334]}, 1: {100: [1.0, 0.96, 0.9795918367346939], 200: [1.0, 0.945, 0.9717223650385605], 300: [1.0, 0.9166666666666666, 0.9565217391304348]}, 2: {100: [1.0, 0.96, 0.9795918367346939], 200: [1.0, 0.96, 0.9795918367346939], 300: [1.0, 0.93, 0.9637305699481865]}, 3: {100: [1.0, 0.96, 0.9795918367346939], 200: [1.0, 0.955, 0.9769820971867007], 300: [1.0, 0.93, 0.9637305699481865]}, 4: {100: [1.0, 0.95, 0.9743589743589743], 200: [1.0, 0.93, 0.9637305699481865], 300: [1.0, 0.91, 0.9528795811518325]}}
        p100 = 0.0
        r100 = 0.0
        f100 = 0.0
        p200 = 0.0
        r200 = 0.0
        f200 = 0.0
        p300 = 0.0
        r300 = 0.0
        f300 = 0.0
        for top in results.keys():
            print("top" , results[top])
            for t in results[top].keys():
                if t == 100:
                    p100 += results[top][t][0]
                    r100 += results[top][t][1]
                    f100 += results[top][t][2]
                if t == 200:
                    p200 += results[top][t][0]
                    r200 += results[top][t][1]
                    f200 += results[top][t][2]
                if t == 300:
                    p300 += results[top][t][0]
                    r300 += results[top][t][1]
                    f300 += results[top][t][2]
        print("Precision 100:  ", p100 / 5)
        print("Recall 100: ", r100 / 5)
        print("F1 Score 100: ", f100 / 5)
        print("Precision 200:  ", p200 / 5)
        print("Recall 200: ", r200 / 5)
        print("F1 Score 200: ", f200 / 5)
        print("Precision 300:  ", p300 / 5)
        print("Recall 300: ", r300 / 5)
        print("F1 Score 300: ", f300 / 5)


    def anomaly_detection(self, sketch_vector, args):
        '''
        :param sketch_vector:
        :param args:
        :return:
        '''
        # print(sketch_vector)
        # sketch_vector = sketch_vector.sort_values(by='graphid', ascending=False)
        # print(sketch_vector)
        # # Supervised Learning ....
        true_anomalies = np.array(sketch_vector['anomaly'])
        # print("True: ", true_anomalies)
        # '''
        #  {'svm': 'Support Vector'}  #
        algo = {'svm':'Support Vector'}#, 'dt':'Decision Tree', 'rf':'Random Forest','knn':"K Nearest"}  #{        algo = {'dt':'Decision Tree', 'rf':'Random Forest',  'svm':'Support Vector'}#{'svm':'Support Vector', 'knn':"K Nearest", 'dt':'Decision Tree','rf':'Random Forest', 'gd':'Gradient Boosting'} #
        for alg in algo.keys():
            results = []
            print("\n", algo[alg])
            for i in range(0, 1):
                if alg == 'dt':
                    y_pred = self.run_decision_tree(sketch_vector)
                elif alg == 'rf':
                    y_pred = self.run_random_forest(sketch_vector)
                elif alg == 'knn':
                    y_pred = self.run_knn(sketch_vector)
                elif alg == 'nn':
                    y_pred = self.run_nn(sketch_vector)
                elif alg == 'gd':
                    y_pred = self.run_gradient_bosting(sketch_vector)
                elif alg == 'svm':
                    y_pred = self.run_svm(sketch_vector)
                # print("Predicted: ", y_pred)
                print(metrics.classification_report(true_anomalies, y_pred))
                print("Accuracy: ", metrics.accuracy_score(true_anomalies, y_pred))
                # print("True: ", true_anomalies)
                # print("Pred: ", y_pred)
                a = metrics.accuracy_score(true_anomalies, y_pred)
                p = metrics.precision_score(true_anomalies, y_pred, average='weighted')
                r = metrics.recall_score(true_anomalies, y_pred, average='weighted')
                f = metrics.f1_score(true_anomalies, y_pred, average='weighted')
                return a, p, r, f
            #     results.append([a,p,r,f])
            # self.print_aggregate_result(results)

    def sub2vec_run(self, args):
        results = []
        for sketch_file in glob.glob(
                'sketches/baseline/sub2vec/*_sub2vec_nci1_dim_128.csv'):
            print("\n\n ", sketch_file, " ---- Anomaly Detection")
            vector = self.read_sketch_baseline(sketch_file)
            a, p, r, f = self.anomaly_detection(vector, args)
            results.append([a, p, r, f])
        self.print_aggregate_result(results)

    def spotlight_run(self, args):
        results = []
        for sketch_file in glob.glob(
                'sketches/baseline/spotlight/*_splt_+nci1.csv'):
            print("\n\n ", sketch_file, " ---- Anomaly Detection")
            vector = self.read_sketch_baseline(sketch_file)
            a, p, r, f = self.anomaly_detection(vector, args)
            results.append([a, p, r, f])
        self.print_aggregate_result(results)
