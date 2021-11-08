import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA,FastICA
from matplotlib import cm
from sklearn.metrics import rand_score, silhouette_score, fowlkes_mallows_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from concurrent.futures import ThreadPoolExecutor
from sklearn.random_projection import  GaussianRandomProjection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,mutual_info_classif,chi2

def make_graph(x_points,predictions,title,dim=3):
    assert dim in (2,3)
    n_clusters = len(set(predictions))
    color_map = cm.get_cmap('coolwarm', n_clusters)
    x_points = PCA(dim).fit_transform(x_points)
    if dim == 3:
        fig = plt.figure(figsize=(12.8,9.6))
        ax = fig.add_subplot(projection='3d')
        ax.set_title(title+" 3D")
        for (x, y, z), label in zip(x_points, predictions):
            color = color_map(label)
            ax.scatter(x,y,z,color=color,alpha=1.0)
    elif dim==2:
        fig = plt.figure(figsize=(12.8,9.6))
        ax = fig.add_subplot()
        ax.set_title(title+" 2D")
        for (x, y), label in zip(x_points, predictions):
            color = color_map(label)
            ax.scatter(x,y,color=color,alpha=0.75)
    fpath_out = "analysis/images/" + title.replace(" ","_")+f"_{dim}D"+".png"
    fig.savefig(fpath_out)
    plt.show()


def viz_km_cluster(X, y, n_clusters,title):
    kmc = KMeans(n_clusters).fit(X)
    predictions = kmc.predict(X)
    #rscore = rand_score(predictions, y)
    #silscore = silhouette_score(X, kmc.labels_, metric='euclidean')
    #fmscore = fowlkes_mallows_score(predictions, y )
    make_graph(x_points=X,
               predictions=predictions,
               title=title)


def viz_em_cluster(X,y,n_clusters,title):
    gmix = GaussianMixture(n_clusters).fit(X)
    predictions = gmix.predict(X)
    #rscore = rand_score(predictions, y)
    #silscore = silhouette_score(X, kmc.labels_, metric='euclidean')
    #fmscore = fowlkes_mallows_score(predictions, y )
    make_graph(x_points=X,
               predictions=predictions,
               title=title)


def viz_pca(X,y,variance,title):
    x_points=PCA(variance).fit_transform(X)
    make_graph(x_points=x_points,
               predictions=y,
               title=title)


def viz_ica(X,y,components,title):
    x_points=FastICA(components).fit_transform(X)
    make_graph(x_points=x_points,
               predictions=y,
               title=title)


def viz_rand_proj(X,y,components,title):
    x_points = GaussianRandomProjection(components).fit_transform(X)
    make_graph(x_points=x_points,
               predictions=y,
               title=title)

def viz_sel_k_best(X,y,k,title):
    x_points = SelectKBest().fit_transform(X,y)
    make_graph(x_points=x_points,
               predictions=y,
               title=title)



def ScoreKM(pipe:Pipeline, X, y)->dict:
    predictions = pipe.predict(X)
    rscore = rand_score(predictions, y)
    silscore = silhouette_score(X, pipe['cluster'].labels_, metric='euclidean')
    fmscore = fowlkes_mallows_score(predictions, y)
    chscore = calinski_harabasz_score(X,pipe['cluster'].labels_)
    return {
        "rscore":rscore,
        "silscore":silscore,
        "fmscore":fmscore,
        "chscore":chscore
    }

def ScoreGM(pipe:GaussianMixture, X, y)-> dict:
    predictions = pipe.predict(X)
    rscore = rand_score(predictions, y)
    fmscore = fowlkes_mallows_score(predictions, y)
    return {
        "rscore": rscore,
        "fmscore": fmscore,
    }




def RunOnlyVizGraphs():
    df1 = pd.read_csv("data/Australian_Rain/balanced_australian_rain.csv").sample(n=3000)
    df2 = pd.read_csv("data/Dataset_Handwritten_English/flattened_images_30x40.csv")

    _y1 = df1.pop('RainTomorrow')
    _x1 = df1.copy(deep=True)

    _y2 = df2.pop('label')
    _x2 = df2.copy(deep=True)

    _y1_unique_labels = _y1.unique().shape[0]
    _y2_unique_labels = _y2.unique().shape[0]

    with ThreadPoolExecutor(2) as pool:
        _t1 = pool.submit(StandardScaler().fit_transform, _x1)
        _t2 = pool.submit(StandardScaler().fit_transform, _x2)

    _x1 = _t1.result()
    _x2 = _t2.result()

    p1 = (_x1, _y1, _y1_unique_labels)
    p2 = (_x2, _y2, _y2_unique_labels)

    viz_km_cluster(*p1, "Australian Rain Clustered Only")
    viz_km_cluster(*p2, "English Chars Clustered Only")
    viz_em_cluster(*p1, "Australian Rain E-Max Clustered Only")
    viz_em_cluster(*p2, "English Chars E-Max Clustered Only")
    viz_pca(*p1, "Australian Rain PCA Only")
    viz_pca(*p2, "English Chars PCA Only")
    viz_ica(*p1, "Australian Rain ICA Only")
    viz_ica(*p2, "English Chars ICA Only")
    viz_rand_proj(*p1, "Australian Rain R-Project Only")
    viz_rand_proj(*p2, "English Chars R-Project Only")
    viz_sel_k_best(*p1, "Australian Rain Sel-K-Best Only")
    viz_sel_k_best(*p2, "English Chars Sel-K-Best Only")





def Do16Combinations():
    handwritten_char_labels = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5',
        6: '6', 7: '7', 8: '8', 9: '9', 17: 'A', 18: 'B',
        19: 'C', 20: 'D', 21: 'E', 22: 'F', 23: 'G',
        24: 'H', 25: 'I', 26: 'J', 27: 'K', 28: 'L',
        29: 'M', 30: 'N', 31: 'O', 32: 'P', 33: 'Q',
        34: 'R', 35: 'S', 36: 'T', 37: 'U', 38: 'V',
        39: 'W', 40: 'X', 41: 'Y', 42: 'Z', 49: 'a',
        50: 'b', 51: 'c', 52: 'd', 53: 'e', 54: 'f',
        55: 'g', 56: 'h', 57: 'i', 58: 'j', 59: 'k',
        60: 'l', 61: 'm', 62: 'n', 63: 'o', 64: 'p',
        65: 'q', 66: 'r', 67: 's', 68: 't', 69: 'u',
        70: 'v', 71: 'w', 72: 'x', 73: 'y', 74: 'z'
    }


    df1 = pd.read_csv("data/Australian_Rain/balanced_australian_rain.csv").sample(n=3000)
    df2 = pd.read_csv("data/Dataset_Handwritten_English/flattened_images_30x40.csv")

    _y1 = df1.pop('RainTomorrow')
    _x1 = df1.copy(deep=True)

    _y2 = df2.pop('label')
    _x2 = df2.copy(deep=True)

    _y1_unique_labels = _y1.unique().shape[0]
    _y2_unique_labels = _y2.unique().shape[0]

    pipelines = {
        "aus-rain":{
            "pca+kmeans": Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(0.85)),
                ("cluster",KMeans(_y1_unique_labels))
            ]),
            "pca+emax":Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(0.85)),
                ("cluster",GaussianMixture(_y1_unique_labels))
            ]),
            "ica+kmeans":Pipeline([
                ("scale", StandardScaler()),
                ("ica", FastICA()),
                ("cluster",KMeans(_y1_unique_labels))
            ]),
            "ica+emax":Pipeline([
                ("scale", StandardScaler()),
                ("ica", FastICA()),
                ("cluster",GaussianMixture(_y1_unique_labels))
            ]),
            "rproj+kmeans":Pipeline([
                ("scale", StandardScaler()),
                ("rproj", GaussianRandomProjection(n_components=int(_x1.shape[1]*0.35))), # reduce features by 65%
                ("cluster",KMeans(_y1_unique_labels))
            ]),
            "rproj+emax": Pipeline([
                ("scale", StandardScaler()),
                ("rproj", GaussianRandomProjection(n_components=int(_x1.shape[1]*0.35))),  # reduce features by 65%
                ("cluster", GaussianMixture(_y1_unique_labels))
            ]),
            "kbest+kmeans":Pipeline([
                ("scale", StandardScaler()),
                ("kbest", SelectKBest(k=int(_x1.shape[1]*0.35))),  # top 35% of features
                ("cluster",KMeans(_y1_unique_labels))
            ]),
            "kbest+emax": Pipeline([
                ("scale", StandardScaler()),
                ("kbest", SelectKBest(k=int(_x1.shape[1] * 0.35))),  # top 35% of features
                ("cluster", GaussianMixture(_y1_unique_labels))
            ])
        },
        "english-chars":{
            "pca+kmeans": Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(0.85)),
                ("cluster",KMeans(_y2_unique_labels))
            ]),
            "pca+emax":Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(0.85)),
                ("cluster",GaussianMixture(_y2_unique_labels))
            ]),
            "ica+kmeans":Pipeline([
                ("scale", StandardScaler()),
                ("ica", FastICA(max_iter=500)),
                ("cluster",KMeans(_y2_unique_labels))
            ]),
            "ica+emax":Pipeline([
                ("scale", StandardScaler()),
                ("ica", FastICA(max_iter=500)),
                ("cluster",GaussianMixture(_y2_unique_labels))
            ]),
            "rproj+kmeans":Pipeline([
                ("scale", StandardScaler()),
                ("rproj", GaussianRandomProjection(n_components=int(_x2.shape[1]*0.35))),  # reduce features by 65%
                ("cluster",KMeans(_y2_unique_labels))
            ]),
            "rproj+emax": Pipeline([
                ("scale", StandardScaler()),
                ("rproj", GaussianRandomProjection(n_components=int(_x2.shape[1]*0.35))),  # reduce features by 65%
                ("cluster", GaussianMixture(_y2_unique_labels))
            ]),
            "kbest+kmeans":Pipeline([
                ("scale", StandardScaler()),
                ("kbest", SelectKBest(k=int(_x2.shape[1]*0.35))),  # top 35% of features
                ("cluster",KMeans(_y2_unique_labels))
            ]),
            "kbest+emax": Pipeline([
                ("scale", StandardScaler()),
                ("kbest", SelectKBest(k=int(_x2.shape[1] * 0.35))),  # top 35% of features
                ("cluster", GaussianMixture(_y2_unique_labels))
            ])
        }
    }
    scores={"aus-rain":{},"english-chars":{}}
    for dataset,pipes in pipelines.items():
        for pipename, pipe in pipes.items():
            subject=f"{dataset}-{pipename}"
            print(f"Working: {subject}")
            if dataset=="aus-rain":
                X, y = _x1, _y1
            elif dataset=="english-chars":
                X, y = _x2, _y2
            pipe.fit(X,y)
            predictions = pipe.predict(X)
            #make_graph(x_points=X,
            #           predictions=predictions,
            #           title=subject,
            #           dim=3)
            #make_graph(x_points=X,
            #           predictions=predictions,
            #           title=subject,
            #           dim=2)
            if "kmeans" in pipename:
                score=ScoreKM(pipe,X,y)
            elif "emax" in pipename:
                score=ScoreGM(pipe,X,y)
            scores[dataset]|=score
            print(f"Done: {subject} {score}")
    return scores




def RunSimpleDimReduceToNNTasks():

    df1 = pd.read_csv("data/Australian_Rain/balanced_australian_rain.csv").sample(n=10000)

    y = df1.pop('RainTomorrow')
    X = df1.copy(deep=True)

    _y_unique_labels = y.unique().shape[0]
    X_train,X_test,y_train,y_test = train_test_split(X,y)


    pipelines = {
        "aus-rain":{
            "pca":Pipeline([
                ("scale", StandardScaler()),
                ("pca", PCA(0.85)),
                ("model",MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,),max_iter=5000))
            ]),
            "ica":Pipeline([
                ("scale", StandardScaler()),
                ("pca", FastICA(max_iter=500)),
                ("model",MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,),max_iter=5000))
            ]),
            "rand_proj":Pipeline([
                ("scale", StandardScaler()),
                ("rproj", GaussianRandomProjection(n_components=int(X_train.shape[1] * 0.35))),  # reduce features by 65%
                ("model",MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,),max_iter=5000))
            ]),
            "sel-kbest":Pipeline([
                ("scale", StandardScaler()),
                ("pca", SelectKBest(k=int(X_train.shape[1] * 0.75))),  # top 75% of features
                ("model",MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,),max_iter=5000))
            ])
        }
    }
    for dataset,pipes in pipelines.items():
        for pipename, pipe in pipes.items():
            pipe.fit(X_train,y_train)
            score=pipe.score(X_test,y_test)
            print(f"{dataset}_{pipename} Score={score}")




def RunFeatureReductionPermutationsTesting(dataset=None):


    if dataset=="eng-char":
        df1 = pd.read_csv("data/Dataset_Handwritten_English/flattened_images_30x40.csv")
        y = df1.pop('label')
        X = df1.copy(deep=True)
    else:
        df1 = pd.read_csv("data/Australian_Rain/balanced_australian_rain.csv").sample(n=10000)
        y = df1.pop('RainTomorrow')
        X = df1.copy(deep=True)

    _y_unique_labels = y.unique().shape[0]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
    multipliers = (0.15, 0.3, 0.5, 0.65, 0.8, 0.95)
    pca_components = multipliers + (None,2,3)
    ica_components = [int(X_train.shape[1]*i) for i in multipliers] + [None,]
    gaussian_components = [int(X_train.shape[1]*i) for i in multipliers]
    k_values = [int(X_train.shape[1]*i) for i in multipliers]
    sel_k_scoring = (
        f_classif,
        mutual_info_classif,
        #chi2
    )
    make_pca_pipeline = lambda n_components: Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=n_components)),
        ("model",MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,),max_iter=5000))
    ])
    make_ica_pipeline = lambda n_components: Pipeline([
        ("scale", StandardScaler()),
        ("pca", FastICA(n_components=n_components, max_iter=500)),
        ("model",MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,),max_iter=5000))
    ])
    make_rp_pipeline = lambda n_components: Pipeline([
        ("scale", StandardScaler()),
        ("pca", GaussianRandomProjection(n_components=n_components)),
        ("model", MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,), max_iter=5000))
    ])
    make_kb_pipeline = lambda k,score_f: Pipeline([
        ("scale", StandardScaler()),
        ("pca", SelectKBest(score_func=score_f, k=k)),
        ("model", MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,), max_iter=5000))
    ])

    pipelines= {}
    for _c in pca_components:
        pipelines.update({f"pca-n={_c}-nn":make_pca_pipeline(_c)})
    for _c in ica_components:
        pipelines.update({f"ica-n={_c}-nn": make_ica_pipeline(_c)})
    for _c in gaussian_components:
        pipelines.update({f"rp-n={_c}-nn": make_rp_pipeline(_c)})
    for _k in k_values:
        for _f in sel_k_scoring:
            pipelines.update({f"kb-k={_k}-f={_f.__name__}-nn": make_kb_pipeline(_k,_f)})


    futures= []
    with ThreadPoolExecutor(TREADS) as pool:
        for name,pipe in pipelines.items():
            futures.append(pool.submit(fit_score_wrapper,pipe,X_train, X_test, y_train, y_test, name))

    results = {}
    for f in futures:
        results.update(f.result())

    return results

def fit_score_wrapper(pipe, X_train, X_test, y_train, y_test, description):
    try:
        print(f"running {description}")
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        return {description:score}
    except Exception as e:
        print(f"Error with {description}: {e}")
        return {}


def RunClusteringPermutationsTesting(dataset=None):

    if dataset=="eng-char":
        df1 = pd.read_csv("data/Dataset_Handwritten_English/flattened_images_30x40.csv")
        y = df1.pop('label')
        X = df1.copy(deep=True)
    else:
        df1 = pd.read_csv("data/Australian_Rain/balanced_australian_rain.csv").sample(n=10000)
        y = df1.pop('RainTomorrow')
        X = df1.copy(deep=True)


    n_clusters = y.unique().shape[0]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
    n_clusters =  y.unique().shape[0]
    pipelines={
        "kmeans-nn":Pipeline([
            ("scale", StandardScaler()),
            ("kmeans", KMeans(n_clusters)),
            ("model", MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,), max_iter=5000))
        ]),
        "emax-nn":Pipeline([
            ("scale", StandardScaler()),
            ("emax", GMMWrapper(n_clusters)),
            ("model", MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=(150,), max_iter=5000))
        ])
    }



    futures= []
    with ThreadPoolExecutor(TREADS) as pool:
        for name,pipe in pipelines.items():
            futures.append(pool.submit(fit_score_wrapper,pipe,X_train, X_test, y_train, y_test, name))


    results = {}
    for f in futures:
        results.update(f.result())

    return results

class GMMWrapper(GaussianMixture):
    def __init__(self,n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10):
        super().__init__(n_components=n_components, covariance_type=covariance_type, tol=tol, reg_covar=reg_covar, max_iter=max_iter, n_init=n_init, init_params=init_params, weights_init=weights_init, means_init=means_init, precisions_init=precisions_init, random_state=random_state, warm_start=warm_start, verbose=verbose, verbose_interval=verbose_interval)
    def transform(self,*args,**kwargs):
        """
            So... what i want is a representation of how likely a given sample is going to be so come from any specifc n_cluster gaussians
            there is not transform method for this, because its not really a model.
            So what im doing instead, is getting the output of get_proba, which outputs basically what we would expect for a transform..
            for each sample it gives you an array of n_cluster length where each element in the array is the liklihood that the sample is part of that gaussian
        """
        return self.predict_proba(*args,**kwargs)



if __name__=="__main__":
    import json
    TREADS=8
    results=Do16Combinations()
    json.dump(results, open("16-resutls-scoring.json", "w"), indent=4)
    #results={
    #    "aus-rain": RunFeatureReductionPermutationsTesting()
    #               | RunClusteringPermutationsTesting(),
    #    "eng-char": RunFeatureReductionPermutationsTesting("eng-char")
    #               | RunClusteringPermutationsTesting("eng-char")
    #}
    #json.dump(results, open("nn_results_clustering.json", "w"), indent=4)

