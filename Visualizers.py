from concurrent.futures import ThreadPoolExecutor
from time import time


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectPercentile,SelectFromModel
from sklearn.preprocessing import StandardScaler
from skimage.io import imread
from sklearn.random_projection import GaussianRandomProjection
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.features import PCA as YBPCA
from ScoreClusterAlgo import ScoreClusterAlgo
import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import resize
from os import path
"""    
    PCA
    ICA
    Randomized Projections
    Any other feature selection algorithm you desire
"""

RSTATE=1234567890





def VisualizePCA():
    dataset_path = "data/Australian_Rain/balanced_australian_rain.csv"
    basename=path.basename(dataset_path).replace(".csv","")
    target = 'RainTomorrow'
    samples=10000
    df = pd.read_csv(dataset_path).sample(n=samples, random_state=RSTATE)
    X=df.drop([target],axis=1).values
    y=df[target].values
    for projection in (2,3):
        outpath = f"analysis/images/YBPCA-{projection}D-{basename}.png"
        if projection==2:
            viz = YBPCA(scale=True,
                        classes=['Rain Tomorrow', 'No Rain Tomorrow'],
                        heatmap=True,
                        colors=['green', 'blue'],
                        projection=projection,
                        alpha=0.25,
                        size=(1080, 720))
        else:
            viz = YBPCA(scale=True,
                        classes=['Rain Tomorrow', 'No Rain Tomorrow'],
                        projection=projection,
                        alpha=0.25,
                        size=(1080, 720),
                        colors=['green', 'blue'])
        viz.fit_transform(X,y)
        viz.show(outpath=outpath,clear_figure=True)


def VisualizeKMClustering():
    dataset_path = "data/Australian_Rain/balanced_australian_rain.csv"
    basename=path.basename(dataset_path).replace(".csv","")
    target = 'RainTomorrow'
    samples=10000
    df = pd.read_csv(dataset_path).sample(n=samples, random_state=RSTATE)
    X=df.drop([target],axis=1).values
    y=df[target].values
    for metric in ['distortion','silhouette','calinski_harabasz']:
        outpath=f"analysis/images/YBKM-{metric}-{basename}.png"
        kmc = KMeans(n_clusters=2,random_state=RSTATE)
        viz = KElbowVisualizer(kmc,
                               k=X.shape[1],
                               metric=metric,
                               locate_elbow=True,
                               size=(1080, 720),
                               alpha=0.25,
                               colors=['green','blue'])
        viz.fit(X)
        viz.show(outpath=outpath,clear_figure=True)





def ClusterShow(X,y,n):
    kmc = KMeans(n)
    kmc.fit_transform(X)




def YBVizPCAImage():
    nsamples=10
    dataset_path="data/Dataset_Handwritten_English/english.csv"
    basename = path.basename(dataset_path).replace(".csv", "")
    df=pd.read_csv(dataset_path)
    image_data=pd.DataFrame([])
    label_data=[]
    count=0
    for label in df['label'].unique():
        _t=df.query(f'label == \'{label}\'').sample(n=nsamples,random_state=RSTATE)
        for row in _t.values:
            count+=1
            print(count)
            fpath=f"data/Dataset_Handwritten_English/{row[0]}"
            label = row[1]
            image = imread(fpath)
            image=resize(image,(450,600),anti_aliasing=True)  # 20x smaller
            image = rgb2gray(image)
            image = image.flatten()
            image=pd.Series(image,name=fpath)
            image_data=image_data.append(image)
            label_data.append(label)
    y=[ord(i)-48 for i in label_data]
    X=image_data
    for projection in (2,3):
        outpath = f"analysis/images/YBPCA-{projection}-{basename}-15x20.png"
        if projection==2:
            viz = YBPCA(scale=True,
                        classes=label_data,
                        heatmap=True,
                        colors=['green', 'blue'],
                        projection=projection,
                        alpha=0.25,
                        size=(1080, 720))
        else:
            viz = YBPCA(scale=True,
                        classes=label_data,
                        projection=projection,
                        alpha=0.25,
                        size=(1080, 720),
                        colors=['green', 'blue'])
        viz.fit_transform(X,y)
        viz.show(outpath=outpath,clear_figure=True)
        new_df = X.copy(deep=True)
        new_df['label'] = y
        new_df.to_csv("data/Dataset_Handwritten_English/flattened_images_450x600.csv",index=False)

def ViewPCAImage():
    def _proc_image(fpath,label):
        image = imread(fpath)
        image = rgb2gray(image)
        image = resize(image, (30, 40), anti_aliasing=True)  # 20x smaller
        image = image.flatten()
        image = pd.Series(image, name=fpath)
        return image,label
    nsamples=55
    dataset_path="data/Dataset_Handwritten_English/english.csv"
    basename = path.basename(dataset_path).replace(".csv", "")
    df=pd.read_csv(dataset_path)
    image_data=[]
    label_data=[]
    count=0
    futures=[]
    with ThreadPoolExecutor(8) as pool:
        for label in df['label'].unique():
            _t=df.query(f'label == \'{label}\'').sample(n=nsamples,random_state=RSTATE)
            for row in _t.values:
                fpath=f"data/Dataset_Handwritten_English/{row[0]}"
                futures.append(pool.submit(_proc_image,fpath,label))
    print(len(futures))
    for i,f in enumerate(futures):
        print(i)
        if f.done():
            data,label=f.result()
            image_data.append(data.values)
            label_data.append(label)
        else:
            print(f.exception())
    image_data=pd.DataFrame(image_data)
    new_df = image_data.copy(deep=True)
    new_df['label'] = label_data
    new_df.to_csv("data/Dataset_Handwritten_English/flattened_images_30x40.csv", index=False)
    new_df=None
    y=[ord(i)-48 for i in label_data]
    components_stats = {"percent_variance":[],"n_components":[],"explained_variance":[]}
    for ii in range(1,7):
        percent_variance=0.15*ii
        suptitle = f"{percent_variance * 100}% of Variance"
        print(suptitle)
        pca = PCA(percent_variance)
        pca_X=pca.fit_transform(image_data)
        ncomponents = len(pca.components_)
        components_stats["percent_variance"].append(percent_variance)
        components_stats["n_components"].append(ncomponents)
        components_stats["explained_variance"].append(pca.explained_variance_ratio_)
        cols=5
        rows=1+ncomponents//cols
        fig = plt.figure(figsize=(rows,cols))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
        for i in range(1,ncomponents):
            ax = fig.add_subplot(cols, rows, i + 1, xticks=[], yticks=[])
            ##ax.imshow(np.reshape(pca.components_[i, :], (900, 1200)), cmap=plt.cm.bone, interpolation='nearest')
            ax.imshow(np.reshape(pca.components_[i, :], (30, 40)), cmap=plt.cm.Greys, interpolation='nearest')
        plt.suptitle(suptitle)
        ##plt.savefig(f"analysis/images/PCA-Images-{percent_variance:0.2f}-{ncomponents}components.png")
        plt.show()
    #pd.DataFrame(components_stats).to_csv("analysis/pca_stats.csv",index=False)

def show_images(image_data):
    cols = 10
    rows = 1+len(image_data)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12), subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.01, wspace=0.01))
    for i,ax in enumerate(axes.flat):
        if i>=len(image_data):
            break
        ax.imshow(image_data[i].reshape(4, 4))
    plt.show()

if __name__=="__main__":
    path1="data/Australian_Rain/balanced_australian_rain.csv"
    path2=""
    #VisualizePCA()
    #VisualizeKMClustering()
    #YBVizPCAImage()
    ViewPCAImage()
