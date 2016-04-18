import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from skimage.transform import downscale_local_mean
from sklearn.cluster import DBSCAN
import itertools
import os.path
from load import *
from test import *


def load_data():
    # generate filenames
    files = []
    for i in range(1,4):
        for j in range(1,5):
            prefix = 'data/AMG%d_exp%d'%(i,j)
            files.append( (prefix+'.tif', prefix+'.zip') )
        
    # load data
    data = []
    for i,(s,r) in enumerate(files):
        if i==8: # lolwtf
            data.append((load_stack(s), load_rois(r, 512, 512, xdisp=9, ydisp=-1)))
        else:
            data.append((load_stack(s), load_rois(r, 512, 512)))
    return data

def get_centroids(rois):
    new_rois = np.zeros(rois.shape)
    for i,r in enumerate(rois):
        x,y = np.where(r!=0)
        x,y = int(x.mean()), int(y.mean())
        new_rois[i][x][y] = 1
    return new_rois

def preprocess(data):
    for i in range(len(data)):
        stk,roi = data[i]
        #stk = downscale_local_mean(stk, (1,2,2)) 
        #roi = downscale_local_mean(roi, (1,2,2))
        stk_max = stk.max(axis=0)
        stk_std = np.std(stk, axis=0)
        stk_median = np.median(stk, axis=0)
        roi_centroids = get_centroids(roi)
        data[i] = ([stk_std],roi)
    return data

def window(x, s, label=False):
    p = np.pad(x,s/2,'constant') if label else np.pad(x,s/2,'edge')
    g = lambda i,j: (p[i:i+s,j:j+s]).flatten()
    l = lambda i,j: p[i+s/2,j+s/2]
    h,w = p.shape
    if label:
        return [], [l(i,j) for i in range(h-s) for j in range(w-s)]
    else:
        return [g(i,j) for i in range(h-s) for j in range(w-s)], []
            
def stk_to_feat_vecs(stk, roi, s):
    """converts one (image stack, roi stack) tuple to matrix of 
    feature vectors and list of labels"""
    rmax = roi.max(axis=0)
    _,labels = window(rmax,s,True)
    all_features = []
    all_labels = []
    for i in stk:
        feature_vecs,_ = window(i,s,False)
        all_features.extend(feature_vecs)
        all_labels.extend(labels)
    assert(len(all_features) == len(all_labels))
    return all_features, all_labels

def training_stks_to_feat_vecs(data,s):
    """merges list of (image stack, roi stack) tuples to single matrix 
    of feature vectors and list of labels"""
    all_features = []
    all_labels = []
    for stk,roi in data:
        feat, lab = stk_to_feat_vecs(stk,roi,s)
        all_features.extend(feat)
        all_labels.extend(lab)
    assert(len(all_features) == len(all_labels))
    return all_features, all_labels   

def labels_to_stk(labels):
    """converts one stacks worth of labels back to array of images"""
    results = np.zeros((len(labels)/(512*512), 512, 512))
    step = 512*512
    for i,s in enumerate(range(0,len(labels),step)):
        results[i,:,:] = np.array(labels[s:s+step]).reshape((512,512))
    return results

def knn_predict_to_points(labels):
    pts = []
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            if labels[x,y] == 1:
                pts.append((x/512., y/512.))
    return pts


def knn_to_predictions(pred, eps, min_samples, radius):
    pred = pred.squeeze()
    pred_pts = knn_predict_to_points(pred)
    dbscan = DBSCAN(metric='manhattan', eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(pred_pts)
    z = zip(clusters, pred_pts)
    centroids = []
    for s in set(clusters):
        if s == -1: continue
        pts_in_cluster = filter(lambda x: x[0] == s, z)
        cluster, pts = zip(*pts_in_cluster)
        xs, ys = zip(*pts)
        xmean, ymean = np.mean(xs), np.mean(ys)
        centroids.append((xmean*512,ymean*512))
    final_predictions = np.zeros((len(centroids),512,512))
    for i,(cx,cy) in enumerate(centroids):
        for x in range(512):
            for y in range(512):
                if abs(x-cx) > radius or abs(y-cy) > radius: continue
                elif np.sqrt((y-cy)**2 + (x-cx)**2) <= radius:
                    final_predictions[i,x,y] = 1
    return final_predictions

def train_threshold_hyperparameters(pred, actual):
    best_parameters = (0,0,0,np.array([0]))
    best_score = 0
    eps_test = [0.00794] #np.logspace(-2.5,-1.5,6)
    min_samples_test = [32] #np.linspace(20,40,6)
    radius_test = [7.8] #np.linspace(3,15,6)
    i = 0
    for eps, min_samples, radius in itertools.product(eps_test, min_samples_test, radius_test):
        print eps, min_samples, radius
        predictions = knn_to_predictions(pred, eps, min_samples, radius)
        s = Score(None, None, [actual], [predictions])
        score = s.total_f1_score
        if score > best_score:
            best_parameters = (eps, min_samples, radius, predictions)
            best_score = score
    return best_parameters

def disp_stk(s):
    M = np.clip( (s-900)/500, 0, 1 )
    M**=0.5
    return make_RGB(M)
    #R = rois.max(axis=0)
    #K = np.zeros((512,512,3))
    #K[:,:,0] = K[:,:,1] = K[:,:,2] = M*0.4
    #K[:,:,2] = np.maximum(M,R)

def make_RGB(a):
    p = np.zeros((a.shape[0],a.shape[1],3))
    p[:,:,0] = a
    p[:,:,1] = a
    p[:,:,2] = a
    return p

def main():
    n_neighbors = int(sys.argv[1])
    window_size = int(sys.argv[2])
    num_principle_components = int(sys.argv[3])
    if len(sys.argv) > 4:
        plot = (sys.argv[4] == 'plot')
    else:
        plot = False

    print "Num neighbors: {}".format(n_neighbors)
    print "Window size: {}".format(window_size)
    print "Number Principle Components: {}".format(num_principle_components)
    
    print "Loading data"
    data = load_data()

    print "Preprocessing data"
    training_data = preprocess(data[0:8])
    test_data = preprocess(data[10:12])
    validation_data = preprocess([data[8]])[0]
    
    print "formatting training set"
    train_feat_vec, train_labels = training_stks_to_feat_vecs(training_data, window_size)
    print "{} training feature vectors length {}".format(len(train_feat_vec), train_feat_vec[0].shape)

    print "performing PCA"
    pca_filename = "pca_" + str(n_neighbors) + "_" + str(window_size) + "_" + str(num_principle_components) + ".pickle"
    if os.path.isfile("./"+pca_filename):
        pca = pickle.load(open(pca_filename,'rb'))
    else:
        pca = PCA(n_components=num_principle_components, copy=True)
        pca.fit(train_feat_vec)
        pickle.dump(pca,open(pca_filename,'wb'))
    explain_variance = pca.explained_variance_ratio_
    print "Explained Variance: {}".format(str(explain_variance))
    
    print "Reducing dimension of training set"
    train_feat_pca = pca.transform(train_feat_vec)
    print "{} training feature vectors post-pca length {}".format(len(train_feat_pca), train_feat_pca[0].shape)
    plt.figure()
    plt.scatter(train_feat_pca[:,0], train_feat_pca[:,1], c=['r' if t==1 else 'b' for t in train_labels], marker='.')
    if plot:
        plt.show()
    plt.savefig("train_pca.png")

    print "training (or loading) KNN classifier"
    knn_filename = "knn_" + str(n_neighbors) + "_" + str(window_size) + "_" + str(num_principle_components) + ".pickle"
    if os.path.isfile("./"+knn_filename):
        knn = pickle.load(open(knn_filename,'rb'))
    else:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1)
        knn.fit(train_feat_pca, train_labels)
        pickle.dump(knn,open(knn_filename,'wb'))

    val_filename = "val_" + str(n_neighbors) + "_" + str(window_size) + "_" + str(num_principle_components) + ".pickle"
    if os.path.isfile("./"+val_filename):
        print "loading knn'd validation set"
        val_stked_labels = pickle.load(open(val_filename, 'rb'))
    else:
        print "Formatting Validation Set"
        val_feat_vec, val_labels = stk_to_feat_vecs(validation_data[0], validation_data[1], window_size)
        print "{} validation feature vectors length {}".format(len(val_feat_vec), val_feat_vec[0].shape)
        
        print "Reducing dimension of validation set"
        val_feat_pca = pca.transform(val_feat_vec)
        print "{} validation feature vectors post-pca length {}".format(len(val_feat_pca), val_feat_pca[0].shape)
        
        print "KNN predicting validation set"
        val_pred_labels = knn.predict(val_feat_pca)
        val_stked_labels = labels_to_stk(val_pred_labels)
        pickle.dump(val_stked_labels,open(val_filename,'wb'))

    print "Training Threshold Hyperparameters"
    eps, min_samples, radius, predictions = train_threshold_hyperparameters(val_stked_labels, validation_data[1])
    print "Best eps: {} Best min_samples: {}, best radius: {}".format(eps, min_samples, radius)
    plt.figure()
    plt.imshow(np.max(validation_data[0], axis=0), cmap="Greys")
    plt.imshow(np.max(validation_data[1], axis=0), alpha=0.1)
    plt.imshow(np.max(predictions, axis=0), alpha=0.1, cmap=plt.get_cmap('spring'))
    plt.title("Validation")

    test_num=0
    test_predictions = []
    for test_stk,test_roi in test_data:
        test_num += 1
        print "formatting test set"
        feat_vec, actual_labels = stk_to_feat_vecs(test_stk, test_roi, window_size)
        print "{} test feature vectors length {}".format(len(feat_vec), feat_vec[0].shape)
        
        print "Reducing the dimensions of test set"
        feat_pca = pca.transform(feat_vec)
        print "{} test feature vectors post-pca length {}".format(len(feat_pca), feat_pca[0].shape)
        plt.figure()
        plt.scatter(feat_pca[:,0], feat_pca[:,1], c=['r' if t==1 else 'b' for t in actual_labels], marker='.')
        plt.savefig("test"+str(test_num)+"_pca.png")

        print "KNN predicting test set"
        test_filename = lambda n: "test" + str(n) + "_" + str(n_neighbors) + "_" + str(window_size) + "_" + str(num_principle_components) + ".pickle"
        if os.path.isfile("./"+test_filename(test_num)):
            pred_stked_labels = pickle.load(open(test_filename(test_num),'rb'))
        else:
            pred_labels = knn.predict(feat_pca)
            pred_stked_labels = labels_to_stk(pred_labels)
            pickle.dump(pred_stked_labels, open(test_filename(test_num), 'wb'))

        print "Clustering test set"
        predictions = knn_to_predictions(pred_stked_labels, eps, min_samples, radius)
        test_predictions.append(predictions)
        plt.figure()
        plt.imshow(np.max(test_stk, axis=0), cmap="Greys")
        plt.imshow(np.max(test_roi, axis=0), alpha=0.2)
        plt.imshow(np.max(predictions, axis=0), alpha=0.2, cmap=plt.get_cmap('spring'))
        plt.title("Test" + str(test_num))


    actual_labels = [t[1] for t in test_data]
    s = Score(None, None, actual_labels, test_predictions)
    print str(s)
    s.plot()


if __name__ == "__main__":
    main()
