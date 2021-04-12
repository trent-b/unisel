import os
import time
from scipy.io import loadmat
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from modAL.models import ActiveLearner
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def unisel(data_X, n_instances, seed):
    ss = StandardScaler()
    
    kmeans = KMeans(n_clusters=n_instances, n_init=1, random_state=seed)
    labels = kmeans.fit_predict(ss.fit_transform(data_X))
    unique_labels = list(set(labels))
    sample_idx = []
    
    for label in unique_labels:
        instance_idx = np.where(labels == label)[0]
        dist = np.sqrt(np.sum((ss.transform(data_X[instance_idx]) - 
                    kmeans.cluster_centers_[label])**2, axis=-1))
        sample_idx.append(instance_idx[np.argmin(dist)])
    
    nonsample_idx = list(set(np.arange(data_X.shape[0])) -
                     set(sample_idx))
    
    return sample_idx, nonsample_idx

def active_learning(model, data_X, data_y, sample_idx, n_iter):
    nonsample_idx = np.array(list(set(np.arange(data_X.shape[0])) - set(sample_idx)))
    learner = ActiveLearner(model, X_training=data_X[sample_idx], y_training=data_y[sample_idx])
    
    for al_idx in range(n_iter):                    
        # query for labels
        query_idx, _ = learner.query(data_X[nonsample_idx])
        
        # supply label for queried instance
        learner.teach(data_X[nonsample_idx][query_idx], data_y[nonsample_idx][query_idx])
        
        sample_idx.append(nonsample_idx[query_idx][0])
        nonsample_idx = np.array(list(set(np.arange(data_X.shape[0])) - set(sample_idx)))
    
    return sample_idx, nonsample_idx

if __name__ == '__main__':
    datasets = ['cover.mat',
                'http.mat',
                'mammography.mat',
                'musk.mat',
                'optdigits.mat',
                'pendigits.mat',
                'satimage-2.mat',
                'smtp.mat',
                'thyroid.mat',
                'vowels.mat']
    sample_sizes = [10, 50, 100, 500, 1000]
    n_repeats = 10
    seed = 0
    
    np.random.seed(seed)
    seeds = np.random.randint(0, np.iinfo(np.int32).max, size=n_repeats)
    
    for dataset in datasets:
        # load data
        data_file = os.path.join('data', dataset)
        
        try:
            data = loadmat(data_file)
        except:
            with h5py.File(data_file, 'r') as f:
                data = {'X':None, 'y':None}
                data['X'] = f['X'][:].T
                data['y'] = f['y'][:].T
        
        # create empty list to store results
        overall_results = []
        
        for sample_size in sample_sizes:
            print('\nstarting dataset', dataset, 'with sample size', sample_size)
            
            baseline_cm = []
            random_cm = []
            unisel_cm = []
            al_cm = []
            al_unisel_cm = []
            rand_n_selected_outliers = []
            rand_n_selected_normals = []
            unisel_n_selected_outliers = []
            unisel_n_selected_normals = []
            al_n_selected_outliers = []
            al_n_selected_normals = []
            al_unisel_n_selected_outliers = []
            al_unisel_n_selected_normals = []
            
            y = data['y'].ravel()
            
            # create results dictionary
            results = {'name':os.path.splitext(dataset)[0],
                       'n_obs':len(y),
                       'n_outliers':np.sum(y==1),
                       'n_samples':sample_size,
                       'baseline_cm':baseline_cm,
                       'rand_cm':random_cm,
                       'unisel_cm':unisel_cm,
                       'al_cm':al_cm,
                       'al_unisel_cm':al_unisel_cm,
                       'rand_n_selected_outliers':rand_n_selected_outliers,
                       'rand_n_selected_normals':rand_n_selected_normals,
                       'unisel_n_selected_outliers':unisel_n_selected_outliers,
                       'unisel_n_selected_normals':unisel_n_selected_normals,
                       'al_n_selected_outliers':al_n_selected_outliers,
                       'al_n_selected_normals':al_n_selected_normals,
                       'al_unisel_n_selected_outliers':al_unisel_n_selected_outliers,
                       'al_unisel_n_selected_normals':al_unisel_n_selected_normals}
            
            if sample_size > len(y):
                overall_results.append(results)
                continue
            
            for seed in tqdm(seeds):
                # shuffle data
                np.random.seed(seed)
                X = data['X']
                y = data['y'].ravel()
                X, X_test, y, y_test = train_test_split(X, y, test_size=0.10,
                                random_state=seed, shuffle=True, stratify=y)
                
                # define models
                rf = RandomForestClassifier(n_estimators=100, class_weight=None, 
                                            n_jobs=-1, random_state=seed)
                
                ####### Baseline #######
                # only run baseline for first case of sample_size since it is
                # independent of sample_size
                if sample_size == sample_sizes[0]:
                    # train and predict
                    rf.fit(X, y)
                    y_pred = rf.predict(X_test)
                    score = confusion_matrix(y_test, y_pred).ravel().tolist()          
                    baseline_cm.append(score)
                
                ####### Random Sampling #######
                sample_idx = np.random.choice(X.shape[0], size=sample_size,
                                              replace=False).tolist()
                nonsample_idx = list(set(np.arange(X.shape[0])) -
                                     set(sample_idx))
                
                # record selected sample labels
                rand_n_selected_normals.append(np.sum(y[sample_idx] == 0))
                rand_n_selected_outliers.append(np.sum(y[sample_idx] == 1))
                
                # train and predict
                rf.fit(X[sample_idx], y[sample_idx])
                y_pred = rf.predict(X_test)
                score = confusion_matrix(y_test, y_pred).ravel().tolist()          
                random_cm.append(score)
                
                ####### Unisel Method #######
                sample_idx, nonsample_idx = unisel(X, sample_size, seed)
                
                # record selected sample labels
                unisel_n_selected_normals.append(np.sum(y[sample_idx] == 0))
                unisel_n_selected_outliers.append(np.sum(y[sample_idx] == 1))
                
                # train and predict
                rf.fit(X[sample_idx], y[sample_idx])
                y_pred = rf.predict(X_test)
                score = confusion_matrix(y_test, y_pred).ravel().tolist()          
                unisel_cm.append(score)
                
                ####### Active Learning Method (random initialization) #######
                first_half_sample_size = int(np.floor(sample_size/2))
                second_half_sample_size = int(np.ceil(sample_size/2))
                sample_idx = np.random.choice(X.shape[0], size=first_half_sample_size,
                                              replace=False).tolist()                
                sample_idx, nonsample_idx = active_learning(rf, X, y, sample_idx,
                                                            n_iter=second_half_sample_size)
                
                # record selected sample labels
                al_n_selected_normals.append(np.sum(y[sample_idx] == 0))
                al_n_selected_outliers.append(np.sum(y[sample_idx] == 1))
                
                # train and predict
                rf.fit(X[sample_idx], y[sample_idx])
                y_pred = rf.predict(X_test)
                score = confusion_matrix(y_test, y_pred).ravel().tolist()          
                al_cm.append(score)
                
                ####### Active Learning Method (unisel initialization) #######
                sample_idx, _ = unisel(X, first_half_sample_size, seed)
                sample_idx, nonsample_idx = active_learning(rf, X, y, sample_idx,
                                                            n_iter=second_half_sample_size)
                
                # record selected sample labels
                al_unisel_n_selected_normals.append(np.sum(y[sample_idx] == 0))
                al_unisel_n_selected_outliers.append(np.sum(y[sample_idx] == 1))
                
                # train and predict
                rf.fit(X[sample_idx], y[sample_idx])
                y_pred = rf.predict(X_test)
                score = confusion_matrix(y_test, y_pred).ravel().tolist()          
                al_unisel_cm.append(score)
                
            results['baseline_cm'] = baseline_cm
            results['rand_cm'] = random_cm
            results['unisel_cm'] = unisel_cm
            results['al_cm'] = al_cm
            results['al_unisel_cm'] = al_unisel_cm
            results['rand_n_selected_normals'] = rand_n_selected_normals
            results['rand_n_selected_outliers'] = rand_n_selected_outliers
            results['unisel_n_selected_normals'] = unisel_n_selected_normals
            results['unisel_n_selected_outliers'] = unisel_n_selected_outliers
            results['al_n_selected_normals'] = al_n_selected_normals
            results['al_n_selected_outliers'] = al_n_selected_outliers
            results['al_unisel_n_selected_normals'] = al_unisel_n_selected_normals
            results['al_unisel_n_selected_outliers'] = al_unisel_n_selected_outliers
            
            overall_results.append(results)
        
        output_file = results['name'] + '_' + time.strftime("%Y%m%d-%H%M%S") + '.csv'
        output_file = os.path.join('output', output_file)
        pd.DataFrame(overall_results).to_csv(output_file, index=False)
