'''Main class'''
from __future__ import print_function
import argparse
import re
from .Parser import read_training as rt
from .Parser import read_testing as rtest
from .Chemoinformatics import fingerprints as fp
from .Chemoinformatics import chemofeatures as cf
from .Chemoinformatics import experimental as exp
from .Analytics import cluster as cl
from .Chemoinformatics import user as usr
from .Parser import build_training as bt
from .Parser import build_testing as btest
from .Train import train_model as tm
from .Analytics import cross_validate as cv
from .Analytics import confirm as cm
import pickle
import numpy as np
from .Distance import distance as ds
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from .PubChemUtils import pubchempy_utils as pcp
from copy import deepcopy
import sys, os, glob
import warnings
from collections import OrderedDict

class Placeholder(object):
    pass



def verbose_print(verbose, line):
    if verbose:
        print(line)



def define_random_seed(args):
    '''Define random seed'''
    if args.random:
        '''
        Users may add a random seed. This helps reproducibility in parts of
        the machine learning that require reandomization
        :param seed: int provided in the program arguments
        '''
        assert args.random > 0, "Random seed must be a positive integer"
        assert isinstance(args.random, int), "Random seed must be a positive integer"
        seed = args.random
        np.random.seed(seed)
        return seed
    else:
        return False



def dictitems(dict):
    if sys.version_info[0] >= 3:
        return dict.items()
    else:
        return dict.iteritems()



def define_proxy(args):
    '''Define the proxy URL'''
    if args.proxy:
        '''
        Users may add a proxy for interacting with the web.
        :param proxy: web address
        '''
        result = urlparse(args.proxy)
        assert result.scheme, "Proxy must be a web address"
        return args.proxy
    else:
        return False



def existing_training_model(training_input, options):
    '''
    Input data are used to prepopulate the model, based on prior runs
    :param datain Input data files are stored as models (if .model) or
    features (if .features) they are pickeled model and feature files
    '''
    verbose_print(options.get('verbose'), "Using existing model")
    trained_model = Placeholder()
    if training_input.endswith('.cluster'):
        with open(training_input, 'rb') as fid:
            cluster = pickle.load(fid)
            trained_model.cluster = cluster
            trained_model.model = cluster.model
    elif training_input.endswith('.model'):
        with open(training_input, 'rb') as fid:
            model = pickle.load(fid)
            trained_model = model
        if options.get('cluster'):
            cluster = cl.Clustering(model.training_data.compound,
                                    seed=options.get('random_seed'))
            cluster.cluster_training(model)
            trained_model.cluster = cluster
    elif training_input.endswith('.features'):
        with open(training_input, 'rb') as fid:
            train = pickle.load(fid)
            train = bt.Process(train, seed=options.get('random_seed'))
            model = tm.Train(train)
            model.train_model()
    if options.get('cross_validate'):
        verbose_print(options.get('verbose'), "Running cross-validational analysis")
        cv.Analysis(model, options.get('random_seed'))
    return trained_model



def add_pubchem_features(compounds, options, training=False):
    '''This function loads the pubchem features and downloads the data'''
    predictors = False
    if compounds.predictors:
        predictors = compounds.predictors
    if training:
        weights = compounds.weights
    else:
        weights = False
    collected_data = pcp.Collect(compounds.compounds, fingerprint=options.get('fingerprint'),
                                 xml=options.get('experimental'), sdf=options.get('chemofeatures'),
                                 proxy=options.get('proxy'), user=options.get('user'), chunks=options.get('chunks'), 
                                 try_count=options.get('try_count'), verbose=options.get('verbose'), predictors=predictors,
                                 weights=weights, smiles=options.get('smiles'))
    return collected_data



def collect_distance_matrix(collected_data):
    '''If clustering the data, retain the original training_data, without
    removing the redundant features'''
    original_data = deepcopy(collected_data)
    original_data = fp.Update(original_data, remove_static=False)
    original_data.update()
    fingerprint_vector = list()
    key_list = list()
    for key, value in dictitems(original_data.compound):
        fingerprint_vector.append(value['binhash'])
        key_list.append(key)
    distance = ds.Distance(fingerprint_vector, key_list).distance    
    return distance



def extract_features(collected_data, options, testing=False, test_input_directory='', remove_static=False):
    '''This function collects and extracts features from the raw user and
    PubChem data'''
    if options.get('fingerprint') is True and not test_input_directory:
        '''This part of the code converts the CACTVS fingerprints into features
        it also removes the fully redundant features, which for many chemical
        properties may be a large portion'''
        collected_data = fp.Update(collected_data, remove_static=remove_static,
                                   verbose=options.get('verbose'))
        collected_data.update()
    if options.get('user') is True and not test_input_directory:
        '''This part of the code extracts the user defined features'''
        collected_data = usr.Update(collected_data,
                                    remove_static=remove_static,
                                    verbose=options.get('verbose'))
        collected_data.update()
    if options.get('experimental') is True and not test_input_directory:
        '''This part of the code extracts PubChem experimental and computed
        features from the PubChem xml files'''
        collected_data = exp.Update(collected_data,
                                    remove_static=remove_static,
                                    verbose=options.get('verbose'))
        collected_data.update()    
    if options.get('chemofeatures') is True:
        '''This code runs PaDEL-Descriptor and extracts relevant 2-D
        and 3-D chemical descriptors'''
        collected_data = cf.Update(collected_data, testing=testing,
                                   remove_static=remove_static,
                                   verbose=options.get('verbose'))
        collected_data.update(padel=True, private=test_input_directory)
    return collected_data



def report_model_validation(model, options):
    verbose_print(options.get('verbose'), "Running cross-validational analysis")
    analysis = cv.Analysis(model, options.get('random_seed'))
    analysis.cross_validate()
    if options.get('plot'):
        analysis.plot_random_ROC()
    analysis.feature_importances()



def train_model(training_input, predictor, options):
    '''
    The program is essentially run in one of two mutually exclusive modes
    (training or test)
    :param train if True, being parsing and training model file
    '''    
    verbose_print(options.get('verbose'), "Training model")
    if training_input.endswith('.model') or training_input.endswith('.cluster') or training_input.endswith('.features'):
        trained_model = existing_training_model(training_input, options)
    else:
        trained_model = Placeholder()
        distance = False
        training_data = False
        verbose_print(options.get('verbose'), "Reading training set")
        if (options.get('distance') is True) or (options.get('cluster') is True) or (options.get('impute') is True):
            '''These functions all require a distance matrix, which is best
            collected using the fingerprint data'''
            options['fingerprint'] = True
        training = rt.Read(training_input, predictor, weights=options.get('weight'), user=options.get('user'), id_name='PubChem')
        '''This block of code generally works on feature collection and
        parsing, including the removal of fully redundant features. The
        difference between remove_static=True and False is whether or not
        to get rid of fully redundant features. Since the distance matrix
        is the same, regardless, it is run using original data'''
        verbose_print(options.get('verbose'), "Collecting features")
        training_data = add_pubchem_features(training, options, training=True)
        if (options.get('cluster') is True) or (options.get('distance') is True) or (options.get('impute') is True):
            verbose_print(options.get('verbose'), "Creating distance matrix")
            '''Collect distance matrix using the original dataset'''
            distance = collect_distance_matrix(training_data)
        '''Extract features from the user and PubChem data'''
        verbose_print(options.get('verbose'), "Extracting features")
        training_data = extract_features(training_data, options, remove_static=False)
        '''Discretize the y-values for the the classification process.
        If no split value is provided then the default for the program
        is to break the value at the median
        '''
        if training_data.compound:
            ids = [id for id, compound in dictitems(OrderedDict(sorted(training_data.compound.items(), key=lambda t: t[0])))]
            train = bt.Process(training_data, split_value=options.get('split_value'),
                               verbose=options.get('verbose'))
            if options.get('impute') is True:
                train.impute_values(distance=distance,
                                    verbose=options.get('verbose'))
            if options.get('error_correct') is True:
                train.check_errors(distance=distance,
                                  verbose=options.get('verbose'))
            if options.get('txt') is True:
                columns = "\t".join(train.feature_names)
                id_array = np.asarray(ids)
                np.savetxt('training_data_metacyc.txt', train.train, 
                           header=columns, delimiter='\t', fmt='%1.4f')
                np.savetxt('id_data_metacyc.txt', id_array, header='ID', fmt="%s")
            if options.get('selection') is True:
                train.feature_selection(verbose=options.get('verbose'),
                                        seed=options.get('random'))
            '''If dataout parameter is set, it prints to pickle a file
            containing the features that were extracted. In later runs
            this can be specified as the data input using the datain
            parameter
            '''
            if options.get('model_name'):
                features_file = 'pre-built_models/'+options.get('model_name') + ".features"
                with open(features_file, 'wb') as fid:
                    pickle.dump(train, fid)
            '''This is where the model is actually trained in the tm module'''
            trained_model = tm.Train(train)
            trained_model.train_model()

            '''If dataout parameter is set, it prints to pickle a file
            containing the RF model. In later runs this can be specified
            as the data input using the datain parameter
            '''
            if options.get('model_name'):
                model_file = 'pre-built_models/'+options.get('model_name') + ".model"
                with open(model_file, 'wb') as fid:
                    pickle.dump(trained_model, fid)
            if options.get('cross_validate'):
                report_model_validation(trained_model, options)
            if options.get('cluster'):
                cluster = cl.Clustering(training_data.compound, seed=options.get('random'))
                cluster.cluster_training(trained_model)
                trained_model.cluster = cluster
                if options.get('model_name'):
                    cluster_file = './pre-built_models/'+options.get('model_name') + ".cluster"
                    with open(cluster_file, 'wb') as fid:
                        pickle.dump(cluster, fid)
    
    return trained_model



def test_model(trained_model, test_input_file, test_input_directory, predictor, options):
    if trained_model is False:
        assert options.get('model_name'), "If the model hasn't been trained, input data\
                             must be specified"
        training_input = './pre-built_models/'+options.get('model_name') + ".model"
        trained_model = existing_training_model(training_input, options)

    testing = False
    test = False
    if test_input_file:
        testing = rtest.Read(test_input_file, user=options.get('user'), id_name='PubChem')
    # else:
    #     testing = Placeholder()
        
        testing.predictors = False
        if (options.get('distance') is True) or (options.get('cluster') is True) or (options.get('impute') is True):
            '''These functions all require a distance matrix, which is best collected
            using the fingerprint data'''
            options['fingerprint'] = True  
            
        verbose_print(options.get('verbose'), "Adding PubChem Features")
        testing_data = add_pubchem_features(testing, options, training=False)
                   
        if (options.get('distance') is True) or (options.get('cluster') is True) or (options.get('impute') is True):
            distance = collect_distance_matrix(testing_data)

        verbose_print(options.get('verbose'), "Extracting features")        
        testing_data = extract_features(testing_data, options, testing=True, remove_static=False)

        test = btest.Process(testing_data, trained_model.features)
        if (options.get('distance') is True) or (options.get('cluster') is True) or (options.get('impute') is True):
            test.impute_values(distance=distance, verbose=options.get('verbose'))

    ''' Incorporate local testing .sdf files '''
    if test_input_directory:
        local_testing_data = pcp.Collect(local=test_input_directory)
        local_testing_data = extract_features(local_testing_data, options, test_input_directory=test_input_directory, remove_static=False)
        local_test = btest.Process(local_testing_data, trained_model.features)#, feature_names=trained_model.feature_names)
        if test_input_file:
            test.compounds += local_test.compounds
            test.test = np.append(test.test,local_test.test, axis=0)
            test.rows += local_test.rows
            test.test_names += local_test.test_names
        else:
            test = local_test

        test.impute_values(simple=True)

    if options.get('txt'):
        columns = "\t".join(test.features)
        id_array = np.asarray(test.test_names)
        np.savetxt('testing_data_metacyc.txt', test.test, 
                   header=columns, delimiter='\t', fmt='%1.4f')
        np.savetxt('id_test_data_metacyc.txt', id_array, header='ID', fmt="%s")



    if options.get('cluster') and hasattr(trained_model, 'cluster'):
        verbose_print(options.get('verbose'),
                      "Rejecting based on clustering of training")
        cluster = trained_model.cluster
        testmatrix = []
        testcompounds = []
        for i in range(len(test.test[:, 0])):
            testmatrix.append(test.test[i, :])
            testcompounds.append(test.test_names[i])
        testmatrix = np.array(testmatrix)
        predicted = cluster.cluster_testing(testmatrix)
        for pred in predicted:
            print(test.test_names[pred])
    prediction = trained_model.clf.predict_proba(test.test)
    prediction_dict = {}
    for i, name in enumerate(test.test_names):
        print(name, prediction[i, ])
        prediction_dict[name] = prediction[i][1]

    return [testing, test, prediction_dict]



def clean_training_testing(training=False, testing=False):
    files = []
    if training:
        files += glob.glob('bcml/Chemoinformatics/db/training/*')
    if testing:
        files += glob.glob('bcml/Chemoinformatics/db/testing/*')
    for f in files:
        os.remove(f)

def check_results_folder(PATH, currentDT, predictor):
    newpath = PATH+'/results_{}_{}/'.format(predictor, currentDT)
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return (newpath)

