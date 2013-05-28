#!/usr/bin/env python

import argparse
import csv

import faithful.bayesfmm as fmm
from faithful import ida, clustering,em
from stats import diagnostics, utils

def main(filename,iterations,saveDiagnostics,outputDir,burnin):
    data = []
    with open(filename,'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = csvreader.next()
        for line in csvreader:
            eruption_time = float(line[0])
            waiting_time = float(line[1])
            data.append([eruption_time,waiting_time])

    #generate ida images
    ida.scatter_plot(data,'%s/faithful_ida_scatter.png' % outputDir)
    ida.histogram(data,'%s/faithful_ida_hist.png' % outputDir)
    ida.linear_regression(data, '%s/faithful_ida_regression.png' % outputDir)
    
    #clustering
    [km2,] = clustering.plot_kmeans(data,filename='%s/faithful_kmeans_k2.png' % outputDir)
    clustering.plot_kmeans(data,ks=(6,),filename='%s/faithful_kmeans_k6.png' % outputDir)
    # train and test data
    data_train,data_test = utils.split_data(data,train_split=0.8)
    [kmeans_model_2,kmeans_model_6] = clustering.plot_kmeans(data_train,ks=(2,6),suppress_output=True)
    clustering.kmeans_predict(data_test,kmeans_model_2,filename='%s/faithful_kmeans_k2_predict.png' % outputDir)
    clustering.kmeans_predict(data_test,kmeans_model_6,filename='%s/faithful_kmeans_k6_predict.png' % outputDir)
    
    #expectation-maximization
    theta,sigma,pi = em.gaussian_em_2(data,reps=25)
    em.plot_against_kmeans(data,theta,km2,filename='%s/faithful_em_kmeans.png' % outputDir)
    
    #build fmm model
    gaussian_fmm = fmm.GaussianFiniteMixtureModel()
    pi,theta,sigma = gaussian_fmm.run(data,k=2,iterations=iterations)
    
    outfilename = None
    if saveDiagnostics:
        outfilename = '%s/diagnostics_%s_%s.png' % (outputDir,'%s','%s')
        
    #generate diagnostics
    diagnostics.stationarity_plot(pi,ylab=r'\pi',filename=(outfilename % ('stationarity','pi') if outfilename is not None else None))
    diagnostics.stationarity_plot(theta,ylab=r'\theta',filename=(outfilename % ('stationarity','theta') if outfilename is not None else None))
    diagnostics.stationarity_plot(sigma,ylab=r'\Sigma',filename=(outfilename % ('stationarity','sigma') if outfilename is not None else None))
    
    #determine model parameters
    

if __name__=='__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='datafile')
    cmdline_parser.add_argument('--iterations', action='store', dest='iterations', type=int, default=500)
    cmdline_parser.add_argument('--saveDiagnostics', action='store_true', default=False)
    cmdline_parser.add_argument('--outputDir', action='store',default='.')
    cmdline_parser.add_argument('--burnin', action='store', type=int, default=0)
    
    parsed_args = cmdline_parser.parse_args()
    
    main(parsed_args.filename, parsed_args.iterations, parsed_args.saveDiagnostics, parsed_args.outputDir, parsed_args.burnin)
    