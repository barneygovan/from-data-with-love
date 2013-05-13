#!/usr/bin/env python

import argparse
import csv

import faithful.bayesfmm as fmm
from stats import diagnostics

def main(filename,iterations,saveDiagnostics,outputDir,burnin):
    data = []
    with open(filename,'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = csvreader.next()
        for line in csvreader:
            eruption_time = float(line[0])
            waiting_time = float(line[1])
            data.append([eruption_time,waiting_time])
    
    #build model
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
    