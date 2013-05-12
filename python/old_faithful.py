#!/usr/bin/env python

import argparse
import csv

import faithful.bayesfmm as fmm

def main(filename,iterations):
    data = []
    with open(filename,'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = csvreader.next()
        for line in csvreader:
            eruption_time = float(line[0])
            waiting_time = float(line[1])
            data.append([eruption_time,waiting_time])
    
    gaussian_fmm = fmm.GaussianFiniteMixtureModel()
    pi,theta,sigma = gaussian_fmm.run(data,k=2,iterations=100)
    
    #print pi[-1]
    #print theta[-1]
    #print sigma[-1]

if __name__=='__main__':
    cmdline_parser = argparse.ArgumentParser()
    cmdline_parser.add_argument('filename', metavar='datafile')
    cmdline_parser.add_argument('--iterations', action='store', dest='iterations', type=int, default=500)
    
    parsed_args = cmdline_parser.parse_args()
    
    main(parsed_args.filename, parsed_args.iterations)