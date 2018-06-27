import os 
import argparse
import re 
import numpy as np 
import pandas as pd 
import collections

def process_arg(arg):
    args = arg.split(",")
    try:
        return map(eval,args)
    except:
        # keep as str 
        return args 
    
def find_result(data, metric, sample_size, sample_type, fold, dim, lambd):
    # print sample_size,sample_type,fold,dim,lambd
    resultx = collections.defaultdict(dict)
    for i in data.index:
        if data.at[i,"sample_type"] in [sample_type,""] and \
            data.at[i,"sample_size"] in [sample_size,""] and \
            data.at[i,"folds"] in [fold,""] and \
            data.at[i,"dim"] in [dim,""] and \
            data.at[i,"lambd"] in [lambd,""]:
            predictor = data.at[i,"predictor"]
            design = data.at[i,"design"]
            resultx[design][predictor] = data.at[i,metric]
    return resultx
            
            
def print_result(resultx, sample_size, sample_type, fold, dim, lambd):
    print "Sample size:     ",sample_size 
    print "Sample type:     ",sample_type 
    print "Folds:           ",fold 
    print "Dimension:       ",dim 
    print "Regularization:  ",lambd 
    print "design            DATE       suspect2vec"
    datez = []
    suspect2vecz = []
    for design in sorted(resultx):
        print "%s%.3f      %.3f" %(design.ljust(18),resultx[design]["DATE"], resultx[design]["suspect2vec"])
        datez.append(resultx[design]["DATE"])
        suspect2vecz.append(resultx[design]["suspect2vec"])
    print "Mean              %.3f      %.3f" %(np.mean(datez),np.mean(suspect2vecz))
    print ""

def main(args):   
    sample_sizes = process_arg(args.sample_size)
    sample_types = process_arg(args.sample_type)
    folds = process_arg(args.folds)
    dims = process_arg(args.dim)
    lambds = process_arg(args.lambd)
    
    data = pd.read_csv("raw_results.csv",index_col=0).replace(np.nan,"")        
    
    for sample_size in sample_sizes:
        for sample_type in sample_types:
            for fold in folds:
                for dim in dims:
                    for lambd in lambds:
                        resultx = find_result(data,args.metric,sample_size,sample_type,fold,dim,lambd)
                        if len(resultx) > 0:
                            print_result(resultx,sample_size,sample_type,fold,dim,lambd)
       
    
def init(parser):
    parser.add_argument("--sample_size",default="0.5",help="Number of suspects in initial subset (sample) of suspect set that" \
                        " is to be ranking.")
    parser.add_argument("--sample_type",default="\"solver\"",help="Method to choose observed suspect set. 'random' for random or "
                        "'solver' for order in which the solver finds them.")
    parser.add_argument("--folds",default="1e12")
    parser.add_argument("--dim",default="20",help="Embedding dimension")
    parser.add_argument("--lambd",default="0",help="Regularization factor")
    parser.add_argument("--metric",default="mean_fscore")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)
    