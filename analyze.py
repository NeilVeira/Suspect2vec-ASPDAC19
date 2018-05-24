import os 
import argparse 
import re
from scipy.stats.mstats import gmean 

import utils 

FLOAT_PATTERN = "([\d\.]+)"
INT_PATTERN = "(\d+)"

def parse_peak_memory(failure):
    mem_file = os.path.join(failure+".vennsawork","logs","vdb","vdb.plot")
    peak_mem = 0
    for line in open(mem_file):
        m = re.match(r"\s+{}\s+{}\s+{}\s+".format(FLOAT_PATTERN,INT_PATTERN,INT_PATTERN), line)
        if m:
            mem = int(m.group(3))
            peak_mem = max(peak_mem,mem)
    return peak_mem
    

def basic_analysis(base_failure, new_failure, verbose=False):
    if verbose:
        print "Analyzing",new_failure
    old_suspectz = utils.parse_suspects(base_failure)
    new_suspectz = utils.parse_suspects(new_failure)
            
    # Parse logs to find which suspects were blocked. 
    log_file = os.path.join(new_failure+".vennsawork","logs","vdb","vdb.log")
    log = open(log_file).readlines()
    blocked = []
    total_suspects = 0
    for line in log:
        m = re.search(r"Predicting not suspect (\S+)\s", line)
        if m:
            blocked.append(m.group(1))
        m = re.search(r"Running with (\d+) suspects:", line)
        if m:
            total_suspects += int(m.group(1))
            
    recall = len(new_suspectz)/float(len(old_suspectz))
    
    # analyze runtime speedup
    base_runtime = utils.parse_runtime(base_failure)
    new_runtime = utils.parse_runtime(new_failure)
    speedup = new_runtime / base_runtime 
    
    # analyze peak memory usage
    base_mem = parse_peak_memory(base_failure)
    new_mem = parse_peak_memory(new_failure)
    #print base_mem,new_mem
    mem_reduce = new_mem / float(base_mem)
    
    if verbose:
        print "Number of true suspects: %i" %(len(old_suspectz))
        print "Number of found suspects: %i (recall %.3f)" %(len(new_suspectz), recall)
        print "Number of blocked suspects: %i out of %i" %(len(blocked),total_suspects)
        print "Relative runtime: %.3f" %(speedup)
        print "Peak memory reduction: %.3f" %(mem_reduce)
    
    # Sanity checking
    # Make sure that adding constraints didn't somehow result in new suspects. 
    # This condition means 100% prediction precision.
    for s in new_suspectz:
        assert s in old_suspectz, "new suspect %s found with blocking" %(s)
        
    # Make sure that all blocked suspects were missed 
    # for s in blocked:
        # assert s not in new_suspectz, "solver found suspect %s which was supposed to be blocked" %(s)
    
    # Make sure that all missing suspects were in fact blocked - no other suspects were missed 
    # for s in old_suspectz:
        # if s not in new_suspectz:
            # assert s in blocked, "solver did not find suspect %s which was not supposed to be blocked" %(s)
            
    return recall, speedup, mem_reduce

  
def main(args):
    if args.design:
        design_list = [args.design]
    else:
        design_list = ["aemb"]
        
    recalls = []
    speedups = []
    mem_reductions = []
    for design in design_list:
        print "Evaluating design", design
        for failure in utils.find_all_failures(design):
            recall, speedup, mem_reduce = basic_analysis(failure, failure+args.new_suffix)
            recalls.append(recall)
            speedups.append(speedup)
            mem_reductions.append(mem_reduce)
            
    print "Mean recall: %.3f" %(gmean(recalls))
    print "Mean speedup: %.3f" %(gmean(speedups))
    print "Mean memory reduction: %.3f" %(gmean(mem_reductions))
    
    
def init(parser):
    parser.add_argument("new_suffix", help="Suffix of failure names to compare against the baseline")
    parser.add_argument("--design", help="Design to analyze. If None does all designs")
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)