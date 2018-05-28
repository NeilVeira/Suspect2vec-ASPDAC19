import os 
import argparse 
import re
import numpy as np
from scipy.stats.mstats import gmean 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab
from matplotlib import gridspec

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
    

def basic_analysis(base_failure, new_failure, verbose=False, min_runtime=0):
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
    if base_runtime < min_runtime:
        return None,None,None,None,None 
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
    # for s in new_suspectz:
        # assert s in old_suspectz, "new suspect %s found with blocking" %(s)
        
    # Make sure that all blocked suspects were missed
    blocked_true = 0
    for s in blocked:
        if s in old_suspectz:
            blocked_true += 1
        #assert s not in new_suspectz, "solver found suspect %s which was supposed to be blocked" %(s)
    block_recall = (len(old_suspectz)-blocked_true)/float(len(old_suspectz))
    if verbose:
        print "Recall directly due to blocked suspects: %.3f" %(block_recall)
    
    if len(blocked) == 0:
        # assert len(old_suspectz) < 20
        block_acc = None 
    else:
        block_acc = 1 - blocked_true/float(len(blocked))
        if verbose:
            print "Block prediction accuracy: %.3f" %(block_acc)
    
    # Make sure that all missing suspects were in fact blocked - no other suspects were missed 
    # for s in old_suspectz:
        # if s not in new_suspectz:
            # assert s in blocked, "solver did not find suspect %s which was not supposed to be blocked" %(s)
            
    return recall, speedup, mem_reduce, block_recall, block_acc
    
    
def recall_vs_time_single(failure):
    log_file = os.path.join(failure+".vennsawork","logs","vdb","vdb.log")
    cnt = 0
    points = []
    for line in open(log_file):
        m = re.search(r".*\d+-\w+-\d+ \d+:\d+:\d+ \((\d+):(\d+):(\d+)\.(\d+)\) ##  ==> solver solution:", line)
        if m:
            t = 3600*int(m.group(1)) + 60*int(m.group(2)) + int(m.group(3)) + float("0."+m.group(4))
            cnt += 1 
            points.append([t,cnt])
    return points
            
def recall_vs_time(base_failure, new_failure):
    base_points = recall_vs_time_single(base_failure)
    new_points = recall_vs_time_single(new_failure)
    
    #normalize against base failure
    max_t = float(base_points[-1][0])
    max_n = float(base_points[-1][1])
    for i in range(len(base_points)):
        base_points[i][0] /= max_t
        base_points[i][1] /= max_n 
    for i in range(len(new_points)):
        new_points[i][0] /= max_t 
        new_points[i][1] /= max_n 
    return base_points, new_points
    

def plot_recall_vs_time(points, outfile, color='r'):
    x = []
    y = []
    points.sort()
    i = 0
    cur_bin = []
    binx = np.concatenate([np.linspace(0,1,21), np.linspace(1,points[-1][0],5)])
    
    for p in points:
        if p[0] > binx[i]:
            if len(cur_bin) > 0:
                x.append(binx[i])
                y.append(gmean(cur_bin))
            cur_bin = []
            i += 1
        cur_bin.append(p[1])
        
    plt.scatter(x,y, color=color)
    plt.xlabel("Relative time")
    plt.ylabel("Mean recall")
    plt.xlim((-0.2,1.2))
    plt.ylim((0,1))
    plt.savefig(outfile)
  
  
def main(args):
    if args.design:
        design_list = [args.design]
    else:
        design_list = ["aemb"]
        
    recalls = []
    speedups = []
    mem_reductions = []
    block_recalls = []
    base_points = []
    new_points = []
    block_accs = []
    
    for design in design_list:
        print "Evaluating design", design
        for failure in utils.find_all_failures(design):
            print failure
            recall, speedup, mem_reduce, block_recall, block_acc = basic_analysis(failure, failure+args.new_suffix, verbose=True, 
                min_runtime = args.min_runtime)
            if recall is not None:
                recalls.append(recall)
                speedups.append(speedup)
                mem_reductions.append(mem_reduce)
                block_recalls.append(block_recall)
                if block_acc is not None:
                    block_accs.append(block_acc)
            
            # base,new = recall_vs_time(failure,failure+args.new_suffix)
            # base_points.extend(base)
            # new_points.extend(new)
            
    print ""
    print "Mean recall: %.3f" %(np.mean(recalls))
    print "Mean speedup: %.3f" %(gmean(speedups))
    print "Mean memory reduction: %.3f" %(gmean(mem_reductions))
    print "Recall directly due to blocked suspects: %.3f" %(np.mean(block_recalls))
    print "Block prediction accuracy: %.3f" %(np.mean(block_accs))
    
    # plot_recall_vs_time(base_points, "plots/recall_vs_time.png", 'r')
    # plot_recall_vs_time(new_points, "plots/recall_vs_time.png", 'b')
    
    
def init(parser):
    parser.add_argument("design", help="Design to analyze. If None does all designs")
    parser.add_argument("new_suffix", nargs='?', default="", help="Suffix of failure names to compare against the baseline")
    parser.add_argument("--min_runtime", type=int, default=0, help="Exclude designs with runtime less than this.")
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)