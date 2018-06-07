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
    

def blocking_analysis(base_failure, new_failure, verbose=False, min_runtime=0):
    if verbose:
        print "Analyzing",new_failure
        
    base_suspectz = utils.parse_suspects(base_failure)
    new_suspectz = utils.parse_suspects(new_failure)
    recall = len(new_suspectz)/float(len(base_suspectz))
    
    # analyze runtime speedup
    base_runtime = utils.parse_runtime(base_failure)
    if base_runtime < min_runtime:
        return None,None,None,None,None 
    new_runtime = utils.parse_runtime(new_failure)
    speedup = new_runtime / base_runtime 
            
    # Parse logs to find which suspects were blocked. 
    log_file = os.path.join(new_failure+".vennsawork","logs","vdb","vdb.log")
    log = open(log_file).readlines()
    blocked = []
    total_suspects = 0
    suspect_union = set([])
    for line in log:
        m = re.search(r"Predicting not suspect (\S+)\s", line)
        if m:
            blocked.append(m.group(1))
        m = re.search(r"## suspect: (\S+), output\(s\): \d+, literal: \d+", line)
        if m:
            suspect_union.add(m.group(1))
    
    # analyze peak memory usage
    base_mem = parse_peak_memory(base_failure)
    new_mem = parse_peak_memory(new_failure)
    mem_reduce = new_mem / float(base_mem)
    
    # Sanity checking
    # Make sure that adding constraints didn't somehow result in new suspects. 
    # This condition means 100% prediction precision.
    # for s in new_suspectz:
        # assert s in base_suspectz, "new suspect %s found with blocking" %(s)
        
    # Make sure that all blocked suspects were missed
    blocked_true = 0
    for s in blocked:
        if s in base_suspectz:
            blocked_true += 1
        #assert s not in new_suspectz, "solver found suspect %s which was supposed to be blocked" %(s)
    block_recall = (len(base_suspectz)-blocked_true)/float(len(base_suspectz))
    
    if len(blocked) == 0:
        # assert len(base_suspectz) < 20
        block_acc = None 
    else:
        block_acc = 1 - blocked_true/float(len(blocked))
    
    # Make sure that all missing suspects were in fact blocked - no other suspects were missed 
    # for s in base_suspectz:
        # if s not in new_suspectz:
            # assert s in blocked, "solver did not find suspect %s which was not supposed to be blocked" %(s)
    
    if verbose:
        print "Number of true suspects: %i" %(len(base_suspectz))
        print "Number of found suspects: %i (recall %.3f)" %(len(new_suspectz), recall)
        print "Number of blocked suspects: %i out of %i" %(len(blocked),len(suspect_union))
        print "Relative runtime: %.3f" %(speedup)
        print "Peak memory reduction: %.3f" %(mem_reduce)    
        print "Recall directly due to blocked suspects: %.3f" %(block_recall)
        print "Block prediction accuracy: %s" %(block_acc)
        print ""
            
    return recall, speedup, mem_reduce, block_recall, block_acc
    

def recall_vs_time_single(failure):
    log_file = os.path.join(failure+".vennsawork","logs","vdb","vdb.log")
    start = 0
    with open(log_file) as f:
        for line in f:
            m = re.search(r".*\d+-\w+-\d+ \d+:\d+:\d+ \((\d+):(\d+):(\d+)\.(\d+)\) ## Oracle::ask()", line)
            if m:
                start = 3600*int(m.group(1)) + 60*int(m.group(2)) + int(m.group(3)) + float("0."+m.group(4))
                break

    cnt = 0
    points = []
    with open(log_file) as f:
        for line in open(log_file):
            m = re.search(r".*\d+-\w+-\d+ \d+:\d+:\d+ \((\d+):(\d+):(\d+)\.(\d+)\) ##  ==> solver solution:", line)
            if m:
                t = 3600*int(m.group(1)) + 60*int(m.group(2)) + int(m.group(3)) + float("0."+m.group(4)) - start
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
    
    
def auc_recall_time(points):
    '''
    Compute area under recall vs time curve from t = 0 to 1. 
    '''
    auc = 0
    #print points
    for i in range(1,len(points)):
        if points[i][0] > 1:
            break
        recall = points[i-1][1]
        dt = points[i][0] - points[i-1][0]
        assert dt >= 0
        auc += recall*dt   

    if points[i-1][0] < 1:
        dt = 1-points[i-1][0]
        recall = points[i-1][1]
        auc += recall*dt 
    return auc 
    

def plot_recall_vs_time(points, outfile, color='r', label=None):
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
                y.append(np.mean(cur_bin))
            cur_bin = []
            i += 1
        cur_bin.append(p[1])
        
    plt.scatter(x, y, color=color, label=label)
    plt.xlabel("Relative runtime")
    plt.ylabel("Mean recall")
    plt.xlim((-0.2,1.2))
    plt.ylim((0,1))
    plt.legend(loc="upper left")
    plt.savefig(outfile)
    
    
def assumption_analysis(base_failure, new_failure, verbose=False, min_runtime=0):
    if verbose:
        print "Analyzing",new_failure
        
    base_suspectz = utils.parse_suspects(base_failure)
    new_suspectz = utils.parse_suspects(new_failure)
    recall = len(new_suspectz)/float(len(base_suspectz))
    # assert len(base_suspectz) <= len(new_suspectz), \
        # "Base failure found %i suspects while new found %i" %(len(base_suspectz), len(new_suspectz))        
        
    base_runtime = utils.parse_runtime(base_failure)
    if base_runtime < min_runtime:
        return None         
    
    # analyze peak memory usage
    base_mem = parse_peak_memory(base_failure)
    new_mem = parse_peak_memory(new_failure)
    mem_reduce = new_mem / float(base_mem)
        
    # parse predictions, solutions, and compute accuracy
    log_file = new_failure+".vennsawork/logs/vdb/vdb.log"
    predictions = []
    found_suspects = set([])
    
    for line in open(log_file):
        m = re.search(r"Predicting next suspect (\S+)", line)
        if m:
            s = m.group(1)
            if s != 0:
                predictions.append(s)
            
        # m = re.search(r"Found suspect (\d+)", line)
        m = re.search(r"==> solver solution:.*:(\S+)\s+", line)
        if m:
            s = m.group(1)
            found_suspects.add(s)
            
    auc_acc = 0
    correct = 0
    total = 0
    for s in predictions:
        total += 1 
        if s in found_suspects:
            correct += 1
        auc_acc += float(correct)/total 
    if total > 0:
        auc_acc /= total
    
    base_points, new_points = recall_vs_time(base_failure, new_failure)
    base_recall_auc = auc_recall_time(base_points)
    new_recall_auc = auc_recall_time(new_points)
    recall_auc_improvement = new_recall_auc / base_recall_auc
    
    if verbose:
        print "Number of true suspects: %i" %(len(base_suspectz))
        print "Number of found suspects: %i (recall %.3f)" %(len(new_suspectz), recall)
        print "Prediction accuracy auc: %.3f" %(auc_acc)
        print "Recall auc improvement: %.3f" %(recall_auc_improvement)
        print "Peak memory reduction: %.3f" %(mem_reduce)
        print ""
        
    return recall_auc_improvement
  
  
def main(args):
    if args.design:
        all_failurez = utils.find_all_failures(args.design)
    else:
        assert args.failure 
        all_failurez = [args.failure]
    
    # infer guidance method 
    f = all_failurez[0]
    log_file = f + args.new_suffix + ".vennsawork/logs/vdb/vdb.log"
    log = open(log_file).read()
    m = re.search(r"Guidance method = (\d+)", log)
    method = int(m.group(1)) if m else 0 
    
    if method in [2,3,4]:
        base_points = []
        new_points = []
        recall_auc_improvementz = []
        
        for failure in all_failurez: 
            recall_auc_improvement = assumption_analysis(failure+args.base_suffix, failure+args.new_suffix, 
                verbose=args.verbose, min_runtime=args.min_runtime)
            
            if recall_auc_improvement is not None:
                recall_auc_improvementz.append(recall_auc_improvement)
                
            # base,new = recall_vs_time(failure,failure+args.new_suffix)
            # base_points.extend(base)
            # new_points.extend(new)
        # plot_recall_vs_time(base_points, "plots/recall_vs_time.png", 'r')
        # plot_recall_vs_time(new_points, "plots/recall_vs_time.png", 'b')  

        print "Arithmetic mean recall auc improvement: %.3f" %(np.mean(recall_auc_improvementz))
        print "Median recall auc improvement: %.3f" %(np.median(recall_auc_improvementz))
        print "Geometric mean recall auc improvement: %.3f" %(gmean(recall_auc_improvementz))
        
    elif method == 1:
        recalls = []
        speedups = []
        mem_reductions = []
        block_recalls = []
        block_accs = []
        
        for failure in all_failurez: 
            recall, speedup, mem_reduce, block_recall, block_acc = blocking_analysis(failure+args.base_suffix, failure+args.new_suffix, 
                verbose=args.verbose, min_runtime = args.min_runtime)
                
            if recall is not None:
                recalls.append(recall)
                speedups.append(speedup)
                mem_reductions.append(mem_reduce)
                block_recalls.append(block_recall)
                if block_acc is not None:
                    block_accs.append(block_acc)    
                
        print ""
        print "Mean recall: %.3f" %(np.mean(recalls))
        print "Mean speedup: %.3f" %(gmean(speedups))
        print "Mean memory reduction: %.3f" %(gmean(mem_reductions))
        print "Recall directly due to blocked suspects: %.3f" %(np.mean(block_recalls))
        print "Block prediction accuracy: %.3f" %(np.mean(block_accs))

    else:
        raise ValueError("Can't analyze debugging; no guidance method used.")
    
    
def init(parser):
    parser.add_argument("new_suffix", default="", help="Suffix of failure names to compare against the baseline")
    parser.add_argument("base_suffix", nargs='?', default="", help="[optional] Suffix of failure names to compare against the baseline")
    parser.add_argument("--design", help="Design to analyze.")
    parser.add_argument("--failure", help="Analyze a single failure (base name).")    
    parser.add_argument("--min_runtime", type=int, default=0, help="Exclude designs with runtime less than this.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)
