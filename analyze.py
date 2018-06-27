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
from matplotlib.font_manager import FontProperties

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
    if not os.path.exists(new_failure+".vennsawork/logs/vdb/vdb.log"):
        print "Skipping failure %s as it appears to have failed or not been run." %(new_failure)
        return None,None,None,None,None 
    if verbose:
        print "Analyzing",new_failure
        
    base_suspectz = utils.parse_suspects(base_failure)
    new_suspectz = utils.parse_suspects(new_failure)
    recall = len(new_suspectz)/float(len(base_suspectz))
    
    # analyze runtime speedup
    base_runtime = utils.parse_runtime(base_failure)
    if base_runtime < min_runtime:
        print "Skipping failure %s due to short runtime" %(new_failure)
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
    start = utils.find_time_of(failure, "Oracle::ask\(\)", default=0)            
    end_time = utils.parse_runtime(failure)

    cnt = 0
    points = []
    for line in open(log_file):
        t = utils.parse_time_of(line, "==> solver solution:")
        if t:
            cnt += 1 
            points.append([t-start,cnt])
    points.append([end_time,cnt])
    return points
    
  
def recall_vs_time(base_failure, new_failure):
    base_points = recall_vs_time_single(base_failure)
    new_points = recall_vs_time_single(new_failure)
    
    #normalize against base failure
    # end_time = max(base_points[-1][0], new_points[-1][0]) 
    end_time = base_points[-1][0]
    base_points.append([end_time,base_points[-1][1]])
    new_points.append([end_time,new_points[-1][1]])    
    max_n = float(base_points[-1][1])
    # print base_points 
    # print new_points
    # for i in range(len(base_points)):
        # print i,base_points[i][0],new_points[i][0]
    
    for i in range(len(base_points)):
        base_points[i][0] /= end_time
        base_points[i][1] /= max_n 
    for i in range(len(new_points)):
        new_points[i][0] /= end_time 
        new_points[i][1] /= max_n 
    return base_points, new_points
    
    
def auc_recall_time(points):
    '''
    Compute area under recall vs time curve from t = 0 to 1. 
    '''
    auc = 0
    dt_tot = 0
    points = [[0,0]] + list(points)
    i = 1
    while i < len(points):
        if points[i][0] > 1:
            break
        recall = points[i-1][1]
        dt = points[i][0] - points[i-1][0]
        assert dt >= 0
        auc += recall*dt  
        dt_tot += dt  
        i += 1

    if points[i-1][0] < 1:
        dt = 1-points[i-1][0]
        dt_tot += dt 
        recall = points[i-1][1]
        auc += recall*dt 
    assert abs(1-dt_tot) < 1e-6 
    return auc 
    
        
def interpolate_value(points, x):
    if x < points[0][0]:
        return 0 
    for i in range(len(points)-1):
        if points[i+1][0] > x:
            return points[i][1]
    return points[-1][1]

def plot_recall_vs_time(base_points, new_points, outfile=None):
    n_points = 101
    x = np.linspace(0,1,n_points)
    y_base = []
    y_new = []
    for i in range(len(x)):
        ys = []
        for points in base_points:
            ys.append(interpolate_value(points,x[i]))
        y_base.append(np.mean(ys))
        ys = []
        for points in new_points:
            ys.append(interpolate_value(points,x[i]))
        y_new.append(np.mean(ys))
    
    plt.clf()
    plt.plot(x, y_base, color='r', label="baseline debug", linestyle="--", linewidth=3)
    plt.plot(x, y_new, color='b', label="directed debug", linewidth=2)
    plt.fill_between(x, np.zeros(len(x)), y_base, color="r", alpha=0.5)
    plt.fill_between(x, y_base, y_new, color="b", alpha=0.25)

    if len(base_points) == 1:
        R_base = auc_recall_time(base_points[0])
        R_new = auc_recall_time(new_points[0])
        font = FontProperties()
        font.set_weight("heavy")
        plt.text(0.7, 0.3, "$R_{base}=%.3f$" %(R_base), fontsize=18, weight="heavy")
        plt.text(0.56, 0.6, "$R'=%.3f$" %(R_new), fontsize=18, weight="heavy")

    plt.xlabel("Relative runtime", fontsize=14)
    plt.ylabel("Suspect recall", fontsize=14)
    plt.xlim((0,1))
    plt.ylim((-0.05,1.05))
    plt.legend(loc="upper left")
    if outfile:
        plt.savefig(outfile)

        
def plot_improvements(outfile, recall_auc_improvementz):    
    plt.clf()
    x = recall_auc_improvementz
    x.sort()
    y = np.array(range(1,len(x)+1)) / float(len(x)) * 100 
    plt.xlabel("Improvement in area under recall-time curve")
    plt.ylabel("Percentage of failures")
    plt.plot(x,y)
    plt.savefig(outfile)

     
def assumption_analysis(base_failure, new_failure, verbose=False, min_runtime=0):
    if not os.path.exists(new_failure+".vennsawork/logs/vdb/vdb.log"):
        print "Skipping failure %s as it appears to have failed or not been run." %(new_failure)
        return None,None,None,None
    elif verbose:
        print "Analyzing",new_failure
        
    base_suspectz = utils.parse_suspects(base_failure)
    new_suspectz = utils.parse_suspects(new_failure)
    if len(base_suspectz) > 0:
        recall = len(new_suspectz)/float(len(base_suspectz))
    else:
        recall = np.nan
    # assert len(base_suspectz) <= len(new_suspectz), \
        # "Base failure found %i suspects while new found %i" %(len(base_suspectz), len(new_suspectz))        
        
    base_runtime = utils.parse_runtime(base_failure)
    if base_runtime < min_runtime:
        print "Skipping failure %s due to short runtime" %(new_failure)
        return None,None,None,None
    new_runtime = utils.parse_runtime(new_failure)
    speedup = new_runtime / base_runtime 
    
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
        print "Relative runtime: %.3f" %(speedup)
        print "Prediction accuracy auc: %.3f" %(auc_acc)
        print "Recall auc improvement: %.3f" %(recall_auc_improvement)
        print "Peak memory reduction: %.3f" %(mem_reduce)
        print ""
        
    return recall_auc_improvement, speedup, base_points, new_points
  
  
def main(args):
    if args.design:
        all_failurez = utils.find_all_failures(args.design)
    else:
        assert args.failure 
        all_failurez = [args.failure]
    
   
    analysis_method = args.method 
    if analysis_method == -1:        
        # infer best method 
        for f in all_failurez:
            log_file = f + args.new_suffix + ".vennsawork/logs/vdb/vdb.log"
            if os.path.exists(log_file):
                log = open(log_file).read()
                m = re.search(r"Guidance method = (\d+)", log)
                guidance_method = int(m.group(1)) if m else 0 
                if guidance_method in [2,3,4,5]:
                    analysis_method = 0 
                    break
                elif guidance_method in [1]:
                    analysis_method = 1  
                    break       
    
    if analysis_method == 0:
        all_base_points = []
        all_new_points = []
        recall_auc_improvementz = []
        speedupz = []
        
        for failure in all_failurez: 
            recall_auc_improvement, speedup, base_points, new_points = assumption_analysis(failure+args.base_suffix, failure+args.new_suffix, 
                verbose=args.verbose, min_runtime=args.min_runtime)
            
            if recall_auc_improvement is not None:
                recall_auc_improvementz.append(recall_auc_improvement)
                speedupz.append(speedup)
                all_base_points.append(base_points)
                all_new_points.append(new_points)

                if args.plot_individual:
                    outfile = "plots/%s_recall_vs_time.png" %(failure.replace("/","_"))
                    plot_recall_vs_time([base_points], [new_points], outfile=outfile)
                    plt.clf()
                
        if args.plot:   
            if args.design:
                design = os.path.basename(args.design)
            else:
                design = args.failure.rstrip("/").replace("/","_")
            outfile = "plots/%s_recall_vs_time.png" %(design)
            plot_recall_vs_time(all_base_points, all_new_points, outfile=outfile)
            plot_improvements("plots/%s_improvements.png" %(design), recall_auc_improvementz)

        print "Geometric mean recall auc improvement (%i failures): %.3f" %(len(recall_auc_improvementz),gmean(recall_auc_improvementz))
        print "Geometric mean relative runtime: %.3f" %(gmean(speedupz))
        
    elif analysis_method == 1:
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
        raise ValueError("Can't determine analysis method to use.")
    
    
def init(parser):
    parser.add_argument("new_suffix", default="", help="Suffix of failure names to compare against the baseline")
    parser.add_argument("base_suffix", nargs='?', default="", help="[optional] Suffix of failure names to compare against the baseline")
    parser.add_argument("--design", help="Design to analyze.")
    parser.add_argument("--failure", help="Analyze a single failure (base name).")    
    parser.add_argument("--min_runtime", type=int, default=0, help="Exclude designs with runtime less than this.")
    parser.add_argument("--method", default=-1, type=int, help="Type of analysis to perform. 0 for area under the " \
                        "recall-time curve; 1 for separate recall and time. If not specified, tries to infer the " \
                        "best method from the debug logs. ")
    parser.add_argument("-p", "--plot", action="store_true", default=False, help="Generate recall-time plot aggregated over all failures.")
    parser.add_argument("-pi", "--plot_individual", action="store_true", default=False, 
                        help="Generate recall-time plot for individual failrues")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)
