#TODO: move everything depending on onpoint and ugly verilog parsing out of this file 
import os 
import argparse
import subprocess
import re
import random
import math
import itertools
import numpy as np
import scipy.stats
import warnings 
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab
from matplotlib import gridspec

from suspect_prediction import SuspectPrediction

EPS = 1e-3
INF = 1e12

#define hierarchical levels of all rtl types
HIERARCHY = {"module":1, 
            "func":1,
            "input":4, 
            "inout":4,
            "wire":4, 
            "reg":4,
            "gate":4,
            "always":2,
            "if":2,
            "cond":3,
            "stmt":3,
            "block":2,
            "assign":3,
            "expr":3,
            "inst":2,
            "case":2,
            "for":2,
            "sequence":2,
            "assume":2,
            "property":2,
            "assert":2,
            "constraint":2 
            }
            
class Suspect(object):
    '''
    Class to encapsulate info of a suspect
    '''
    def __init__(self, filename, name, l, r, rtl_type, id):
        self.filename = filename
        self.name = name
        self.rtl_type = rtl_type 
        if not HIERARCHY.has_key(rtl_type):
            raise Exception("Unknown type %s (file %s: %s-%s)" %(rtl_type, filename,l,r))
        self.level = HIERARCHY[rtl_type]
        #self.level = name.count("/")+1
        self.l = l 
        self.r = r 
        self.id = id
        
    def __eq__(self, other):
        return self.name == other.name 
        
    def __lt__(self,other):
        return self.id < other.id
        
    def __str__(self):
        return "suspect %s: %s %s %s %s - %s" %(str(self.id).ljust(3), self.name.ljust(50), self.rtl_type.ljust(20), \
                self.filename.ljust(30),self.l.ljust(6),self.r.ljust(8))
        
        
class Failure(object):
    '''
    Class to encapsulate info of a failure
    '''
    def __init__(self, fail_dir, suspectz, timez, fail_time):
        self.fail_dir = fail_dir
        self.suspectz = suspectz
        self.timez = timez
        self.fail_time = fail_time
        self.module = None
        self.num_suspects = len(suspectz)
        self.num_suffix_suspects = 0
        self.runtime = 0 
        self.suffix_runtime = 0
        
    def __str__(self):
        return "Failure %s" %(self.fail_dir)
        
        
def do_prediction_random(sample, n, k, args):
    '''
    Random prediction mechanism, for comparison purposes. 
    '''
    if args.verbose >= 2:
        print "Doing random prediction with initial sample", sample
    ranking = list(sample)
    
    not_sample = []
    for j in range(n):
        if j not in ranking:
            not_sample.append(j)
    
    while len(ranking) < k:
        id = random.choice(not_sample)
        ranking.append(id)
        not_sample.remove(id)
        
    return ranking
        
        
def do_prediction_optimal(sample, n, k, target, args):
    '''
    Optimal prediction mechanism, for comparison purposes. 
    '''
    if args.verbose >= 2:
        print "Doing optimal prediction with initial sample", sample
    ranking = list(sample)
    
    #first predict all correct suspects
    for id in target:
        if len(ranking) >= k:
            break
        if id not in ranking:
            ranking.append(id)
            
    #predict remaining suspects arbitrarily until we have k 
    for i in range(n):
        if len(ranking) >= k:
            break
        if i not in ranking:
            ranking.append(i)
        
    return ranking
    
    
                
def evaluate_prediction(suspect_union, target_ids, ranking, est_size, args):
    '''
    Compute the metric measuring the precision of the prediction algorithm.
    This metric is the area under the curve of (% correct) vs (number of suspects). 
    suspect_union: list of Suspect objects sorted by id
    target_ids: list of ids of the full set of suspects 
    ranking: list of ids of the ranking suspects 
    '''      
    total = len(ranking)
    actual_size = len(target_ids)
    for i in range(total):
        assert 0 <= ranking[i] < len(suspect_union) #ranking valid suspect
        assert ranking[i] not in ranking[i+1:] #no suspect should be ranking twice  
        
    correct_cnt = 0
    incorrect_cnt = 0 
    mean_prec = 0 #mean precision over all sizes
    mean_rec = 0 #mean recall over all sizes
    mean_jac = 0 #mean jaccard index over all sizes
    exact_prec = 0 #precision at correct size 
    exact_rec = 0 #recall at correct size 
    exact_jac = 0 #jaccard index at correct size 
    est_jac = 0 #jaccard index at estimated size 
    
    for i in range(total):
        if ranking[i] in target_ids:
            correct_cnt += 1 
        else:
            incorrect_cnt += 1
        mean_prec += float(correct_cnt)/(i+1)          #precision = tp/(tp+fp)
        mean_rec += float(correct_cnt)/actual_size #recall = tp / (tp+fn)
        mean_jac += float(correct_cnt) / (actual_size + incorrect_cnt)
        if i == actual_size-1:
            exact_prec = float(correct_cnt) / (i+1)
            exact_jac = float(correct_cnt) / (actual_size + incorrect_cnt)
            exact_rec = float(correct_cnt) / actual_size
        if i == est_size-1:
            est_jac = float(correct_cnt) / (actual_size + incorrect_cnt)
            
    mean_prec /= total 
    mean_rec /= total
    mean_jac /= total
    
    size_err = float(abs(est_size-actual_size)) / actual_size
    
    if args.verbose >= 2:
        print "Estimated size:",est_size 
        print "Actual size:",len(target_ids)
        print "Inferred suspects:" 
        for id in ranking:
            assert suspect_union[id].id == id
            print suspect_union[id],
            if id in target_ids:
                print "correct"
            else:
                print "incorrect"                  
    if args.verbose >= 2:
        print "Means over all k:"
        print "    precision = %.3f" %(mean_prec)
        print "    recall = %.3f" %(mean_rec)
        print "    jaccard index = %.3f" %(mean_jac)
        print "At correct prediction size:"
        print "    precision = %.3f" %(exact_prec)
        print "    recall = %.3f" %(exact_rec)
        print "    jaccard index = %.3f" %(exact_jac)
        print "At estimated prediction size:"
        print "    jaccard index = %.3f" %(est_jac)
        print "    estimation error = %.3f" %(size_err)
        print ""
        
    return mean_prec, mean_rec, mean_jac, exact_prec, exact_rec, exact_jac, est_jac, size_err
        

def process_report(fail_dir, suspect_union=set([]), debug_level=INF):
    '''
    Parse the debug reports and other data from the given failure directory. 
    Create Suspect objects for each suspect found. Add unique suspects to suspect_union. 
    debug_level: maximum level of suspects to consider (all suspects at a deeper 
    level are ignored).
    Returns: Failure object containing all important data about the failure, or 
    None if unsuccessful due to missing data / bad path.
    '''
    suspect_path = os.path.join(fail_dir, "suspects.txt")
    time_path = os.path.join(fail_dir, "suspect_times.txt")
    suffix_path = os.path.join(fail_dir, "suffix_data.txt")
    if not (os.path.exists(suspect_path) and os.path.exists(time_path) and os.path.exists(suffix_path)):
        return None
    with open(suspect_path) as f:
        suspect_report = f.read()
    with open(time_path) as f:
        time_report = f.read()
    with open(suffix_path) as f:
        suffix_data = f.read()
        
    report_suspects = []
    timex = {}
    suspect_timez = []  
    
    #parse all available times from time_report 
    for m in re.findall(r"solution\s+\d+\s+([\w/]+)@(\d+)-\d+",time_report):
        if not timex.has_key(m[0]) or int(m[1]) > timex[m[0]]: #take maximum time
            timex[m[0]] = int(m[1])            
            
    m = re.search(r"Failure time: (\d+)", suffix_data)
    fail_time = int(m.group(1))
   
    # Parse suspect report 
    for suspect_parse in re.findall(r"rtl\s+([\w/]+)\s+(\w+)\s+([\w\./]+)\s+([\d\.]+)\s+([\d\.]+)", suspect_report, flags=re.DOTALL):
        s = Suspect(suspect_parse[2], suspect_parse[0], suspect_parse[3], \
            suspect_parse[4], suspect_parse[1], len(suspect_union))
        if s.level > debug_level:
            continue
            
        # Need to find item in suspect_union which is "equal" to s.
        # How to do a find efficiently on a set in python???
        for item in suspect_union:
            if item == s:
                s = item
                break
        else:
            suspect_union.add(s)
        
        # Try to determine a time for suspect s 
        if timex.has_key(s.name):
            suspect_timez.append((timex[s.name],s.id,s.name))
        else:                
            # Check if s.name is a prefix of any suspect in timex, and use the maximum
            # of all such suspects for s's time
            t = 0
            for suspect in timex.keys():
                if suspect.startswith(s.name+"/") and timex[suspect] > t:
                    t = timex[suspect]
            timex[s.name] = t 
            suspect_timez.append((t,s.id,s.name))
    
    suspect_timez.sort()
    suspect_timez.reverse()
    
    idz = [x[1] for x in suspect_timez]
    timez = [x[0] for x in suspect_timez]
    failure = Failure(fail_dir, idz, timez, fail_time)
    
    # Get info on suffix debugging. Note that "Number of suffix suspects: 0" means 
    # that no suffix debug data actually exists for this failure. 
    m = re.search(r"Number of suffix suspects: (\d+)", suffix_data)
    failure.num_suffix_suspects = int(m.group(1))
    m = re.search(r"Debug runtime: ([\.\d]+)", suffix_data)
    failure.runtime = float(m.group(1))
    m = re.search(r"Suffix debug runtime: ([\.\d]+)", suffix_data)
    failure.suffix_runtime = float(m.group(1))
    
    return failure

   
def experiment_k(design, all_suspectz, suspect_union, args):
    '''
    Evaluate suspect prediction over all values of k
    '''
    m = len(all_suspectz)
    n = len(suspect_union)
    mean_precs = np.zeros(m)
    all_rankingz = []
    rand_rankingz = []
    
    # Axis 0 of all_metrics (in order): 
    # mean precision, mean recall, mean jaccard index, exact precision, exact recall, exact jaccard index, estimated jaccard index, size error 
    # where "mean" = mean over all k, "exact" = value at true k, "estimated" = value at estimated k
    # Axis 1 of all metrics: Prediction, random, optimal 
    # Axis 2 of all metrics: result for each data point in leave-one-out evaluation
    all_metrics = np.zeros((8,3,m))
    est_sizes = np.zeros(m)
    '''mean_precs = np.zeros((3,m))
    mean_recs = np.zeros((3,m))
    mean_jacs = np.zeros((3,m))
    exact_precs = np.zeros((3,m))
    exact_recs = np.zeros((3,m))
    exact_jacs = np.zeros((3,m))
    est_jacs = np.zeros((3,m))
    size_errs = np.zeros((3,m))'''
    
    predictor = SuspectPrediction(args.prior_var)
    
    # leave-one-out evaluation
    for i in range(m):
        # Generate training & test data
        train_data = all_suspectz[:i] + all_suspectz[i+1:]
        test_data = all_suspectz[i]
        if args.sample_type == "random":
            random.shuffle(test_data)
        sample = test_data[:int(math.ceil(args.sample_size*len(test_data)))]
        
        # Prediction
        predictor.fit(train_data)
        pred,ranking = predictor.predict(sample, return_full_rank=True)        
        est_size = len(pred)
        est_sizes[i] = est_size
        
        # Comparison with random and optimal
        rand = do_prediction_random(sample, n, n, args)
        opt = do_prediction_optimal(sample, n, n, test_data, args)
        all_rankingz.append(ranking)
        rand_rankingz.append(rand)
        
        # Evaluation
        all_metrics[:,0,i] = evaluate_prediction(suspect_union, test_data, ranking, est_size, args)
        all_metrics[:,1,i] = evaluate_prediction(suspect_union, test_data, rand, est_size, args)
        all_metrics[:,2,i] = evaluate_prediction(suspect_union, test_data, opt, len(test_data), args)
        
        '''mean_prec, mean_rec, mean_jac, exact_prec, exact_rec, exact_jac, est_jac, size_err = evaluate_prediction(suspect_union, test_data, ranking, est_size, args)
        mean_precs[0][i] = mean_prec 
        mean_recs[0][i] = mean_rec 
        mean_jacs[0][i] = mean_jac
        exact_precs[0][i] = exact_prec 
        exact_recs[0][i] = exact_rec 
        exact_jacs[0][i] = exact_jac
        est_jacs[0][i] = est_jac
        size_errs[0][i] = size_err 
        mean_prec, mean_rec, mean_jac, exact_prec, exact_rec, exact_jac, est_jac, size_err = evaluate_prediction(suspect_union, test_data, rand, est_size, args)
        mean_precs[1][i] = mean_prec 
        mean_recs[1][i] = mean_rec 
        mean_jacs[1][i] = mean_jac
        exact_precs[1][i] = exact_prec 
        exact_recs[1][i] = exact_rec 
        exact_jacs[1][i] = exact_jac
        est_jacs[1][i] = est_jac
        size_errs[1][i] = size_err 
        mean_prec, mean_rec, mean_jac, exact_prec, exact_rec, exact_jac, est_jac, size_err = evaluate_prediction(suspect_union, test_data, opt, len(test_data), args)
        mean_precs[2][i] = mean_prec 
        mean_recs[2][i] = mean_rec 
        mean_jacs[2][i] = mean_jac
        exact_precs[2][i] = exact_prec 
        exact_recs[2][i] = exact_rec 
        exact_jacs[2][i] = exact_jac
        est_jacs[2][i] = est_jac
        size_errs[2][i] = size_err '''
        
    print "Overall stats (%i failures)" %(m)
    print "Means over all k:"
    print "    precision = %.3f" %(np.mean(all_metrics[0][0])) 
    print "        random = %.3f" %(np.mean(all_metrics[0][1]))
    print "        optimal = %.3f" %(np.mean(all_metrics[0][2]))
    print "    recall = %.3f" %(np.mean(all_metrics[1][0])) 
    print "    jaccard index = %.3f" %(np.mean(all_metrics[2][0])) 
    print "        random = %.3f" %(np.mean(all_metrics[2][1]))
    print "        optimal = %.3f" %(np.mean(all_metrics[2][2]))
    print "At exact k:"
    print "    precision = %.3f" %(np.mean(all_metrics[3][0])) 
    print "        random = %.3f" %(np.mean(all_metrics[3][1]))
    print "        optimal = %.3f" %(np.mean(all_metrics[3][2]))
    print "    recall = %.3f" %(np.mean(all_metrics[4][0])) 
    print "    jaccard index = %.3f" %(np.mean(all_metrics[5][0])) 
    print "        random = %.3f" %(np.mean(all_metrics[5][1]))
    print "        optimal = %.3f" %(np.mean(all_metrics[5][2]))
    print "At estimated k:"
    print "    jaccard_index = %.3f" %(np.mean(all_metrics[6][0]))
    print "        random = %.3f" %(np.mean(all_metrics[6][1]))
    print "        optimal = %.3f" %(np.mean(all_metrics[6][2]))
    print "    size estimation error = %.3f" %(np.mean(all_metrics[7][0]))
      
    '''print "    precision = %.3f" %(np.mean(mean_precs[0])) 
    print "        optimal = %.3f" %(np.mean(mean_precs[2]))
    print "        random = %.3f" %(np.mean(mean_precs[1]))
    print "    recall = %.3f" %(np.mean(mean_recs[0])) 
    print "    jaccard index = %.3f" %(np.mean(mean_jacs[0])) 
    print "        optimal = %.3f" %(np.mean(mean_jacs[2]))
    print "        random = %.3f" %(np.mean(mean_jacs[1]))
    print "At exact k:"
    print "    precision = %.3f" %(np.mean(exact_precs[0])) 
    print "        optimal = %.3f" %(np.mean(exact_precs[2]))
    print "        random = %.3f" %(np.mean(exact_precs[1]))
    print "    recall = %.3f" %(np.mean(exact_recs[0])) 
    print "    jaccard index = %.3f" %(np.mean(exact_jacs[0])) 
    print "        optimal = %.3f" %(np.mean(exact_jacs[2]))
    print "        random = %.3f" %(np.mean(exact_jacs[1]))
    print "At estimated k:"
    print "    jaccard_index = %.3f" %(np.mean(est_jacs[0]))
    print "        optimal = %.3f" %(np.mean(est_jacs[2]))
    print "        random = %.3f" %(np.mean(est_jacs[1]))
    print "    size estimation ratio = %.3f" %(np.mean(size_errs[0]))'''
        
    '''if args.plots:
        #make plots
        n = len(all_rankingz[0]) #assumes prediction was run to full 
        opt = np.zeros(n)
        rand = np.zeros(n)
        pred = np.zeros(n)
        
        #compute accuracy *for all* k
        for fail_id in range(m):
            correct_cnt = 0 
            rand_correct_cnt = 0         
            for i in range(n):
                opt[i] += float(min(i+1,len(all_suspectz[fail_id]))) / (i+1)
                if all_rankingz[fail_id][i] in all_suspectz[fail_id]:
                    correct_cnt += 1
                pred[i] += float(correct_cnt)/(i+1)
                if rand_rankingz[fail_id][i] in all_suspectz[fail_id]:
                    rand_correct_cnt += 1
                rand[i] += float(rand_correct_cnt)/(i+1)
        
        opt = opt/m * 100 
        pred = pred/m * 100 
        rand = rand/m * 100
        
        print "making plots"
        ax = plt.subplot(111, xlabel='k', ylabel='Accuracy (%)', title=design)
        ax.plot(opt, "-g", label="Optimal")
        ax.plot(pred, "-r", label="Predicted")
        ax.plot(rand, "-b", label="Random")
        ax.legend(loc="upper right")
        ax.set_ylim([0,110])
        ax.set_xlim([0,len(all_rankingz[0])+1])
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        plt.tight_layout()
        matplotlib.pylab.savefig("plots/acc_vs_k/%s_acc_vs_k.png" %(design))
        plt.close()
        
        #plot set size estimation 
        zipped = [(len(all_suspectz[i]),est_sizes[i]) for i in range(len(est_sizes))]
        zipped.sort()
        exact_sizez = [zipped[i][0] for i in range(len(zipped))]
        est_sizez = [zipped[i][1] for i in range(len(zipped))]
        ax = plt.subplot(111, xlabel='Failure #', ylabel='Suspect set size', title=design)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        ax.plot(exact_sizez, '.g', label="Exact", markersize=18)
        ax.plot(est_sizez, 'xr', label="Estimated", markersize=12)
        ax.legend(loc="upper left", numpoints=1)
        plt.tight_layout()
        matplotlib.pylab.savefig("plots/sizes/%s_sizes.png" %design)
        plt.close()'''
       

def update_plot(design, xdata, ydata, data_file, plot_file, xlabel, do_sample_stuff=False):
    #read existing data for other designs
    all_datax = {}
    if os.path.exists(data_file):
        for line in open(data_file):
            prev_design,prev_data = line.split(",")
            prev_data = map(float,prev_data.split())
            all_datax[prev_design] = prev_data 
    all_datax[design] = ydata
    
    #write all data back to file 
    f = open(data_file,"w")
    for design in all_datax.keys():
        f.write(design+",")
        f.write(" ".join(map(str, all_datax[design])))
        f.write("\n")
    f.close()
         
    #make plots 
    if do_sample_stuff:
        fig, axarr = plt.subplots(2, sharex=False, figsize=(8,12), gridspec_kw={'height_ratios':[3,2]})
        axarr[0].set_title("(a)")
        ax1 = axarr[0]
    else:
        fig, ax1 = plt.subplots()  
        
    numpoints = max([len(all_datax.values()[i]) for i in range(len(all_datax))])
    mean = np.zeros(numpoints)
    cnt = np.zeros(numpoints)
    xdata = np.linspace(xdata[0], xdata[0]+(xdata[1]-xdata[0])*(numpoints-1), numpoints)
    ax1.set_xlabel(xlabel)  
    ax1.set_ylabel("Prediction accuracy (%)")
    ax1.set_xlim((0,xdata[len(xdata)-1]+xdata[0]))
    ax1.set_ylim((60,100)) 
    
    c = 0 
    colourz = ["y","g","r","c","m","b","k"]
    markerz = ["o","^","s"]
    stylez = [":"] 
    for design in all_datax.keys():
        y = np.array(all_datax[design])*100  
        style = stylez[c%len(stylez)]
        colour = colourz[c%len(colourz)]
        marker = markerz[c%len(markerz)]
        ax1.plot(xdata[:len(y)], y, style+colour+marker, linewidth=2.0, label=design, markersize=4) 
        for i in range(len(y)):
            cnt[i] += 1
            mean[i] += y[i]
        c += 1
    mean /= cnt
    ax1.plot(xdata, mean, "-k", label="mean", linewidth=3.0)
        
    small_legend_size = 12
    if do_sample_stuff:
        ax1.plot(xdata, xdata, "--k")
        ax1.legend(loc="lower right", prop={'size': small_legend_size}) 
        
        #plot runtimes vs sample size
        datax = load_runtime_data()
        all_x = []
        all_y = []
        for d in datax.keys():
            all_x.extend(datax[d][0])
            all_y.extend(datax[d][1])
        binx = []
        biny = []
        for i in range(9):
            bin_low = 0.05 + 0.1*i
            bin_high = bin_low+0.1 
            vals = []
            for j in range(len(all_x)):
                if bin_low <= all_x[j] < bin_high:
                    vals.append(all_y[j])
            binx.append((bin_low+bin_high)/2 * 100)
            biny.append(scipy.stats.mstats.gmean(vals))  
        #ax1.plot(all_x, all_y, "xg")
        axarr[1].set_title("(b)")
        axarr[1].set_xlabel(xlabel)
        axarr[1].set_xlim((0,100))
        axarr[1].set_ylabel("Suffix debug vs full debug runtime")
        axarr[1].set_ylim((0,0.5))
        axarr[1].plot(binx, biny, "-ob", label="runtime") 
        axarr[1].legend(loc="upper right", numpoints=1) 

        #plot triage data 
        ax2 = axarr[1].twinx()
        ax2.set_ylabel("Error in binning NMI")
        mean = np.zeros(8)
        num_designs = 0
        for line in open("nmi_scores.txt"):
            line = line.split(",")
            des = line[0]
            scores = np.array(map(float,line[1:-1]))
            correct = float(line[-1])
            err = abs(scores-correct)/correct
            #plt.plot(x, err, "--k")
            mean += err 
            num_designs += 1
        mean /= num_designs
        x = np.linspace(10,10*len(mean),len(mean))
        ax2.set_xlim((0,100))
        ax2.set_ylim((0,0.5))
        ax2.plot(x, mean, "-sr", label="mean error in NMI")
        ax2.legend(loc="upper left", numpoints=1)
        
    else:    
        plt.legend(loc="lower right", prop={'size': small_legend_size}) 
            
    matplotlib.pylab.savefig(plot_file)
    plt.close()
    
    
def load_runtime_data():
    datax = {}
    data_cache = open("runtime_data.txt")
    for line in data_cache:
        d,x,y  = line.strip().split(";")
        d = d.strip()
        datax[d] = (eval(x), eval(y))
    return datax       
       
def plot_runtimes(design, all_failurez):
    xdata = []
    ydata = []
    for f in all_failurez:
        if f.num_suffix_suspects != 0 and f.num_suffix_suspects < f.num_suspects and f.runtime != 0 and f.suffix_runtime != 0:
            xdata.append(float(f.num_suffix_suspects)/f.num_suspects)
            ydata.append(float(f.suffix_runtime)/f.runtime)
            
    datax = load_runtime_data()
    datax[design] = (xdata,ydata)
    
    data_cache = open("runtime_data.txt","w")
    for d in datax.keys():
        data_cache.write("%s ; %s ; %s\n" %(d,str(datax[d][0]), str(datax[d][1])))
    data_cache.close()    
            
    '''plt.xlabel("Sample size")
    plt.ylabel("suffix debug vs full debug runtime")
    plt.xlim((0,1))
    #plt.ylim((50,100))
    all_x = []
    all_y = []
    for d in datax.keys():
        all_x.extend(datax[d][0])
        all_y.extend(datax[d][1])
    
    binx = []
    biny = []
    for i in range(9):
        bin_low = 0.05 + 0.1*i
        bin_high = bin_low+0.1 
        vals = []
        for j in range(len(all_x)):
            if bin_low <= all_x[j] < bin_high:
                vals.append(all_y[j])
        binx.append((bin_low+bin_high)/2 * 100)
        biny.append(scipy.stats.mstats.gmean(vals))
        
    f = open("runtime_bins.txt","w")
    f.write(str(binx)+"\n")
    f.write(str(biny)+"\n")
    f.close()
        
    plt.plot(all_x, all_y, "xg")
    plt.plot(binx, biny, "ob")
    matplotlib.pylab.savefig("plots/runtimes.png")
    plt.close()'''
       
        
def experiment_sample_size(design, all_suspectz, suspect_union, map_weights, args, all_failurez):
    '''
    '''
    m = len(all_suspectz)
    n = len(suspect_union)
    sample_interval = 0.1
    num_points = int(0.8/sample_interval)+1
    data = np.zeros((m,num_points+1))
    predicted_suspectz = [[] for _ in range(num_points)] #suspect lists for each failure, for each sample size
    
    for i in range(m):
        train_data = all_suspectz[:i] + all_suspectz[i+1:]
        weights = get_weights(train_data, n, map_weights)
        test_data = all_suspectz[i]
        data[i][-1] = 1.0 #perfect accuracy at full sample
        
        for j in range(num_points):
            sample_size = 0.1 + j*sample_interval
            sample = test_data[:int(math.ceil(sample_size*len(test_data)))]
            ranking, est_size = do_prediction_spbp(weights, sample, n, i, args)
            mean_prec, mean_rec, mean_jac, exact_prec, exact_rec, exact_jac, est_jac, size_err = evaluate_prediction(suspect_union, test_data, ranking, n, args)
            #TODO: don't use exact_prec... maybe est_jac?
            data[i][j] = exact_prec 
            predicted_suspectz[j].append(ranking[:est_size])
            
    x = np.linspace(10,100,num_points+1)
    cur_data = np.mean(data,axis=0)
    for i in range(num_points):
        print "At %i%% sample: %.3f" %(int(x[i]), cur_data[i])
        
    if args.plots:
        update_plot(design, x, cur_data, "sample_size_data.txt", "plots/vs_sample_size.png", "Sample size (%)", True)
        
        
def experiment_train_size(design, all_suspectz, suspect_union, map_weights, args):
    m = len(all_suspectz)
    n = len(suspect_union)
    train_sizez = range(5,len(all_suspectz)-1,5)
    data = np.zeros((len(all_suspectz), len(train_sizez)))
    
    for i in range(len(all_suspectz)):
        train_data = all_suspectz[:i] + all_suspectz[i+1:]
        random.shuffle(train_data)
        test_data = all_suspectz[i]
        sample = test_data[:int(math.ceil(args.sample_size*len(test_data)))]
        
        for j in range(len(train_sizez)):
            weights = get_weights(train_data[:train_sizez[j]], n, map_weights) 
            ranking, scorez = do_prediction_spbp(weights, sample, n, i, args)
            mean_prec, mean_rec, mean_jac, exact_prec, exact_rec, exact_jac, est_jac, size_err = evaluate_prediction(suspect_union, test_data, ranking, n, args)
            data[i][j] = exact_prec
    
    cur_data = np.mean(data, axis=0)
    for i in range(len(cur_data)):
        print "Accuracy at training size %i: %.3f" %(5*(i+1), cur_data[i])
    
    if args.plots:
        update_plot(design, np.linspace(5,80,16), cur_data, "train_size_data.txt", "plots/vs_train_size.png", "Training set size, T")

            
def main(args):
    design_dir = args.design_dir.rstrip("/") 
    debug_level = args.level
    if not os.path.exists(design_dir):
        raise ValueError("design %s does not exist" %(design_dir))
        return False     
        
    design = os.path.basename(design_dir).strip() #TODO: this is too hacky
    suspect_union = set([]) 
    all_suspectz = []
    all_failurez = []
    num_bugs = 0
    
    # Parse debug reports to extract suspect sets
    print "Reading suspects..."
    for bug_dir in sorted(os.listdir(design_dir)):  
        num_bugs += 1     
        # parse bug module 
        bug_desc = open(os.path.join(design_dir,bug_dir,"bug_desc.txt")).read()
        module = re.search(r"Module:\s+(\w+)",bug_desc).group(1)
        
        for item in sorted(os.listdir(os.path.join(design_dir,bug_dir))):
            fail_dir = os.path.join(design_dir, bug_dir, item)
            if os.path.isdir(fail_dir):                    
                failure = process_report(fail_dir, suspect_union, debug_level)
                if failure is not None:
                    failure.module = module
                    if args.verbose:
                        print "Parsed",failure
                    all_suspectz.append(failure.suspectz)
                    all_failurez.append(failure)
            
    '''def cmp_by_id(a, b):
        if a.id == b.id:
            return 0
        elif a.id < b.id:
            return -1 
        else:
            return 1'''
    suspect_union = list(suspect_union)
    cmp_by_id = lambda a,b: 0 if a.id == b.id else (-1 if a.id < b.id else 1)
    suspect_union.sort(cmp_by_id) 
    n = len(suspect_union)
    m = len(all_suspectz)
    
    print "Number of bugs:",num_bugs
    print "Number of failures:",m
    print "Total number of suspects across all bugs:",n
    if args.verbose:
        print "Suspect union:"
        for item in suspect_union:
            print item
        print ""
        # print "All suspect sets:"
        # for sset in all_suspectz:
            # print sorted(sset)
        # print ""
    
    if args.experiment == "k":
        experiment_k(design, all_suspectz, suspect_union, args)
    # TODO: fix this functionality 
    # elif args.experiment == "sample_size" or args.experiment == "triage":
        # experiment_sample_size(design, all_suspectz, suspect_union, map_weights, args, all_failurez)
    # elif args.experiment == "train_size":
        # experiment_train_size(design, all_suspectz, suspect_union, map_weights, args)
    # elif args.experiment == "runtime":
        # plot_runtimes(design, all_failurez)    
    else:
        raise ValueError("Invalid experiment option %s" %(args.experiment))
    
        
def init(parser):
    parser.add_argument("design_dir",help="Design to run")
    parser.add_argument("--experiment",default="k",help="Type of evaluation experiment to run on the design")
    parser.add_argument("--level",type=int,default=INF,help="Maximum hierarchical level of suspects to consider. Default is all.")
    parser.add_argument("--sample_size",type=float,default=0.5 ,help="Number of suspects in initial subset (sample) of suspect set that" \
                        " is to be ranking.")
    parser.add_argument("--prior_var",type=float,default=0.2,help="Hyperparameter for prior in MAP estimation")
    parser.add_argument("-v","--verbose",action="store_true",default=False,help="Verbose mode")
    parser.add_argument("--plots",action="store_true",default=False,help="Generate plots")
    parser.add_argument("--sample_type",default="suffix")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)

    