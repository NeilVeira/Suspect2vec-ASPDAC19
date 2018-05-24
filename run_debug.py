import os 
import argparse
import re
import signal
import random

import utils
        

def run_debug(name, verbose=False):
    print "Running debug on %s..." %(name)
    stdout,stderr = utils.run("onpoint-cmd --template-file=%s.template" %(name))
    
    if stdout == None:
        print "onpoint-cmd exceeded time limit of 48 hours"
        return False 
    
    #check logs for errors
    log_file = "onpoint-cmd-%s.log" %(name)
    if not os.path.exists(log_file):
        print "Error:"
        print stdout
        print stderr 
        return False        
    log = open(log_file).read()
    if "error:" in log.lower():
        print "vdb failed, check logs"
        return False        

    return True 
    
    
def analyze_suspect_prediction(base_failure, new_failure, verbose=False):
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
    
    if verbose:
        print "Number of true suspects: %i" %(len(old_suspectz))
        print "Number of found suspects: %i (recall %.3f)" %(len(new_suspectz), recall)
        print "Number of blocked suspects: %i out of %i" %(len(blocked),total_suspects)
        print "Relative runtime: %.3f" %(speedup)
    
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
            
    return recall, speedup

    
def main(base_template, new_name=None, min_suspects=999999, aggressiveness=0.5, skip_run=False, verbose=False):
    if not os.path.exists(base_template):
        raise ValueError("File %s does not exist" %(base_template))
    base_template = base_template.rstrip("/")
    dir = os.path.dirname(base_template)    
    orig_dir = os.getcwd()
    os.chdir(dir)
    name = os.path.basename(base_template)[:-len(".template")]  

    if new_name is None:
         success = run_debug(name, verbose=verbose)
         os.chdir(orig_dir)  
         return success
    
    else:   
        if not skip_run:
            with open("args.txt","w") as f:
                f.write("%i\n" %(min_suspects))
                f.write("%.3f\n" %(aggressiveness))
            assert os.system("cp %s_embeddings.txt embeddings.txt" %(name)) == 0
            #print os.getcwd(), "cp %s.template %s.template" %(name,new_name)
            assert os.system("cp %s.template %s.template" %(name,new_name)) == 0
            
            # Change project name 
            linez = open(new_name+".template").readlines()
            for i in range(len(linez)):
                if linez[i].startswith("PROJECT="):
                    linez[i] = "PROJECT=%s\n" %(new_name)
            with open(new_name+".template","w") as f:
                f.write("".join(linez))
        
            success = run_debug(new_name, verbose=verbose)
            if not success:
                os.chdir(orig_dir) 
                return False
                
        assert os.path.exists(new_name+".vennsawork")
        
        analyze_suspect_prediction(name, new_name, verbose=verbose)
            
        os.chdir(orig_dir)  
        return True
    
    
def init(parser):
    parser.add_argument("base_template", help="Path to template file to run onpoint-cmd")
    parser.add_argument("new_name", nargs='?', default=None, help="Name of new template file")
    parser.add_argument("--min_suspects", type=int, default=999999, 
        help="Minimum number of suspects to find before predicting")
    parser.add_argument("--aggressiveness", type=float, default=0.5, help="Threshold below which suspects are blocked")
    parser.add_argument("-v","--verbose", action="store_true", default=False, help="Display more info")
    parser.add_argument("--skip_run", action="store_true", default=False, 
        help="Skip running the debug and just do the suspect prediction analysis")
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args.base_template, args.new_name, args.min_suspects, args.aggressiveness, args.skip_run, args.verbose)
