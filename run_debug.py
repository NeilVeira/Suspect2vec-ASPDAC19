import os 
import argparse
import re
import signal
import random

import utils
import analyze

METHODS = [None,"block","assump","opt_assump","rev_assump","assump_block"]
VDB_OPTIONS = "--max=1 --rtl-implications=no --suspect-implications=none --oracle-solver-stats=debug --oracle-problem-stats=debug --skip-hard-suspects=no --time-diagnosis=no --diagnose-command=rtl --suspect-types=all"

def run_debug(name, timeout=60*60*24, verbose=False):
    print "Running debug on %s..." %(name)
    log_file = "onpoint-cmd-%s.log" %(name)
    if os.path.exists(log_file):
        os.remove(log_file)
        
    stdout,stderr = utils.run("onpoint-cmd --template-file=%s.template" %(name), timeout=timeout)    
    if stdout == None:
        print "onpoint-cmd exceeded time limit of %i seconds" %(int(timeout))
        return False 
    
    #check logs for errors
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
    
    
def main(base_name, new_name=None, min_suspects=999999, aggressiveness=0.5, guidance_method=None, 
    timeout=60*60*24, pass_timeout=300, verbose=False):
    if not os.path.exists(base_name+".template"):
        raise ValueError("File %s does not exist" %(base_name+".template"))
    dir = os.path.dirname(base_name)    
    orig_dir = os.getcwd()
    os.chdir(dir)
    base_name = os.path.basename(base_name)

    with open("args.txt","w") as f:
        f.write("%i\n" %(min_suspects))
        f.write("%.3f\n" %(aggressiveness))
        f.write("%i\n" %METHODS.index(guidance_method))
            
    if new_name is None:
        success = run_debug(base_name, timeout=timeout, verbose=verbose)
        os.chdir(orig_dir)  
        return success
    
    else:               
        assert os.system("cp %s_input_embeddings.txt input_embeddings.txt" %(base_name)) == 0
        assert os.system("cp %s_output_embeddings.txt output_embeddings.txt" %(base_name)) == 0
        assert os.system("cp %s.template %s.template" %(base_name,new_name)) == 0
        
        true_suspectz = utils.parse_suspects(base_name)
        with open("true_suspects.txt","w") as f:
            f.write("\n".join(true_suspectz))                
        
        # Modify template file as needed 
        utils.write_template(new_name+".template", "PROJECT=", "PROJECT="+new_name)
        general_options = VDB_OPTIONS + " --solver-cpu-limit=%i" %(pass_timeout)
        utils.write_template(new_name+".template", "GENERAL_OPTIONS=", "GENERAL_OPTIONS=\"%s\"" %(general_options))
    
        success = run_debug(new_name, timeout=timeout, verbose=verbose)
        if not success:
            os.chdir(orig_dir) 
            return False
                
        assert os.path.exists(new_name+".vennsawork")
        num_suspects = utils.parse_suspects(new_name)
        if len(num_suspects) == 0:
            return False 
        
        if guidance_method == "block":
            analysis_func = analyze.blocking_analysis
        elif guidance_method == "assump" or guidance_method == "opt_assump" or guidance_method == "rev_assump" \
            or guidance_method == "assump_block":
            analysis_func = analyze.assumption_analysis
        else:        
            os.chdir(orig_dir)  
            return True
            
        try:
            if "_1pass" in new_name:
                analysis_func(base_name+"_1pass", new_name, verbose=verbose, min_runtime=0)
            else:
                analysis_func(base_name, new_name, verbose=verbose, min_runtime=0)
        except:
            os.chdir(orig_dir)
            return False 
            
        os.chdir(orig_dir)  
        return True
    
    
def init(parser):
    parser.add_argument("base_name", help="Name of baseline failure.")
    parser.add_argument("new_name", nargs='?', default=None, help="Name of new failure to run. " \
        "If not specified base_name will be run.")
    parser.add_argument("--min_suspects", type=int, default=999999, 
        help="Minimum number of suspects to find before predicting")
    parser.add_argument("--aggressiveness", type=float, default=0.5, help="Threshold below which suspects are blocked")
    parser.add_argument("-v","--verbose", action="store_true", default=False, help="Display more info")
    parser.add_argument("--timeout", type=int, default=3600, help="Time limit in seconds for a single debugging run.")
    parser.add_argument("--pass_timeout", type=int, default=4000, help="Time limit in seconds for a single pass.")
    parser.add_argument("--method", type=str, default=None, help="Solver guidance method. " \
        "Must be one of %s" %METHODS)
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args.base_name, args.new_name, args.min_suspects, args.aggressiveness, args.method, args.timeout, 
        args.pass_timeout, args.verbose)
