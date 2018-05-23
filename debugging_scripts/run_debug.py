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
    else:
        report,_ = utils.run("stdb %s.vennsawork/vennsa.stdb.gz report" %(name)) 
       
    if verbose:
        print report
    # if not check_solutions(report)[0]:
        # print "vdb did not find actual bug location"
        # return False
    # else:
        # print "vdb successful!"
        # return True
    return True 

    
def main(base_template, new_template=None, min_suspects=999999, aggressiveness=0.5, verbose=False):
    base_template = base_template.rstrip("/")
    dir = os.path.dirname(base_template)    
    orig_dir = os.getcwd()
    os.chdir(dir)
    name = os.path.basename(base_template)[:-len(".template")]  

    if new_template is not None:
        new_name = os.path.basename(new_template)[:-len(".template")]      
        with open("args.txt","w") as f:
            f.write("%i\n" %(min_suspects))
            f.write("%.3f\n" %(aggressiveness))
        os.system("cp %s_embeddings.txt embeddings.txt" %(name))
        os.system("cp %s.template %s.template" %(name,new_name))
        
        # Change project name 
        linez = open(new_name+".template").readlines()
        for i in range(len(linez)):
            if linez[i].startswith("PROJECT="):
                linez[i] = "PROJECT=%s\n" %(new_name)
        with open(new_name+".template","w") as f:
            f.write("".join(linez))
        name = new_name
    
    success = run_debug(name, verbose=verbose)
    
    os.chdir(orig_dir)    
    return success
    
    
def init(parser):
    parser.add_argument("base_template", help="Path to template file to run onpoint-cmd")
    parser.add_argument("new_template", nargs='?', default=None, help="Name of new template file")
    parser.add_argument("--min_suspects", type=int, default=999999, help="Minimum number of suspects to "\
        "find before predicting")
    parser.add_argument("--aggressiveness", type=float, default=0.5, help="Threshold below which suspects are blocked")
    parser.add_argument("-v","--verbose", action="store_true", default=False)
   
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args.base_template, args.new_template, args.min_suspects, args.aggressiveness, args.verbose)
