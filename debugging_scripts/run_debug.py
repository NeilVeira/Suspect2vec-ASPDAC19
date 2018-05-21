import os 
import argparse
import re
import subprocess32 as subprocess
import signal
import random

import utils
        

def init(parser):
    parser.add_argument("base_template",help="Path to template file to run onpoint-cmd")
    parser.add_argument("new_template", help="Name of new template file")
    parser.add_argument("--min_suspects", type=int, default=10, help="Minimum number of suspects to "\
        "find before predicting")
    parser.add_argument("--aggressiveness", type=float, default=0.5, help="Threshold below which suspects are blocked")
    
    
def main(base_template, new_template, min_suspects=10, aggressiveness=0.5):
    base_template = base_template.rstrip("/")
    dir = os.path.dirname(base_template)    
    orig_dir = os.getcwd()
    os.chdir(dir)
    name = os.path.basename(base_template)[:-len(".template")]    
    
    with open("args.txt","w") as f:
        f.write("%i\n" %(min_suspects))
        f.write("%.3f\n" %(aggressiveness))
    os.system("cp %s_embeddings.txt embeddings.txt" %(name))
    os.system("cp %s.template %s" %(name,new_template))
    
    
    
    os.chdir(orig_dir)
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args.base_template, args.new_template, args.min_suspects, args.aggressiveness)
