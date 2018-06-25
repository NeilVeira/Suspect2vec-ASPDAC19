import os 
import argparse 
import re
import numpy as np

import utils
import run_debug     
    
def add_to_git(args):
    for failure in utils.find_all_failures(args.design):
        template_file = os.path.basename(failure)+".template"
        bug_dir = os.path.dirname(failure)
        # print bug_dir 
        
        for item in [template_file, "rtl", "bug_desc.txt", "tb", "filelist.l"]:
        # for item in [template_file]:
            cmd = "git add -f %s/%s" %(bug_dir,item)
            print cmd 
            os.system(cmd)


def mean_suspect_set_size(args):
    num_suspectz = []
    for failure in utils.find_all_failures(args.design, include_failed=False):
        suspect_list = failure.replace("designs","suspect_lists") + "_suspects.txt"
        cnt = len(open(suspect_list).readlines())
        num_suspectz.append(cnt)
    print np.mean(num_suspectz)/2

    
def main(args):
    mean_suspect_set_size(args)

    # for failure in utils.find_all_failures(args.design):
        # print failure 
        # t = utils.parse_runtime(failure)
        # print t 
    
    # for failure in utils.find_all_failures(args.design, include_failed=True):        
    #    utils.write_template(failure+".template", "--max=1", "GENERAL_OPTIONS=\"%s\"" %run_debug.VDB_OPTIONS)
    
def init(parser):
    parser.add_argument("design")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)
