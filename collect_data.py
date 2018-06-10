import os 
import argparse
import re 

import utils

def main(args):   
    for failure in utils.find_all_failures(args.design):
        suspectz = utils.parse_suspects(failure+args.suffix)
        # check whether the failure+suffix instance finished, if not use failure 
        end = utils.find_time_of(failure+args.suffix, "\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*  VDB Process Ends  \*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*")
        if end:
            log_file = failure+args.suffix+".vennsawork/logs/vdb/vdb.log"
        else:
            log_file = failure+".vennsawork/logs/vdb/vdb.log"
            
        
        ordered_suspectz = []
        for line in open(log_file):
            pattern = r"##  ==> solver solution:  .*:([\w/]+)"
            m = re.search(pattern,line)
            if m:
                s = m.group(1)
                assert s not in ordered_suspectz
                ordered_suspectz.append(s)
        print len(suspectz),len(ordered_suspectz)
        
        with open("suspects.txt","w") as f:
            for s in ordered_suspectz:
                f.write(s+"\n")
        utils.copy_file("suspects.txt", failure.replace("designs","suspect_lists")+"_suspects.txt")
       
            
        
    
    
def init(parser):
    parser.add_argument("design")
    parser.add_argument("suffix", nargs='?', default="", help="Suffix to append to the name of the project")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)