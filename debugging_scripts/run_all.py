import os 
import argparse 

import utils 
import run_debug

    
def main(args):
    # TODO: add options for running debug with suspect prediction 
    unsuccessful = []
    all_failurez = utils.find_all_failures(args.design)
    if args.start is not None:
        for i in range(len(all_failurez)):
            if all_failurez[i].endswith(args.start):
                start = i 
                break 
        else:
            raise ValueError("No failure %s found" %(args.start))
    else:
        start = 0
        
    for i in range(start,len(all_failurez)):
        failure = all_failurez[i]
        template_file = failure+".template"
        print template_file
        success = run_debug.main(template_file)
        if not success:
            unsuccessful.append(failure)
    
    if len(unsuccessful) > 0:
        print "The following runs were unsuccessful:"
        for f in unsuccessful:
            print f 
        
    
def init(parser):
    parser.add_argument("design")
    parser.add_argument("--start",help="Failure to start running at")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)