import os 
import argparse 

import utils 
import run_debug


def find_failure(failure, all_failurez):
    for i in range(len(all_failurez)):
        if all_failurez[i].endswith(failure):
            return i 
    else:
        raise ValueError("No failure %s found" %(failure))
    
    
def main(args):
    # TODO: add options for running debug with suspect prediction 
    unsuccessful = []
    all_failurez = utils.find_all_failures(args.design)
    if args.start is not None:
        start = find_failure(args.start, all_failurez)
    else:
        start = 0
    if args.stop is not None:
        stop = find_failure(args.stop, all_failurez)
    else:
        start = len(all_failurez)-1
        
    for i in range(start,stop+1):
        failure = all_failurez[i]
        name = os.path.basename(failure)
        print failure
        if args.new_suffix is None:
            success = run_debug.main(failure, verbose=args.verbose)
        else:
            success = run_debug.main(failure, name+args.new_suffix, args.min_suspects, \
                args.aggressiveness, verbose=args.verbose)
        
        if not success:
            unsuccessful.append(failure)
    
    if len(unsuccessful) > 0:
        print "The following runs were unsuccessful:"
        for f in unsuccessful:
            print f 
        
    
def init(parser):
    parser.add_argument("design")
    parser.add_argument("new_suffix", nargs='?', default=None, help="Suffix to append to the name of the new project")
    parser.add_argument("--start",help="Failure to start running at")
    parser.add_argument("--stop",help="Failure to stop running at")
    parser.add_argument("--min_suspects", type=int, default=999999, help="Minimum number of suspects to "\
        "find before predicting")
    parser.add_argument("--aggressiveness", type=float, default=0.5, help="Threshold below which suspects are blocked")
    parser.add_argument("-v","--verbose", action="store_true", default=False, help="Display more info")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)