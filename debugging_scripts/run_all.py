import os 
import argparse 

import utils 
import run_debug

    
def main(args):
    # TODO: add options for running debug with suspect prediction 
    
    for failure in utils.find_all_failures(args.design):
        template_file = failure+".template"
        print template_file
        success = run_debug.main(template_file)
        assert success
        
    
def init(parser):
    parser.add_argument("design")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)