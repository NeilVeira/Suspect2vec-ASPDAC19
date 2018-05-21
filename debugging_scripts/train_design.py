import os 
import argparse 
import sys 
sys.path.insert(0,"suspect2vec")

import utils 
from suspect2vec import Suspect2Vec

def init(parser):
    parser.add_argument("design")
    
    
def main(args):
    all_failurez = utils.find_all_failures(args.design)
    all_suspectz = [utils.parse_suspects(failure) for failure in all_failurez]
    
    for i in range(len(all_failurez)):
        print all_failurez[i]
        train_data = all_suspectz[:i] + all_suspectz[i+1:]
        predictor = Suspect2Vec()
        predictor.fit(train_data)
        embeddingx = predictor.get_embeddings()
        with open(all_failurez[i]+"_embeddings.txt","w") as f:
            for key in embeddingx:
                f.write(key)
                for x in embeddingx[key]:
                    f.write(" %.6f" %(x))
                f.write("\n")
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    init(parser)
    args = parser.parse_args()
    main(args)