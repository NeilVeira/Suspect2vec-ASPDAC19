import os 
import re 
import subprocess32 as subprocess

def run(cmd, verbose=False, timeout=5*60*60*24):
    if verbose:
        print cmd
    try:
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
        stdout, stderr = proc.communicate(None, timeout=timeout)
        return stdout,stderr
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        return None,None   
        
def find_all_failures(dir):
    results = []
    for item in sorted(os.listdir(dir)):
        if item.startswith("random_bug") or item.startswith("buggy"):             
            for sub_item in sorted(os.listdir(os.path.join(dir,item))):
                m = re.match(r"fail_\d+\.vennsawork\Z", sub_item)
                if m:
                    results.append(os.path.join(dir, item, sub_item[:-len(".vennsawork")]))
    return results 
                
                
def parse_suspects(failure):
    cache_path = os.path.join(failure+".vennsawork", "suspect_cache.txt")
    stdb_path = os.path.join(failure+".vennsawork","vennsa.stdb.gz")
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) > os.path.getmtime(stdb_path):
        report = open(cache_path).read()
    else:
        report, _ = run("stdb %s report" %(stdb_path))
        with open(cache_path,"w") as f:
            f.write(report)
    
    suspectz = []
    for suspect_parse in re.findall(r"rtl\s+([\w/]+)\s+(\w+)\s+([\w\./]+)\s+([\d\.]+)\s+([\d\.]+)", report, flags=re.DOTALL):
        suspectz.append(suspect_parse[0])
    return suspectz 
        