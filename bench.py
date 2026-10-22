from pathlib import Path
import json,os,subprocess,sys,statistics,time
root=Path(__file__).parent;python=os.environ["LCMX_PYTHON"]
trees={arm:Path(os.environ["LCMX_"+arm.upper()]) for arm in ("base","candidate","pre1628")}
workers={}
for arm,tree in trees.items():
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(tree)+":"+os.environ["LCMX_HOST_STUB"],TMPDIR=str(root))
 workers[arm]=subprocess.Popen([python,str(root/"bench_worker.py"),str(tree),str(root/("bench-home-"+arm))],cwd=tree,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=(root/("bench-"+arm+".stderr")).open("w"),text=True)
count=int(sys.argv[1]);samples=[];start=time.perf_counter()
try:
 with (root/("cost-"+str(count)+".jsonl")).open("w") as f:
  for n in [200,2000]:
   for path,pattern in [("rewind",True),("rewind",False),("steady",True),("steady",False)]:
    for sample in range(count+5):
     order=list(trees);offset=sample%3;order=order[offset:]+order[:offset]
     if sample%2:order.reverse()
     for arm in order:
      q=dict(n=n,path=path,pattern=pattern,sample=sample-5)
      w=workers[arm];w.stdin.write(json.dumps(q)+"\n");w.stdin.flush()
      line=w.stdout.readline()
      if not line:raise RuntimeError(arm+" worker stopped")
      d=dict(q,arm=arm,**json.loads(line))
      if sample>=5:f.write(json.dumps(d)+"\n");f.flush();samples.append(d)
     if sample%25==24:print(f"cost n={n} path={path} pattern={pattern} paired={sample-4}/{count} elapsed={time.perf_counter()-start:.1f}s",flush=True)
    summary={arm:statistics.median([x["ns"] for x in samples if x["n"]==n and x["path"]==path and x["pattern"]==pattern and x["arm"]==arm])/1e6 for arm in trees}
    print(json.dumps(dict(n=n,path=path,pattern=pattern,median_ms=summary)),flush=True)
finally:
 for w in workers.values():w.stdin.close();w.wait()
