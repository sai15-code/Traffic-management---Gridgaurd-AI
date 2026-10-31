"""Optional SUMO metrics collector. SUMO must be installed separately."""
import argparse,os,sys,csv

def run_simulation(config_path,output_csv,steps=1000,gui=False):
    if not os.path.isfile(config_path): raise FileNotFoundError(f"SUMO config not found: {config_path}")
    home=os.environ.get("SUMO_HOME")
    if home and os.path.isdir(os.path.join(home,"tools")) and os.path.join(home,"tools") not in sys.path:
        sys.path.append(os.path.join(home,"tools"))
    try: import traci
    except ImportError as exc: raise RuntimeError("TraCI unavailable. Install SUMO separately and configure SUMO_HOME.") from exc
    traci.start(["sumo-gui" if gui else "sumo","-c",config_path,"--no-step-log","true"])
    rows=[]
    try:
        for step in range(steps):
            traci.simulationStep(); ids=traci.vehicle.getIDList()
            waiting=sum(traci.vehicle.getWaitingTime(v) for v in ids)
            speed=sum(traci.vehicle.getSpeed(v) for v in ids)/len(ids) if ids else 0
            rows.append({"step":step,"vehicles_in_network":len(ids),"total_waiting_seconds":round(waiting,2),"mean_speed_mps":round(speed,3)})
            if traci.simulation.getMinExpectedNumber()<=0: break
    finally: traci.close()
    os.makedirs(os.path.dirname(output_csv) or ".",exist_ok=True)
    with open(output_csv,"w",newline="",encoding="utf-8") as f:
        fields=list(rows[0]) if rows else ["step","vehicles_in_network","total_waiting_seconds","mean_speed_mps"]
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--output",default="results/sumo_metrics.csv")
    p.add_argument("--steps",type=int,default=1000); p.add_argument("--gui",action="store_true"); args=p.parse_args()
    print(f"Saved {len(run_simulation(args.config,args.output,args.steps,args.gui))} rows.")
