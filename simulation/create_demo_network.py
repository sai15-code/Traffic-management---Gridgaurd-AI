"""Generate a small illustrative SUMO network; requires netconvert from SUMO."""
import os,shutil,subprocess
HERE=os.path.dirname(os.path.abspath(__file__))
nodes=os.path.join(HERE,"demo.nod.xml"); edges=os.path.join(HERE,"demo.edg.xml")
net=os.path.join(HERE,"demo.net.xml"); routes=os.path.join(HERE,"demo.rou.xml"); config=os.path.join(HERE,"demo.sumocfg")
def main():
    if not shutil.which("netconvert"): raise SystemExit("netconvert not found. Install SUMO and add it to PATH.")
    node_xml="""<nodes><node id="a" x="0" y="0"/><node id="b" x="200" y="0" type="traffic_light"/><node id="c" x="400" y="0"/></nodes>"""
    edge_xml="""<edges><edge id="ab" from="a" to="b" numLanes="2" speed="13.9"/><edge id="ba" from="b" to="a" numLanes="2" speed="13.9"/><edge id="bc" from="b" to="c" numLanes="2" speed="13.9"/><edge id="cb" from="c" to="b" numLanes="2" speed="13.9"/></edges>"""
    route_xml="""<routes><vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" maxSpeed="13.9"/><route id="r1" edges="ab bc"/><flow id="f1" type="car" route="r1" begin="0" end="300" vehsPerHour="400"/></routes>"""
    for path,content in [(nodes,node_xml),(edges,edge_xml),(routes,route_xml)]:
        with open(path,"w",encoding="utf-8") as f:f.write(content)
    subprocess.run(["netconvert","--node-files",nodes,"--edge-files",edges,"--output-file",net],check=True)
    with open(config,"w",encoding="utf-8") as f:
        f.write(f'<configuration><input><net-file value="{os.path.basename(net)}"/><route-files value="{os.path.basename(routes)}"/></input><time><begin value="0"/><end value="350"/></time></configuration>')
    print("Created",config)
if __name__=="__main__": main()
