from dataclasses import dataclass

@dataclass
class RiskResult:
    score: float
    level: str
    explanation: str

def calculate_risk(queue_severity, occupancy, queue_growth, downstream_congestion):
    q,o,g,d=[max(0.0,min(1.0,float(v))) for v in (queue_severity,occupancy,queue_growth,downstream_congestion)]
    score=.35*q+.25*o+.20*g+.20*d
    level="LOW" if score<.40 else ("MEDIUM" if score<.70 else "HIGH")
    reasons=[]
    if q>=.6: reasons.append("high queue severity")
    if o>=.6: reasons.append("high queue-region occupancy")
    if g>=.3: reasons.append("queue appears to be growing")
    if d>=.6: reasons.append("downstream congestion")
    return RiskResult(round(score,3),level,", ".join(reasons) or "no single indicator is currently high")

if __name__=="__main__":
    print(calculate_risk(.7,.8,.5,.6))
