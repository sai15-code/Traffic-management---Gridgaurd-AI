from dataclasses import dataclass

@dataclass
class Candidate:
    name: str
    average_delay: float
    queue_severity: float
    blocked_intersection_penalty: float
    @property
    def cost(self):
        return .50*self.average_delay+.30*self.queue_severity+.20*self.blocked_intersection_penalty

def choose_best_action(candidates):
    candidates=list(candidates)
    if not candidates: raise ValueError("Provide at least one candidate.")
    return min(candidates,key=lambda candidate:candidate.cost)

if __name__=="__main__":
    options=[Candidate("Baseline",42,.75,.8),Candidate("Green-time adjustment",33,.58,.5)]
    print("Illustrative only:",choose_best_action(options).name)
