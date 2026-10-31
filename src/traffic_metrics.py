from dataclasses import dataclass

@dataclass
class TrafficMetrics:
    vehicle_count: int
    roi_vehicle_count: int
    occupancy: float
    queue_growth: float
    downstream_congestion: float = 0.0

def calculate_metrics(vehicle_centres, width, height, roi=(.15,.45,.85,.98),
                      previous_roi_count=None, downstream_congestion=0.0):
    x1,y1,x2,y2=roi
    points=list(vehicle_centres)
    count=sum(x1*width <= x <= x2*width and y1*height <= y <= y2*height for x,y in points)
    occupancy=min(count/10.0,1.0)
    growth=0.0 if previous_roi_count is None else max(0.0,min(1.0,(count-previous_roi_count)/max(previous_roi_count,1)))
    return TrafficMetrics(len(points),count,occupancy,growth,max(0.0,min(1.0,downstream_congestion)))
