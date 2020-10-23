# 扫描线
# 事件先排序，再扫描
#题一： 数飞机
"""
Definition of Interval.
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param airplanes: An interval array
    @return: Count of airplanes are in the sky.
    """
    def countOfAirplanes(self, airplanes):
        if not airplanes:
            return 0 
        new_airplanes = []    
        for e in airplanes:   
            new_airplanes.append((e.start, 1))
            new_airplanes.append((e.end, 0))
            
        new_airplanes.sort()  # 一定记得排序
        count = 0
        max_ = 0
        for e in new_airplanes:
            if e[1] == 0:
                count -= 1 
            if e[1] == 1:
                count += 1 
            max_ = max(max_, count)
        return max_
