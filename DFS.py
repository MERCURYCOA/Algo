# 题一：[2,3,6,7]找到所有和为7的数， 客重复使用
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, candidates, target):
        # write your code here
        results = []
        if len(candidates) == 0:
            return results
        
        candidates = sorted(list(set(candidates)))  # 去重 排序 才能用recursion
        self.recursion(candidates, target, 0,[],results)
        return results
    def recursion(self, candidates, remainTarget, index, current, results):
        # 递归出口   
        if remainTarget== 0:
                results.append(list(current))  # 将list加到list里面，必须用list()
        # 递归拆解：找到下一个需要加到current里的数
        for i in range(index, len(candidates)):

            if remainTarget < candidates[i]: # 当前值和之前current里得的数的和超过了target，比如：[2,2,2,2]，退出当前层的recursion函数，这时返回的是current是上一层的,比如：[2,2,2]
                break
            current.append(candidates[i])  # 如果当前值与current里得的数的和没有超过target， 将当前值加入current
            self.recursion(candidates, remainTarget - candidates[i], i, current, results)  # 下一层recursion的开始index还是i， 因为可以重复使用同一个数字
            current.pop() # 这里去掉current最后一个数，向后查看  比如[2,2,2,2]break之后， 返回的是[2,2,2]，这里去掉最后一个2， 在for循环中向后查看3
        
         
        
