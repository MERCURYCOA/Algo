# DFS 时间复杂度2^n

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
            self.recursion(candidates, remainTarget - candidates[i], i, current, results)  # 下一层recursion的开始index还是i， 因为可以重复使用同一个数字； 
            # 如果要求不能重复使用数字， i改成i+1
            current.pop() # 这里去掉current最后一个数，向后查看  比如[2,2,2,2]break之后， 返回的是[2,2,2]，这里去掉最后一个2， 在for循环中向后查看3
        
         
# 题二： calculate sumII  与I区别，原candidates不需要去重，只需排序，但是求出来的result需要去除， 比如 1' 1'' 1''' 选谁， 标准是需要一个1，选1', 需要2个1， 选1', 1''
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, candidates, target):
        results = []
    
        # 集合为空
        if len(candidates) == 0:
            return results
        
        # 利用set去重后排序
        candidatesNew=sorted(candidates)
        # dfs
        self.dfs(candidatesNew, 0, [], target, results)
    
        return results

    def dfs(self, candidatesNew, index, current, remainTarget,  results):
        # 到达边界
        if remainTarget == 0:
            results = results.append(list(current))
            return results

        # 递归的拆解：挑一个数放入current
        for i in range(index, len(candidatesNew)):
            if i!= 0 and candidatesNew[i] == candidatesNew[i-1] and i>index:  
                continue
            #results去重， 假设需要两个1， 但是candidaes有1', 1'' ,1''', 在选1‘， 1’‘时可以加进去，之后有current.pop()操作，然后index到1’‘’， current变成[1', 1'''],这是不允许的
            #这时 index 是 1‘， i-1是1’‘， i是1''', 符合上述if判断条件， continue，继续向后查看， 注意不是break, 因为后面的可能有2，3...
            # 剪枝
            if remainTarget < candidatesNew[i]:
                break

            current.append(candidatesNew[i])
            self.dfs(candidatesNew, i+1, current, remainTarget - candidatesNew[i], results)
            current.pop()
# 题三：字符串解码 expression expanding
#字符串递归
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def expressionExpand(self, s):
        if not s:
            return ""
        self.right = 0
        return self.dfs(s, 0, '')
        
    def dfs(self, s, index, res):
        while index < len(s):
            if s[index] == ']': 
                self.right = index
                return res
            elif s[index].isdigit():
                num = ''
                while s[index] != '[':   #这里把‘[’跳过去了，所以外面if条件里不会遇到‘[’
                    num += s[index]     # 数字在这里也是字符，所以要把多位数找全
                    index += 1 
                res += int(num) * self.dfs(s, index+1, '')
            else:
                res += s[index]
            index = 1 + max(index, self.right)
        return res
