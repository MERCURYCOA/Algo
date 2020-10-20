# 题一： 连接图

class Solution:
    def __init__(self, n):
        self.father = {}
        for i in range(1, n+1):
            self.father[i] = i 

    def find(self, node):   
        path = []     # 回溯
        cur = node
        while self.father[cur] != cur:
            path.append(cur)    # 中间节点都放到path里
            cur = self.father[cur] #指针指向下一个节点

        for n in path:    # 让中间节点的father都变成最后一个节点也就是祖先节点，存起来
            self.father[n] = cur

        return cur      # 当前node的father是指针最后停留的节点

    def connect(self, a, b):
        self.father[a] = self.find(b)   # 让a的父亲指向b的祖先

    def query(self, a, b):
        return self.find(a) == self.find(b)   # 看a的祖先和b的祖先是否相同

solution = Solution(5)
solution.connect(1,2)
solution.connect(2,3)
solution.connect(3,4)
solution.connect(4,5)
print(solution.find(1))  # 5
