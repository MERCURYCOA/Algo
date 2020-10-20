# 题一： 连接图I

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


# 题二：连接图 II

class Solution:
    def __init__(self, n):
        self.father = {}
        for i in range(1, n+1):
            self.father[i] = i 

    def find(self, node):
        path = []
        cur = node
        while self.father[cur] != cur:
            path.append(cur)
            cur = self.father[cur]

        for n in path:
            self.father[n] = cur

        return cur

    def connect(self, a, b):
        self.father[a] = self.find(b)

    def query(self, a):     # 图的本质是dict, 查询图其实是查询dict
        fa = self.find(a)
        count = 0
        for key, value in self.father.items():
            if value == fa:
                count += 1
            else:
                continue

        return count

solution = Solution(8)
solution.connect(1,2)
solution.connect(2,3)
solution.connect(3,4)
solution.connect(4,5)
solution.connect(7,8)

print(solution.find(1))
print(solution.query(7))
