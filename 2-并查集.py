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
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
    def query(self, a, b):
        return self.find(a) == self.find(b)   # 看a的祖先和b的祖先是否相同

solution = Solution(5)
solution.connect(1,2)
solution.connect(2,3)
solution.connect(3,4)
solution.connect(4,5)
print(solution.find(1))  # 5


# 题二：连接图 II

class ConnectingGraph2:
    """
    @param: n: An integer
    """
    def __init__(self, n):
        self.father = {}
        self.count = {}
        for i in range(1, n + 1):
            self.father[i] = i
            self.count[i] = 1

    """
    @param: a: An integer
    @param: b: An integer
    @return: nothing
    """
    def connect(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            self.count[root_b] += self.count[root_a]

    """
    @param: a: An integer
    @return: An integer
    """
    def query(self, a):
        return self.count[self.find(a)]

    def find(self, node):
        path = []
        while node != self.father[node]:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
            
        return node

solution = Solution(8)
solution.connect(1,2)
solution.connect(2,3)
solution.connect(3,4)
solution.connect(4,5)
solution.connect(7,8)

print(solution.find(1))
print(solution.query(7))

# 题三： 连接图III

class Solution:
    def __init__(self, n):
        self.father = {}
        self.size = n   # 实时维护一个size就是联通图个数，最初个数是n
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
        self.size -= 1          # 每次连接两个节点，联通图个数减1

    def query(self):
        return self.size        # 返回当前联通图个数

solution = Solution(8)
solution.connect(1,2)
solution.connect(2,3)
solution.connect(3,4)
solution.connect(4,5)
solution.connect(7,8)

print(solution.find(1))
print(solution.query())
