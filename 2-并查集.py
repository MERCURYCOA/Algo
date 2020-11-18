# 模版1: 找到祖先，路径压缩
 def find(self, node):
        path = []
        while self.father[node] != node:
            path.append(node)
            node = self.father[node] 
            
        for n in path:
            self.father[n] = node 
        return node 

# 模版2: 合并祖先
 def connect(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            
            
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
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
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
# 题四： number of islands
# find和union  记住
class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """

                
    def numIslands(self, grid):
        if not grid or grid[0] is None:
            return 0
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        n, m = len(grid), len(grid[0])
        self.father = {}  #不可以用visited, 因为要union必须访问visited
        self.islands = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    self.father[(i,j)] = (i, j)
                    self.islands += 1
        for x in range(n):
            for y in range(m):
                for i in range(4):
                    if grid[x][y]:
                        x_ = x + dx[i]
                        y_ = y + dy[i]
                        if x_ >= 0 and x_ < n and y_ >= 0 and y_ < m and grid[x_][y_] == 1:
                            self.union((x, y), (x_, y_))
                        
        return self.islands
                
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            self.islands -= 1 
                    
    def find(self, point): # point = (x, y)
        path = []
        while self.father[point] != point:
            path.append(point)
            point = self.father[point] 
            
        for p in path:
            self.father[p] = point
            
        return point

# 题五： 岛屿的个数II
"""
Definition for a point.
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
"""



class Solution:
    """
    @param n: An integer
    @param m: An integer
    @param operators: an array of point
    @return: an integer array
    """
    def numIslands2(self, n, m, operators):
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]
        res = []
        visited = set()
        self.father = {}
        self.size = 0 
        for point in operators:
            x, y = point.x, point.y   # 转还成熟悉的x, y
            if (x, y) in visited:
                res.append(self.size)  # 重复的点也要加进当前的self.size
                continue 
            self.father[(x, y)] = (x, y)
            visited.add((x, y))
            self.size += 1 
            for i in range(4):
                x_ = x + dx[i]
                y_ = y + dy[i]
                if (x_, y_) in visited:
                    self.union((x, y), (x_, y_))
                    
            res.append(self.size)
        return res
        
        
        
    def union(self, point_a, point_b):
        root_a = self.find(point_a)
        root_b = self.find(point_b)
        if root_a != root_b:
            self.father[root_a] = root_b
            self.size -= 1
        
    def find(self, point):
        path = []
        while point != self.father[point]:
            path.append(point)
            point = self.father[point]
            
        for p in path:
            self.father[p] = point
            
        return point
