# 内容一：二叉树的宽度搜索  - 标配：queue  
# 内容二：图的宽度搜索（拓扑排序 Topological Sorting）  - 标配：hashmap
# 棋盘上的宽搜 BFS

# 什么时候应该使用BFS?

# 1， 图的遍历 Traversal in Graph
#    • 层级遍历 Level Order Traversal
#    • 由点及面 Connected Component
#    • 拓扑排序 Topological Sorting

# 2， 最短路径 Shortest Path in Simple Graph • 仅限简单图求最短路径
#    • 即，图中每条边长度都是1，且没有方向
# 注意：最长路径用动态规划求解


# 题一：binary tree level order traversal  #因为是level order, 所以每一层都得是一个list,不能是把所有integer放进一个list里面
# 用deque, popleft()时间复杂度O(1)

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

from collections import deque
class Solution:
    """
    @param root: A Tree
    @return: Level order a list of lists of integer
    """
    def levelOrder(self, root):
        # write your code here
        if not root:
            return []
            
        result = []
        queue = deque([root])  # 注意不能直接deque(root), root应该是list才能在下面进行for循环
        while queue:
            level = []
            level_size = len(queue)  # 用level size控制下一层pop出来几个node
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return result
   

# 题二： 跟题一倒过来，从最底层到root进行level order traversal
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
from collections import deque
class Solution:
    """
    @param root: A tree
    @return: buttom-up level order a list of lists of integer
    """
    def levelOrderBottom(self, root):
        # write your code here
        if not root:
            return []
            
        result = []
        queue = deque([root])
        while queue:
            level = []
            level_size = len(queue)
            for _ in range(level_size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            result.append(level)
        return list(reversed(result))  #注意不要用list.reverse(),那样返回为空
                level.append(node.val)
                level.append(node.val)
# 题三：二叉树的锯齿形层次遍历
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""
from collections import deque
class Solution:
    """
    @param root: A Tree
    @return: A list of lists of integer include the zigzag level order traversal of its nodes' values.
    """
    def zigzagLevelOrder(self, root):
        if not root:
            return []
            
        res = []
        queue = deque([root])
        count = True
        while queue:
            level = []
            size = len(queue)
            for _ in range(size):
                node = queue.popleft()
                level.append(node.val)
                if node.left:
                   queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            if count == True:
                res.append(level)
            else: 
                res.append(list(reversed(level)))  # 此题最重要警示： reversed(A)不是list， 需要强转成list。
            count = bool(1-count) 
            
        return res
                
            

# 题三：二叉树序列化和反序列化：
# 序列化的意思是把各种object（树，图，integer,bouble等）变成string，以方便网络传输和内存变外存，外存恢复内存。 这里是把树结构变成[1,2,3,#,4]的形式

# 序列化：
# 将根加入队列。
# 每取出队首元素，将左右子节点加入队尾，直到队列为空。
# 将BFS序转换为题目要求的字符串。

# 反序列化：
# 将data分割成节点值。
# 令root为第一个值对应的节点。
# 将root加入队列。
# 每当队列非空：
# 令level_size等于当前队列节点个数。
# 执行level_size次，从队列中取出节点，并将接下来两个节点值连接到节点上。
# 返回root。
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None


from collections import deque
class Solution:
    
    def serialize(self, root):
        # write your code here
        if not root:
            return []
        result = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if type(node) is TreeNode:
                result.append(node.val)
                if node.left:
                    queue.append(node.left)
                else:
                    queue.append(None)
                if node.right:
                    queue.append(node.right)
                else:
                    queue.append(None)
            else:
                result.append("#")
        return result
   
    def deserialize(self, data):
        if not data:
            return None 
        n = len(data)
        root = self.get_treenode(data[0])
        queue = collections.deque([root])
        pointer = 1
        
        # 分层进行宽度优先搜索
        # 获得该层的节点个数level_size，并将接下来level_size * 2个值连接到节点上
        while queue:
            if pointer < n:
                level_size = len(queue)
                for _ in range(level_size):
                    node = queue.popleft()
                    
                    node.left = self.get_treenode(data[pointer])
                    if node.left is not None:
                        queue.append(node.left)
                    pointer += 1
                    if pointer == n:
                        break
                    
                    node.right = self.get_treenode(data[pointer])
                    if node.right is not None:
                        queue.append(node.right)
                    pointer += 1
                    if pointer == n:
                        break
            else:
                break
        return root
    
    def get_treenode(self, s):
        if s == '#':
            return None
        else:
            return TreeNode(int(s))
# 题四：判断图为树  
# 条件： 节点数n，边的个数必须为n-1； 各点联通
# 代码的体现： len(edges) == n-1, 访问过的节点数==n， 用queue储存当前节点所连接的点并且还没被访问到，用visited数组记录访问过的节点，最后判断len(visited) == n
class Solution:
    """
    @param n: An integer
    @param edges: a list of undirected edges
    @return: true if it's a valid tree, or false
    """
    def validTree(self, n, edges):
        # write your code here
        if len(edges) != n-1:
            return False
        neighbors = collections.defaultdict(list)
        for k,v in edges:
            neighbors[k].append(v)
            neighbors[v].append(k)
        
        queue = [0]
        visited = []
        while queue:
            cur = queue.pop()
            visited.append(cur)
            queue+= [x for x in neighbors[cur] if x not in visited]
        
        return len(visited) == n
# 题五： copy graph  # 注意：这里的copy指，node需要全新地址的node, 只有数跟之前的一样
# 三步走： BFS copy所有节点， connect新的所有节点
from collections import deque

class UndirectedGraphNode:
     def __init__(self, x):
         self.label = x
         self.neighbors = []

class Solution:
    """
    @param node: A undirected graph node
    @return: A undirected graph node
    """
    def getNodes(self, node): # 通过BFS遍历所有节点，这里返回的节点还是old node
        queue = deque([node])
        result = set([node])
        while queue:
            n = queue.popleft()
            result.add(n)
            queue += [neighbor for neighbor in n.neighbors if neighbor not in result]
        return result
    
    def cloneGraph(self, node):
        # write your code here
        if not node:
            return node
        root = node 
        nodes = self.getNodes(node)
        
        # copy nodes 
        mapping = {}  # 创建字典, 在old node和 new old之间创建映射
        for node in nodes:
            mapping[node] = UndirectedGraphNode(node.label) #以old node为key, 创建new node作为value
        
        # connect nodes
        for node in nodes:
            new_code = mapping[node]  
            for neighbor in node.neighbors:    #这里要重点理解  1， 找到node的邻居节点， 注意neighbor也是节点！！！
                new_neighbor = mapping[neighbor]   #2， mapping[neighbor]其实就是mapping[node]， 这里的意思是找到当前old node的neighbor节点的映射new node
                new_code.neighbors.append(new_neighbor)  #举例：mapping = {old_node0 [neighbor:old_node2]: new_node0, old_node1: new_node1,old_node2: new_node2}
        return mapping[root]                             # 现在要给new_node0找new_neighbor, 因为old_node0的neighbor是old_node_2,所以new_node0的neighbor一定是new_old0
                                                         # 但是怎么找到new_node2呢？已知 old_node0的neighbor:old_node2,并且mapping[old_node2] = new_code2
                                                         # 所以当前new_code0的new_neighbor 是 mapping[neighbor]，本质上也就是mapping[old_node2]
    
# 题六： 拓扑排序 - 针对给定的有向图找到任意一种拓扑排序的顺序.
#给定一个有向图，图节点的拓扑排序定义如下:
#对于图中的每一条有向边 A -> B , 在拓扑排序中A一定在B之前.
#拓扑排序中的第一个节点可以是图中的任何一个没有其他节点指向它的节点.
#关键是：节点的入度必须在节点出现之前，相当于先修课程
class DirectedGraphNode:
    def __init__(self, x):
        self.label = x
        self.neighbors = []


import collections

class Solution:
    """
    @param: graph: A list of Directed graph node
    @return: Any topological order for the given graph.
    """
    def topSort(self, graph):
        node_to_indegree = self.get_indegree(graph)

        # bfs
        order = []
        start_nodes = [n for n in graph if node_to_indegree[n] == 0]
        queue = collections.deque(start_nodes)
        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] -= 1  # 找到一个入度就在字典中减去一个入读
                if node_to_indegree[neighbor] == 0: #当入度全部找到时，该节点加到queue中，继续遍历这个节点的neighbors, 同时因为入度为0，说明指向节点的节点都已经加到order了，所以也要将该节点加到order里
                    queue.append(neighbor)
        return order
    
    def get_indegree(self, graph):
        node_to_indegree = {x: 0 for x in graph}

        for node in graph:
            for neighbor in node.neighbors:
                node_to_indegree[neighbor] += 1
                # neighbor是本节点指向的那些点，不包括被指向的点，因为是有向图
                #这里统计每个节点的neighbor节点的入度，就是计算每个节点被多少个节点
                #入度为0的节点都可以作为开头节点
        return node_to_indegree
# 题七： 重构序列 - 判断给出的org是不是seq可以唯一重构的图
# 步骤： 1， build graph, 将给出的seq建成图，其实就是字典{node: neighbors[]}
# 2, 找到每个节点的入度
# 3， 通过topological sort对节点进行序列化， 判断order是否唯一，如果唯一，order长度是否等于org, 如果长度等于，判断是不是完全相等
class Solution:
    """
    @param org: a permutation of the integers from 1 to n
    @param seqs: a list of sequences
    @return: true if it can be reconstructed only one or false
    """
    def sequenceReconstruction(self, org, seqs):
        # write your code here
        graph = self.build_graph(seqs)  #通过seq建graph
        topo_order = self.topological_sort(graph) # 通过graph找到唯一order
        return topo_order == org  # 判断order 和 org是不是相等
        
    def build_graph(self, seqs):
        graph = {}
        for seq in seqs:
            for node in seq:
                if node not in graph:
                    graph[node] = set()
                    
        for seq in seqs:
            for i in range(1, len(seq)):
                graph[seq[i-1]].add(seq[i])
        return graph
    
    def get_indegrees(self, graph):
        indegrees = {node: 0 for node in graph}
        for node in graph:
            for neighbor in graph[node]:
                indegrees[neighbor] += 1 
        return indegrees
        
    def topological_sort(self, graph):
        indegrees = self.get_indegrees(graph)
        queue = []
        for node in graph:
            if indegrees[node] == 0:
                queue.append(node)
                
        order = []
        while queue:
            if len(queue) > 1:   #如果queue里面同时有两个及以上节点，说明同一层有两个及以上节点，也就是有两个及以上路径，那么order一定不唯一，所以直接返回None
                return None
            node = queue.pop()
            order.append(node)
            for neighbor in graph[node]:
                indegrees[neighbor] -= 1 
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
        if len(order) == len(graph):
            return order 
        return None
 
# 题八：岛的个数： 0是海， 1是岛， 两个1连在一起是一个岛
from collections import deque

DIRECTIONS = [(1, 0), (0, -1), (-1, 0), (0, 1)] #四个方向坐标变化

class Solution:
    """
    @param grid: a boolean 2D matrix
    @return: an integer
    """
    def numIslands(self, grid):
        if not grid or not grid[0]:
            return 0
            
        islands = 0
        visited = set()
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1 and (i, j) not in visited: # not visited说明这个1周边没有1，可以算一个岛
                    self.bfs(grid, i, j, visited)  #作用是遍历(i,j)坐标周边，把1四周的坐标放到visited里，因为1周围的1不能再算一次
                    islands += 1
                    
        return islands                    
    
    def bfs(self, grid, x, y, visited):
        queue = deque([(x, y)])
        visited.add((x, y))
        while queue:
            x, y = queue.popleft()
            for delta_x, delta_y in DIRECTIONS:
                next_x = x + delta_x
                next_y = y + delta_y
                if not self.is_valid(grid, next_x, next_y, visited):  #注意这里valid有3个条件：边界内，没有访问过，值为1
                    continue
                queue.append((next_x, next_y))  #只有1周围如果有1，放到queue里面，查看后面这个1是不是周围还有1， 因为相邻的1只能算1个岛
                visited.add((next_x, next_y))

    def is_valid(self, grid, x, y, visited):
        n, m = len(grid), len(grid[0])
        if not (0 <= x < n and 0 <= y < m):
            return False
        if (x, y) in visited:
            return False
        return grid[x][y]  # 这里有隐含条件， grid[x][y] == 1。 如果当前节点是0，not valid, 是1， valid
