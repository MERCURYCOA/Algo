# 内容一：树的宽度搜索  - 标配：queue  # 内容二：图的宽度搜索  - 标配：hashmap

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
        return result
   
# 用两个list
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
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
        queue = [root]
        while queue:
            level = []
            result.append([node.val for node in queue])
            for node in queue:
                if node.left:
                    level.append(node.left)
                if node.right:
                    level.append(node.right)
            queue = level
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

# 题三：二叉树序列化和反序列化：
# 序列化的意思是把各种object（树，图，integer,bouble等）变成string，以方便网络传输和内存变外存，外存恢复内存。 这里是把树结构变成[1,2,3,#,4]的形式
# 关键 处理None节点
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
        # write your code here
        if not data:
            return None
    
        root = TreeNode(data[0])
        queue = deque([root])
        pos = 1
        while queue and pos < len(data)-1:
            node = queue.popleft()
            if data[pos] == '#':
                node.left = None
            else:
                node.left = TreeNode(data[pos])
                queue.append(node.left)
            if data[pos +1] == '#':
                node.right = None
            else:
                node.right = TreeNode(data[pos+1])
                queue.append(node.right)
            pos += 2
        
        if queue and pos == len(data)-1:
            node = queue.popleft()
            if data[pos] == '#':
                node.left = None
            else:
                node.left = TreeNode(data[pos])
        return root
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
