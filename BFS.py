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
