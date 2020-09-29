# 7个节点的二叉树深度是多少？ 最坏是7,都长在左枝或都长在右枝，最好是log7,就是每个节点都有两个枝。
# 二叉树一般用递归，递归最重要的是深度，深度太深会stack overflow
# 前序，中序，后序 指的是根所在的位置 前序 - 根左右， 中序 - 左根右， 后序 - 左右根。历遍要到达最后的叶子才行。 讲解： https://blog.csdn.net/qq_33243189/article/details/80222629
# 遍历二叉树的方法：迭代和递归， 递归包含分治法和遍历法。注意：递归和迭代都包含在深度搜索dfs中。
#
# 题一： 前序遍历二叉树
# 分治法和遍历法的区别：分治法 - 每次递归都创建一个新的result列表，最终结果是把所有列表连接起来。 遍历法 - 全局变量result， 每次递归结果都放到这个列表中。
# 所以分治法用result.extend, 遍历法用result.append
# 递归
# 方法一：分治法
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def preorderTraversal(self, root):
        result = []
        if root == None:
            return result
        
        left = self.preorderTraversal(root.left)
        right = self.preorderTraversal(root.right)
        
        #前序
        result.append(root.val)
        result.extend(left)   # extend与append区别： extend是把两个list连接起来; append是把后面的list作为一个元素放到前面的list里。
        result.extend(right)
        #中序
        #result.extend(left)
        #result.append(root.val)
        #result.extend(right)
        #后序
        #result.extend(left)
        #result.extend(right)
        #result.append(root.val)
        return result

Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.preorderTraversal(Node1))

# 遍历：
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def postorderTraversal(self, root, result):  # 注意：遍历的参数有2个，root和result
        if root == None:
           return []
        result.append(root.val)
        self.postorderTraversal(root.left, result)
        self.postorderTraversal(root.right,result)       
        #中序
        # self.postorderTraversal(root.left, result)
        # result.append(root.val)
        # self.postorderTraversal(root.right,result)
        #后序
        # self.postorderTraversal(root.left, result)
        # self.postorderTraversal(root.right,result)
        # result.append(root.val)
        
        return result
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.postorderTraversal(Node1, []))
# 方法二：非递归
# 用stack/ queue

# 后序（一）非常巧妙
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def postorderTraversal(self, root):
        stack = [root]
        while stack:
            t = stack.pop()
            if type(t) is TreeNode:
                stack.append(t.val)  # 这里最tricky, 之前t被pop出，这里重新存入的是t.val而不是t。前一步判断type(t) is TreeNode，作用是不让t.val重新进入循环，而是直接print出来
                if t.right:
                    stack.append(t.right)
                if t.left:
                    stack.append(t.left)
            else:
                print(t)

Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.postorderTraversal(Node1)
#后序（二）
 class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def postorderTraversal(self, root):
        if root == None:
            return []
        stack = []
        res = []
        while stack or root:
            while root:
                stack.append(root)
                if root.left:
                    root = root.left
                else:
                    root = root.right
            s = stack.pop()
            res.append(s.val)
            if stack and s == stack[-1].left:
                root = stack[-1].right
            else:
                root = None
        return res
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.postorderTraversal(Node1))

# 前序
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def preorderTraversal(self, root):
        if root == None:
            return []
        stack = [root]
        res = []
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        return res
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.preorderTraversal(Node1))
    
# 中序
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def inorderTraversal(self, root):
        if root == None:
            return []
        stack = []
        res = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.inorderTraversal(Node1))      

#题二：求二叉树最大深度
# 分治法：
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def max_depth(self, root):
        if root == None:
           return 0
        left = self.max_depth(root.left)
        right = self.max_depth(root.right)
        
        return max(left, right)+1
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.max_depth(Node1))

#题三：求所有路径
# 递归和迭代的区别：递归调用自身，迭代不调用自身而用stack去遍历二叉树所有节点
# 递归法
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def all_paths(self, root):
        paths = []
        if root == None:
           return paths
        
        if root.left == None and root.right == None:
            paths.append(str(root.val))
            return paths
        paths = []
        left = self.all_paths(root.left)
        right = self.all_paths(root.right)

        for path in left + right:
            paths.append(str(root.val) + '->' + path)

        return paths
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.all_paths(Node1))
      
# 题四：Minimum subtree, 注意subtree不是指每个node, node.left, node.right, 是每个node一直到叶子
import sys
class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class Solution:
    def findSubtree(self, root):
        minimum_sum, subtree, sum_of_root = self.helper(root)
        return subtree
    def helper(self, root):
        if root is None:
            return sys.maxsize, None, 0

        left_min, left_subtree, left_sum = self.helper(root.left)
        right_min, right_subtree, right_sum = self.helper(root.right)
        
        sum_of_root = left_sum + right_sum + root.val
        
        if left_min == min(left_min, right_min, sum_of_root):
            return left_min, left_subtree, sum_of_root
        if right_min == min(left_min, right_min, sum_of_root):
            return right_min, right_subtree, sum_of_root
        
        return sum_of_root, root, sum_of_root
    
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.findSubtree(Node1))

# 判断平衡树
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if this Binary tree is Balanced, or false.
    """
    def isBalanced(self, root):
        # write your code here
        balanced, max_height = self.helper(root)
        return balanced
        
    def helper(self, root):
        if root is None:
            return True, 0
            
        l_balanced, leftHeight = self.helper(root.left) # 必须记录高度才能计算左右子树高度之差，这里helper重要的思想是 返回两个类型的值  bool 和 int
        r_balanced, rightHeight = self.helper(root.right)
        if not (l_balanced and r_balanced):
            return False, 0
        
        if abs(leftHeight - rightHeight) > 1:
            cur_balance = False
        else:
            cur_balance = True
        return  cur_balance, max(leftHeight, rightHeight) + 1
