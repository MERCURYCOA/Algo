# 7个节点的二叉树深度是多少？ 最坏是7,都长在左枝或都长在右枝，最好是log7,就是每个节点都有两个枝。
# 二叉树一般用递归，递归最重要的是深度，深度太深会stack overflow
# 前序，中序，后序 指的是根所在的位置 前序 - 根左右， 中序 - 左根右， 后序 - 左右根。历遍要到达最后的叶子才行。 讲解： https://blog.csdn.net/qq_33243189/article/details/80222629
# 遍历二叉树的方法：迭代和递归， 递归包含分治法和遍历法。注意：递归和迭代都包含在深度搜索dfs中。
#
# 题一： 前序遍历二叉树
# 分治法和遍历法的区别：分治法 - 每次递归都创建一个新的result列表，最终结果是把所有列表连接起来。 遍历法 - 全局变量result， 每次递归结果都放到这个列表中。
# 所以分治法用result.extend, 遍历法用result.append
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
