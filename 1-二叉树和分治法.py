# 树状时间复杂度是O(N), 不是logN,因为logN只有完全二叉树满足。 时间复杂度 = 节点树*每个节点处理时间O(1)，空间复杂度是O(h),h是深度
# 用O(N)时间，把n的问题，分成n/2的问题，总时间复杂度是NlogN
# 用O（1）时间，把n的问题，分成n/2的问题，总时间复杂度是O(N)
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

# 题五：判断平衡树
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

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
# 题六：最大平均数子树
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: the root of binary tree
    @return: the root of the maximum average of subtree
    """
    
    
    def findSubtree2(self, root):
        # write your code here
        if not root:
            return root
        max_ave_subtree, max_average, num_node = self.helper(root)
        return max_ave_subtree
    
    def helper(self, root):
        if root == None:
            return None, 0, 0

        left_subtree, left_average, left_num = self.helper(root.left)
        right_subtree, right_average, right_num = self.helper(root.right)
        
        cur_average = (root.val + (left_average*left_num) + (right_average*right_num))/(1+left_num+right_num)
        
        if left_subtree and left_average == max(left_average, right_average, cur_average):
            return left_subtree, left_average, left_num+right_num+1   #注意子树的数量是增加的， 不能用left_num   # 哪些是变的，哪些不变，要清楚
        if right_subtree and right_average == max(left_average, right_average, cur_average):
            return right_subtree, right_average, right_num+left_num+1   
        return root, cur_average, left_num+right_num+1  
# 题七：最近公共祖先 I
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
class Solution:
    """
    @param: root: The root of the binary search tree.
    @param: A: A TreeNode in a Binary.
    @param: B: A TreeNode in a Binary.
    @return: Return the least common ancestor(LCA) of the two nodes.
    """
    def lowestCommonAncestor(self, root, A, B):
        # write your code here
        if not root:
            return None
        if root is A or root is B:   # 处理当前节点
            return root
        
        left = self.lowestCommonAncestor(root.left, A, B)
        right = self.lowestCommonAncestor(root.right, A, B)
        
        if not left and not right: # 对left和right分类讨论
            return None
        if left and right:
            return root
        if not left and right:
            return right
        if left and not right:
            return left
        return None
# 题八 最近公共祖先 III  # 比I多加了一个限制，判断有没有a, 有没有B
class TreeNode:
    def __init__(self, val):
        this.val = val
        this.left, this.right = None, None

class Solution:
    """
    @param: root: The root of the binary tree.
    @param: A: A TreeNode
    @param: B: A TreeNode
    @return: Return the LCA of the two nodes.
    """
    def lowestCommonAncestor3(self, root, A, B):
        have_a, have_b, parent = self.dfs(root, A, B)
        if have_a and have_b:
            return parent
        else:
            return None

    def dfs(self, root, A, B):
        if not root:
            return False, False, None 
        
        
        left_have_a, left_have_b, left = self.dfs(root.left, A, B)
        right_have_a, right_have_b, right = self.dfs(root.right, A, B)

        have_a = left_have_a or right_have_a or root == A 
        have_b = left_have_b or right_have_b or root == B
        if A == root or B == root:
            return have_a, have_b, root
       
        if left and right:
            return have_a, have_b, root
        if left and not right:
            return have_a, have_b, left 
        if right and not left:
            return have_a, have_b, right
        return False, False, None
 
 
# 题九： 最近公共祖先 II   #建立parent set
parentSet = set()
        # 把A的祖先节点都加入到哈希表中
        curr = A
        while (curr is not None):
            parentSet.add(curr)
            curr = curr.parent
        # 遍历B的祖先节点，第一个在哈希表中出现的即为答案
        curr = B
        while (curr is not None):
            if (curr in parentSet):
                return curr
            curr = curr.parent
        return None
# 题十： 验证二叉查找树   
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
import sys

class Solution:
    """
    @param root: The root of binary tree.
    @return: True if the binary tree is BST, or false
    """
    def isValidBST(self, root):
        # write your code here
        validate, _, _ = self.helper(root)
        return validate
    def helper(self, root):    
        if not root:
            return True, - sys.maxsize-1, sys.maxsize  # 这里拿只有一个节点时举例子，当root.left和root.right为None时，需要返回什么
        
        left_validate, left_max, left_min = self.helper(root.left)
        right_validate, right_max, right_min = self.helper(root.right)
        
        if left_validate and right_validate: 
            return left_validate and right_validate and left_max < root.val and root.val < right_min, max(right_max, root.val), min(left_min, root.val)
        return False, max(right_max, root.val), min(left_min, root.val)  #这里的返回值根据isValidBST需要的返回值确定
     
 
 # 题十一：左右反转二叉树
 """
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def invertBinaryTree(self, root):
        # write your code here
        if not root:
            return root 
        left = self.invertBinaryTree(root.left)
        right = self.invertBinaryTree(root.right)

        root.left = right
        root.right = left
        return root   
 # 题十二： 上下左右翻转二叉树
 """
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    题意理解： 如果有右孩子，那么一定存在左孩子。
    翻转的意思是：如果有2个孩子，让左孩子做新的根节点，右孩子做新根的左孩子，旧根做新根的右孩子。
    如果只有左孩子，那么让左孩子做新根，让旧根做新根的右孩子。

    @param root: the root of binary tree
    @return: new root
    """
    def upsideDownBinaryTree(self, root):
        if not root:
            return None 
        return self.dfs(root)

    def dfs(self, root):  # 输入为本层旧根
        if root.left == None:  # 如果有右孩子，那么一定有左孩子，这里判断，如果左孩子是None, 那么此根就一定没有孩子，是叶子
            return root

        newroot = self.dfs(root.left) # 深度优先：先到达最深处，做翻转，之后向上逐层翻转

        root.left.right = root  # root.left要做本层的新根，那么其右孩子=本层的旧根root
        root.left.left = root.right  # 本层新根的左孩子(root.left.left)是旧根的右孩子root.right
        root.left = None   # 将旧根与原孩子断开，称为叶子
        root.right = None 
              # 上面的过程可以理解为：根左右按顺时针顺序向前走一步
        return newroot  # 返回本层新根
 
 # 题十三：二叉树转单链表
 class TreeNode:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None
class ListNode:
    def __init__(self,val):
        self.val = val
        self.next = None

class Solution:
    def BTtoLinkedList(self, root):
        return self.helper(root)
    def helper(self, root):
        if not root:
            return None
        left = self.helper(root.left)
        right = self.helper(root.right)
        root_node = ListNode(root.val)
        if not left and not right:
            return root_node
        if left and right:
            cur = left
            while cur.next:
                cur = cur.next
            cur.next = root_node
            root_node.next = right
            return left
        if left and not right:
            cur = left
            while cur.next:
                cur = cur.next
            cur.next = root_node
            return left
        if not left and right:
            root_node.next = right
            return root_node
Node1 = TreeNode(1)
Node2 = Node1.left = TreeNode(2)
Node3 = Node1.right = TreeNode(3)
Node4 = Node2.left = TreeNode(4)
Node4 = Node2.right = TreeNode(5)
solution = Solution()
print(solution.BTtoLinkedList(Node1).val)
# 题十二： 二叉树转双链表 （1<->2<->3）
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
   
    """
    @param root: root of a tree
    @return: head node of a doubly linked list
    """
    def treeToDoublyList(self, root):
        if root is None:
            return None
        
        head, tail = self.dfs(root)
        
        # 因为题目要求是环，所以还要首尾相连
        tail.right = head
        head.left = tail
        
        return head
        
    def dfs(self, root):
        if root is None:
            return None, None
            
        left_head, left_tail = self.dfs(root.left)
        right_head, right_tail = self.dfs(root.right)
        
        # left tail <-> root
        if left_tail:
            left_tail.right = root
            root.left = left_tail
        # root <-> right head
        if right_head:
            root.right = right_head
            right_head.left = root
        
        # 其实 root 肯定是有的，第三个 or 只是为了好看
        head = left_head or root or right_head
        tail = right_tail or root or left_tail
        
        return head, tail

# 题十三：二叉树拆成伪链表形式，只有右枝
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:
    """
    @param root: a TreeNode, the root of the binary tree
    @return: nothing
    """
    def flatten(self, root):
        self.flatten_and_return_last_node(root)
        return root
    # restructure and return last node in preorder
    def flatten_and_return_last_node(self, root):
        if root is None:
            return None
            
        left_last = self.flatten_and_return_last_node(root.left)  #最终返回的root肯定十不变的，变得十下面的枝和叶子，所以不需要返回root节点  # 注意做题前要明确返回的是什么形式
        right_last = self.flatten_and_return_last_node(root.right)
        
        # connect 
        if left_last is not None:  # 如果没有左枝只有右枝，不需要做任何事情
            left_last.right = root.right
            root.right = root.left
            root.left = None
        
        if right_last:
            return right_last
        if not right_last and left_last:
            return left_last
        if not left_last and not right_last:
            return root
        # return right_last or left_last or root # 上面3个if语句也可以写成这个形式，return中or运算，从左到右运算，遇到真值返回真值，如果没有真值返回False
