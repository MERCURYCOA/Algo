# 迭代器与递归的区别是，每次next()都能弹出下一个元素，可以单独拿这个元素出来作什么，或者就停在当前元素，不再向下迭代，递归就不行，递归一旦开始执行，没办法停下
# 这个是怎么实现的呢？实现next（）/ haseNext()函数时，每次找到下一个元素时return 下一个元素或者return True 
# 解题步骤：
1. List 转 Stack
2. 主函数逻辑放在HasNext里面 
3. Next只做一次pop处理

# 题一：摊平嵌套列表
# 实现迭代器
"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation

class NestedInteger(object):
    def isInteger(self):
        # @return {boolean} True if this NestedInteger holds a single integer,
        # rather than a nested list.

    def getInteger(self):
        # @return {int} the single integer that this NestedInteger holds,
        # if it holds a single integer
        # Return None if this NestedInteger holds a nested list

    def getList(self):
        # @return {NestedInteger[]} the nested list that this NestedInteger holds,
        # if it holds a nested list
        # Return None if this NestedInteger holds a single integer
"""

class NestedIterator(object):

    def __init__(self, nestedList):
        self.next_elm = None
        self.stack = nestedList[::-1]
        
    # @return {int} the next element in the iteration
    def next(self):
        # Write your code here
        #if self.hasNext:
        return self.next_elm
        
    # @return {boolean} true if the iteration has more element or false
    def hasNext(self):
        while self.stack:
            element = self.stack.pop()
            if element.isInteger():
                self.next_elm = element.getInteger()
                return True                             # 这里找到下一个元素，return, 就停下了， 下次再执行时，从stack.pop出top元素即可
            else:
                cur_list = element.getList()
                while cur_list:
                    self.stack.append(cur_list.pop())
        return False

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

# 题一.2 摊平扁平化 
# 只需返回列表

class Solution(object):

    # @param nestedList a list, each element in the list 
    # can be a list or integer, for example [1,2,[1,2]]
    # @return {int[]} a list of integer
    def flatten(self, nestedList):
        if not nestedList:
            return []
        res = []
        stack = nestedList[::-1]
      
        while stack:
            element = stack.pop()
            if type(element) == int:
                res.append(element)   # 找到下一个元素不需返回，只要加到res里即可
            else:
                while element:
                    stack.append(element.pop())
                    
        return res
        
# 题二： 二叉查找树迭代器
"""
Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

Example of iterate a tree:
iterator = BSTIterator(root)
while iterator.hasNext():
    node = iterator.next()
    do something for node 
"""


class BSTIterator:
    """
    @param: root: The root of binary tree.
    """
    def __init__(self, root):
        self.stack = []
        while root != None:
            self.stack.append(root)
            root = root.left

    """
    @return: True if there has next node, or false
    """
    def hasNext(self):
        return len(self.stack) > 0

    """
    @return: return next node
    """
    def next(self):
        node = self.stack[-1]
        if node.right is not None:
            n = node.right
            while n != None:
                self.stack.append(n)
                n = n.left
        else:
            n = self.stack.pop()
            while self.stack and self.stack[-1].right == n:
                n = self.stack.pop()
        
        return node
