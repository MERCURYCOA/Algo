# stack 操作时间复杂度  push  O(1), pop O(1), top O(1)  注意top是查看最顶的元素但是不pop出来
# 栈的3个考点：1 暂时保存信息  2 翻转栈  3 优化dfs，变成非递归
# 单调栈：求每个元素左边/右边第一个比它小/大的元素时，用单调栈

#题一：实现最小栈 min stack
class MinStack:
    
    def __init__(self):
        self.stack = []
        self.min_stack = []

    """
    @param: number: An integer
    @return: nothing
    """
    def push(self, number):
        self.stack.append(number)
        if not self.min_stack or self.min_stack[-1]>=number: #必须带上=， 原因：假设stack有两个相同最小值 7，7， 其中一个最小值pop出去之后，stack里面还有一个最小值，所以min_stack也应该压入两个7才对，也就是说当前number跟min_stack最小值相等时，也应该被压入min_stack里
            self.min_stack.append(number)

    """
    @return: An integer
    """
    def pop(self):
        number = self.stack.pop()
        if self.min_stack and self.min_stack[-1] == number:
            self.min_stack.pop()
        return number
    """
    @return: An integer
    """
    def min(self):
        if self.min_stack:
            return self.min_stack[-1]  # 这里不能用pop(),因为只是查看不需要pop出来，如果pop出来，最小值就变了
# 题二：用2个栈实现queue  - 翻转栈
class MyQueue:
    
    def __init__(self):
        self.stack = []
        self.stack_ = []

    """
    @param: element: An integer
    @return: nothing
    """
    def push(self, element):
        self.stack.append(element)

    """
    @return: An integer
    """
    def pop(self):
        while self.stack:
            self.stack_.append(self.stack.pop())
        if self.stack_:
            element = self.stack_.pop()
        while self.stack_:                  # 记得还要从stack_放回到stack里
            self.stack.append(self.stack_.pop())
        return element

    """
    @return: An integer
    """
    def top(self):
        while self.stack:
            self.stack_.append(self.stack.pop())
        if self.stack_:
            element = self.stack_[-1]
        while self.stack_:
            self.stack.append(self.stack_.pop())
        return element
    
 # 题三： expression expand 字符串解码

class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def expressionExpand(self, s):
        if not s:
            return ''
        stack = []
        
        for c in s:
            if c == ']':
                strs = []
                while stack and stack[-1] != '[':
                    strs.append(stack.pop())
                stack.pop() #弹出'['
                num = 0
                base = 1
                while stack and stack[-1].isdigit():  # 出栈的数字顺序是反的，需要正过来
                    num += int(stack.pop()) * base
                    base *= 10
                stack.append(''.join(reversed(strs))*num)  # strs是字符串数组，需要用''.join变成string
           
            else:
                stack.append(c)
        return ''.join(stack)        
# 注意：这种解法不可以让'['直接忽略，还是需要压入栈内，然后到'['时pop出来。如果直接elif c =='[': continue, 3[2[ab]]这种情况会出现错误
# 下面这种解法可以不让'['入栈，但是与上面的解法有2点不同 1: 数字在存入栈之前就转换好了,记为number 2：elif c=='[':stack.append(number), 遇到[时，就把相邻的number压入栈内
class Solution:
    # @param {string} s  an expression includes numbers, letters and brackets
    # @return {string} a string
    def expressionExpand(self, s):
        stack = []
        number = 0
        for char in s:
            if char.isdigit():
                number = number * 10 + ord(char) - ord('0')
            elif char == '[':
                stack.append(number)
                number = 0
            elif char == ']':
                strs = []
                while len(stack):
                    top = stack.pop()
                    if type(top) == int:
                        stack.append(''.join(reversed(strs)) * top)
                        break
                    strs.append(top)
            else:
                stack.append(char)
        return ''.join(stack)
# 题四：直方图最大矩形覆盖
# 当前高度out, 需要弹出，说明下一个进栈的数i对应的高度小于out高度，这样，stack[-1]是左边第一个小于out的数， 下一个进栈的i是右边第一个小于out的数，这两个数中间是out高度能组成的最大矩形
# 我的解法 - 略显丑陋
import sys
class Solution:
    """
    @param height: A list of integer
    @return: The area of largest rectangle in the histogram
    """
    def largestRectangleArea(self, height):
        if not height:
            return 0
            
        stack = []
        max_ = -sys.maxsize-1
        for i in range(len(height)):
            while stack and height[i] < height[stack[-1]]:
                out = stack.pop()
                if stack:   # 注意stack剩最后一个元素时，弹出后就无法查看stack[-1]
                    max_ = max(max_, height[out]*(i-stack[-1]-1)) # 当前弹出的高度的矩形左边可到达stack[-1] (左边第一个比当前高度小的数),右侧可到达i（右边第一个比当前高度小的数），细节处理真正涵盖的长度i-stack[-1]-1
                else:
                    max_ = max(max_, height[out]*(i)) # 最后弹出的元素一定是小于该元素前面的所有元素，因为比它小的在它入栈之前就被弹出了，
            stack.append(i)
            
        
        remost = len(height)-1
        while stack:
            out = stack.pop()
            if stack:
                max_ = max(max_, height[out]*(remost-stack[-1]))  # 剩到stack里，这个stack又是单调上升栈，说明当前高度的矩形左侧可以到达stack[-1],右侧可到达len(height)-1
            else:
                max_ = max(max_, height[out]*len(height)) # 特殊细节处理，真正意义剩到最后的元素，一定是整个直方图最低的，可以贯穿整个直方图
        if max_ == -sys.maxsize-1:
            max_= 0
        return max_
   # 下面是别人的解法 - 巧妙
# 在heights后面补0,这样for循环完stack里就没有元素了，就不需要再来while stack
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        res = 0
        heights = heights + [0]
        for right in range(len(heights)):
            while stack and heights[right] < heights[stack[-1]]: # 没有等号，因为下一个进来的高度跟当前高度相等时，同一个矩形可以延伸，还没找到右边第一个比当前高度小的数
                cur = stack.pop()
                if stack:
                    left = stack[-1]
                else:
                    left = -1
                res = max(res, (right - left - 1) * heights[cur])
            stack.append(right)
        return res
# 题五： 最大树 
# 方法一： 维护单调递减栈
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
"""

class Solution:
    """
    @param A: Given an integer array with no duplicates.
    @return: The root of max tree.
    """
    def maxTree(self, A):
        # write your code here
        if not A:
            return None
        
        stack = []
        for index, num in enumerate(A + [sys.maxsize]): # dummy node的作用
            cur = TreeNode(num)
            # 单调递减栈
            while stack and stack[-1].val < cur.val:  
                out = stack.pop()

                if stack and stack[-1].val < cur.val:  # 左边第一个比out大的数 stack[-1].val， 右边第一个比out大的数 cur.val， out一定是较小的那个数的儿子
                    stack[-1].right = out              # 左边 < 右边, 让out成为左边的右儿子 （因为stack[-1]还存在说明这个位置的元素比之前的大，左儿子已经占了）
                else:                                  # 左边 > 右边， 让out成为右边数cur的左儿子，因为右儿子位置需要留给继续向右遍历时小于cur的数/ 还有一种情况是out弹出后，栈内没有元素了（说明out左边没有比它大的元素，只有右边）
                    cur.left = out 
                    
            stack.append(cur)
            
        return stack[-1].left
