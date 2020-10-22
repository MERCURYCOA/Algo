# stack 操作时间复杂度  push  O(1), pop O(1), top O(1)  注意top是查看最顶的元素但是不pop出来
# 栈的3个考点：1 暂时保存信息  2 翻转栈  3 优化dfs，变成非递归

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
