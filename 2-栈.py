# stack 操作时间复杂度  push  O(1), pop O(1), top O(1)
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
