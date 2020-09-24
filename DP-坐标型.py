# |   |   |   |
# |   |   |   |
# |   |   |   |
# 动态规划 - 坐标型
# （也是计数型）
# 坐标型解法：对于输入数组A, 创建一个与其大小相同的F数组

# 题一：
# 机器人从网格左上角到右下角，有多少种unique的方式？一次只能向右或向下移动。

class Solution:
    def total_unique_paths(self, A):
        n = len(A)
        m = len(A[0])
        f = [[None] * (m+1) for _ in range(n+1)]  #创建空二维数组
        # 合规判断
        if n < 1 or m < 1:
           return -1
    
        for i in range(1, n+1):
            for j in range(1, m+1):
                if i == 1 or j == 1:     # initialization
                    f[i][j] = 1
                else:
                    f[i][j] = f[i-1][j] + f[i][j-1]
        return f[i][j]
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.total_unique_paths(3,3))
# 题二：
# 与题一类似，只是格子中有障碍， 令有障碍的格子表示为 A[i][j]=1，其他格子为0， 坐标（0，0）和（n,m）有障碍直接返回0，因为永远无法到达
# 关键：初始化：f数组全为0，分类讨论， 
#               1. 有障碍一票否决,如果A[i][j]是1， 则对应的f[i][j] = 0， 这个时候希望进入下个循环，而不希望执行下面的语句，所以continue
#               2. 初始化左上角的f[0][0],有两种可能，第一，A[0][0]有障碍，即A[0][0]=1, 则进入第一个if语句，f[0][0]初始化为0； 第二， A[0][0]没有障碍，则f[0][0]初始化为1
#               ** 第1，2个if语句是对f[i][j]进行初始化，一个是处理有障碍的格子，一个是处理左上角的格子，因为左上角是计算的起始点
#               3. 在题一中，第一行，第一列，以及内陆地区的格子，是分开处理的，因为第一行第一列只有一种到达方式，而内陆有两种到达方式。但是，在本题中，因为有障碍存在，所以边界格子和内陆格子都有可
#                  能有0，1种方式到达,所以，思路变成：从上边加还是从左边加，如果i>0,即不在第一行，则把上面的加过来f[i][j] = f[i][j] + f[i-1][j]， j>0,即不在第1列，则把左边的加过来
#                  f[i][j] = f[i][j] + f[i][j-1]， 注意：这里的i,j表示没有障碍的格子，因为如果有障碍，就在第一个语句执行的，到不来了下面。如果左边和上边都没有障碍，i>0和j>0两条语言都执行，
#                  就可以把f[i-1][j]和f[i][j-1]都加上。
#                  **这里第一行第一列也被包含了。假设：第一行第3列无障碍，语句f[i][j] = f[i][j] + f[i][j-1]会加上左边第二列的f[0][1]，正合题意。
class Solution:
    def total_unique_paths(self, A):
        n = len(A)
        m = len(A[0])
        f = [[0] * m for _ in range(n)]  #创建空二维数组
        # 合规判断
        if n < 1 or m < 1:
           return -1
    
        for i in range(0, n):
            for j in range(0, m):
                # 不能用if else, 应该用多个if, 因为这些情况是分类，不是互斥的
                # 可能同时满足i>0, j>0, 这个时候如果不想执行剩下的语句，用continue
                if A[i][j] == 1:
                    f[i][j] = 0
                    continue
                if i == 0 and j ==0:
                    f[i][j] = 1
                    continue
                if i>0:
                    f[i][j] = f[i][j] + f[i-1][j]
                if j >0:
                    f[i][j] = f[i][j] + f[i][j-1]
        return f[n-1][m-1]
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.total_unique_paths([[0,0,1,0],[0,1,0,0],[0,0,0,0]]))
    
# 题三：找到双向最大上升连续子序列，从左到右上升和从右到左上升
# 题目转换成：求上升，然后翻转数组，之后在求上升，最后比较那个大返回哪个

class Solution:
    def max_increase_subarray(self, A):
        n = len(A)
        f = [0]*n
        f[0] = 1
        max_subarray = 0
        for i in range(1,n):
            if A[i] > A[i-1]:
                f[i] = f[i-1] + 1
                max_subarray = max(max_subarray, f[i])
            else:
                f[i] = 1
        return max_subarray
    def bidirection(self, A):
        n = len(A)
        if n == 0:
            return 0
        r1 = self.max_increase_subarray(A)
        i = 0
        j = n-1
        while (i < j):
            temp = A[i]
            A[i] = A[j]
            A[j] = temp
            i += 1
            j -= 1
        r2 = self.max_increase_subarray(A)
        return max(r1,r2)
if __name__ == '__main__':
    solution = Solution()
    print(solution.bidirection([1,2,3,5,4,3,2,1,7,1]))
# 题四
# N*M 网格中有数字，求从左上到右下角路径上数字和最小是多少？
#方法一：
import sys

class Solution:
    def min_path_sum(self, A):
        n = len(A)
        m = len(A[0])
        if n == 0:
            return -1
        f = [[0]*m for _ in range(n)]
        for i in range(0, n):
            for j in range(0, m):
                if i ==0 and j == 0:
                    f[i][j] = A[i][j]
                    continue   # 不要忘记continue，你不希望再执行后面的语句， 这里不想continue，后面的分类就要单独处理第一行，第一列
                temp = sys.maxsize
                if j > 0:
                    temp = min(temp, f[i][j-1])
                if i > 0:
                    temp = min(temp, f[i-1][j])
                f[i][j] = temp + A[i][j]      # 这里在i>0和j>0内循环的是temp, 不是f[i][j],因为最小值只能是上面或左边其中的一个路径，通过temp找到最小的，然后加上当前A[i][j]
        return f[n-1][m-1]
if __name__ == '__main__':
    solution = Solution()
    print(solution.min_path_sum([[1,2]]))
# 方法二： 滚动数组

