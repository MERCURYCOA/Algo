# |   |   |   |
# |   |   |   |
# |   |   |   |
# 动态规划 - 坐标型 + 位运算动态规划
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
    """
    @param obstacleGrid: A list of lists of integers
    @return: An integer
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        #获取网格的长宽
        n,m = len(obstacleGrid),len(obstacleGrid[0])
        if n == 0 and m == 0:
            return 0
        dp=[[0] * m for _ in range(n)]
        if obstacleGrid[0][0] == 0:
            dp[0][0] = 1
        for i in range(0,n):
            for j in range(0,m):
                if i == 0 and j == 0:
                    continue
                #若遇到障碍物，则跳过
                if obstacleGrid[i][j] == 1:
                    continue
                #对于上边界，第一个障碍物或边界左边的所有边界点皆可到达
                if i == 0:
                    dp[i][j] = dp[i][j-1]
                    continue
                #对于左边界，第一个障碍物或边界前的所有边界点皆可到达
                if j == 0:
                    dp[i][j] = dp[i-1][j]
                    continue
                #到达当前点的路径数等于能到达此点上面的点和左边点的路径数之和
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[n-1][m-1]
    
# 第2次自己做的：有些麻烦，跟别人简便做法比：1， 简便做法利用了初始化f全为0，所以在边界上obstacleGrid[i][j]=1时，就直接continue, 
# 2 又因为边界上所有值只跟左边或上边有关，所以dp[i][j] = dp[i-1][j] 就可以解决边界遇到obstacal之后的所有f必须全为0的问题。         
class Solution:
    """
    @param obstacleGrid: A list of lists of integers
    @return: An integer
    """
    def uniquePathsWithObstacles(self, obstacleGrid):
        if not obstacleGrid or obstacleGrid[0] is None or obstacleGrid[0][0] == 1:
            return 0 
            
        n, m = len(obstacleGrid), len(obstacleGrid[0])
        f = [[0] * m for _ in range(n)]
        
        for j in range(m):
            if obstacleGrid[0][j] == 1:
                for t in range(j, m):
                    f[0][t] = 0
                break 
            elif obstacleGrid[0][j] == 0:
                f[0][j] = 1 
                
        for i in range(n):
            if obstacleGrid[i][0] == 1:
                for t in range(i, n):
                    f[t][0] = 0
                break   
            elif obstacleGrid[i][0] == 0:
                f[i][0] = 1
        for i in range(1, n):
            for j in range(1, m):
                if obstacleGrid[i][j] == 1:
                    f[i][j] = 0
                else:
                    f[i][j] = f[i-1][j] + f[i][j-1]
        return f[n-1][m-1]


# 题三：找到双向最大上升连续子序列，从左到右上升和从右到左上升
# 题目转换成：求上升，然后翻转数组，之后在求上升，最后比较那个大返回哪个
# 解法一：空间O(N)
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
        r2 = self.max_increase_subarray(A[::-1])
        return max(r1,r2)
if __name__ == '__main__':
    solution = Solution()
    print(solution.bidirection([1,2,3,5,4,3,2,1,7,1]))
#解法二： 空间O(1)  局部变量和全局变量
class Solution:
    def max_increase_subarray(self, A):
        n = len(A)
        #f = [0]*n
        #f[0] = 1
        max_subarray = 1
        local = 1
        for i in range(1, n):
            if A[i] > A[i-1]:
                local+=1
                
            else:
                max_subarray = max(max_subarray, local)
                local = 1
        return max_subarray
    def bidirection(self, A):
        n = len(A)
        if n == 0:
            return 0
        r1 = self.max_increase_subarray(A)
    
        r2 = self.max_increase_subarray(A[::-1])
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
    
# 第2次做：
class Solution:
    """
    @param grid: a list of lists of integers
    @return: An integer, minimizes the sum of all numbers along its path
    """
    def minPathSum(self, grid):
        if not grid or not grid[0]:
            return -1
        n, m = len(grid), len(grid[0])
        f = [[0] * m] * n
        f[0][0] = grid[0][0]
        for i in range(n):
            for j in range(m):
                if i == 0:
                    f[i][j] = f[i][j-1] + grid[i][j]
                    continue
                if j == 0:
                    f[i][j] = f[i-1][j] + grid[i][j]
                    continue
                f[i][j] = min(f[i-1][j], f[i][j-1]) + grid[i][j] 
                
        return f[n-1][m-1]

# 方法二： 滚动数组
#关键：开2行数组，建2个指针，old, new, 每行一换，也就是在i循环这一层，需要让old new交替，也就是old = new, new = 1 [其实就是0 -> 1, 1 -> 0]
#     然后把后面每个i改成new, i-1改成old， 返回f[new][m-1]
import sys

class Solution:
    def min_path_sum(self, A):
        n = len(A)
        m = len(A[0])
        if n == 0:
            return -1
        f = [[0]*m for _ in range(0,2)]
        old = new = 0
        for i in range(0, n):
            old = new
            new = 1 - new
            for j in range(0, m):
                if i == 0 and j == 0:
                    f[new][j] = A[i][j]
                    continue
                temp = sys.maxsize
                if j > 0:
                    temp = min(temp, f[new][j-1])
                if i > 0:
                    temp = min(temp, f[old][j])
                f[new][j] = temp + A[i][j]
        return f[new][m-1]
if __name__ == '__main__':
    solution = Solution()
    print(solution.min_path_sum([[1,2]]))
# 第2次做：滚动数组
class Solution:
    """
    @param grid: a list of lists of integers
    @return: An integer, minimizes the sum of all numbers along its path
    """
    def minPathSum(self, grid):
        if not grid or not grid[0]:
            return -1
        n, m = len(grid), len(grid[0])
        f = [[0] * m] * 2
        for i in range(n):
            for j in range(m):
                if i == 0 and j == 0:
                    f[i][j] = grid[i][j]
                    continue
                if i == 0:
                    f[i][j] = f[i][j-1] + grid[i][j]
                    continue
                if j == 0:
                    f[i&1][j] = f[(i-1)&1][j] + grid[i][j]
                    continue
                f[i&1][j] = min(f[(i-1)&1][j], f[i&1][j-1]) + grid[i][j] 
                
        return f[(n-1)&1][m-1]
# 题四：炸弹袭击
# 方法一： 预处理出每个点向四个方向能炸到的人数，然后枚举所有点，取最大值即可
class Solution:
    # @param {character[][]} grid Given a 2D grid, each cell is either 'W', 'E' or '0'
    # @return {int} an integer, the maximum enemies you can kill using one bomb
    def maxKilledEnemies(self, grid):
        # Write your code here
        m, n = len(grid), 0
        if m:
            n = len(grid[0])
        result, rows = 0, 0
        cols = [0 for i in xrange(n)]

        for i in xrange(m):
            for j in xrange(n):
                if j == 0 or grid[i][j-1] == 'W':
                    rows = 0
                    for k in xrange(j, n):
                        if grid[i][k] == 'W':
                            break
                        if grid[i][k] == 'E':
                            rows += 1

                if i == 0 or grid[i-1][j] == 'W':
                    cols[j] = 0
                    for k in xrange(i, m):
                        if grid[k][j] == 'W':
                            break
                        if grid[k][j] == 'E':
                            cols[j] += 1

                if grid[i][j] == '0' and rows + cols[j] > result:
                    result = rows + cols[j]
                print(rows, cols, result)
        return result 

solution = Solution()
print(solution.maxKilledEnemies(["0E00","E0WE","0E00"]))

# 方法二：四个dp数组
# 这道题翻来覆去就是一个概念，计算顺序。 四个方向的预处理计算分别是4个DP。 最后过一遍，求四个方向的和。

# 比如，从左到右。

# 固定每一行（相当于01背包为毛要首先遍历item？）
# 对固定的行，从左到右计算每一列。
# 如果是 W 就清零，
# 如果是E就累加dp[i] = dp[i-1] + 1
# 如果是0就直接转移上一个状态 dp[i] = dp[i-1]

class Solution:

    """

    @param grid: Given a 2D grid, each cell is either 'W', 'E' or '0'

    @return: an integer, the maximum enemies you can kill using one bomb

    """

    def maxKilledEnemies(self, grid):

        if not grid or len(grid) == 0 or len(grid[0]) == 0 :
            return 0
        row, col = len(grid), len(grid[0])
        # init
        up = [[0] * col for _ in range(row)]
        down = [[0] * col for _ in range(row)]
        left = [[0] * col for _ in range(row)]
        right = [[0] * col for _ in range(row)]
        # up
        for i in range(row) : 
            for j in range(col) :
                if grid[i][j] != 'W' :
                    if grid[i][j] == 'E' :
                        up[i][j] = 1
                    if i > 0 :
                        up[i][j] += up[i - 1][j]
        # down
        for i in range(row - 1, -1, -1) :
            for j in range(col) :
                if grid[i][j] != 'W' :
                    if grid[i][j] == 'E' :
                        down[i][j] = 1
                    if i + 1 < row :
                        down[i][j] += down[i + 1][j]
        # right
        for i in range(row) : 
            for j in range(col - 1, -1, -1) :
                if grid[i][j] != 'W' :
                    if grid[i][j] == 'E' :
                        right[i][j] = 1
                    if j + 1 < col :
                        right[i][j] += right[i][j + 1]
                        
        # left
        for i in range(row) : 
            for j in range(col) :
                if grid[i][j] != 'W' :
                    if grid[i][j] == 'E' :
                        left[i][j] = 1
                    if j > 0 :
                        left[i][j] += left[i][j - 1]
        
        # sum 
        res = 0
        for i in range(row):
            for j in range(col) :
                if grid[i][j] == '0' :
                    res = max(res, up[i][j] + down[i][j] + left[i][j] + right[i][j])
        
        
        return res
# 位运算
# 1 
class Solution:
    """
    @param num: a non negative integer number
    @return: an array represent the number of 1's in their binary
    """
    def countBits(self, num):
        dp = [0]
        
        for i in range(1, num+1):
            dp.append(dp[i>>1] + i%2)  # 这里不可以用 i&1代替1或0. 因为  1 + 1&1 = 0而不是1， 因为加号优先于&， 就变成2&1
            print(dp[i>>1], i%2)
        return dp 
