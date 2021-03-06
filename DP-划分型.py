#f[N]取决于f[N-1]和f[N-2]
# 题一： 解码方法  到i的解码方法，不仅跟i-1有关，还跟i-2有关，所以构造从2开始，去分别寻找i-2, i-1
class Solution:
    """
    @param s: a string,  encoded message
    @return: an integer, the number of ways decoding
    """
    def numDecodings(self, s):

        #如果s以 '0' 开头, 则直接返回0

        if s == "" or s[0] == '0':

            return 0

        #dp[i]表示字符串的前i位解码有多少种解码方式

        #数组初值均为0，dp[0] = dp[1] = 1

        n = len(s)
        dp = [0]*(n+1)
        dp[0] = 1
        for i in range(1, n + 1):   # '192011'  dp = [1, 1, 2, 0]
            #若s[i - 1]表示的数是1到9

            if int(s[i - 1 : i]) != 0:

                dp[i] += dp[i - 1]

            #若s[i - 2]和s[i - 1]表示的数是10到26

            if i > 1 and 10 <= int(s[i - 2 : i]) <= 26:  # i > 1重要

                dp[i] += dp[i - 2]

        return dp[n]


# f需要n+1个格子，初始化f[-1] =0, f里的i跟A里的i可以对齐，最后返回f[n-1],如果初始化f[0] = 1, f的i和A的i不对齐，即f[i]储存的是A[i-1]之前的总计数，最后返回f[n]
class Solution:
    def decode_path(self, S):
        A = [int(x) for x in list(S)]
        n = len(A)
        f = [0]*(n+1)
        
        for i in range(1,n+1):
            f[0] = 1
            if A[i-1] >= 1 and A[i-1] <= 9:
                f[i] += f[i-1]
            if i > 1:
                if A[i-2]==1 or A[i-2] ==2 and A[i-1]<=6:
                    f[i] += f[i-2]
        return f[n]

if __name__ == '__main__':
    solution = Solution()
    print(solution.decode_path("12"))
# 题二：完美平方

class Solution:
    """
    @param n: a positive integer
    @return: An integer
    """
    def numSquares(self, n):
        dp = [n for _ in range(n+1)]
        for i in range(n+1):
            if i**2 <= n:
                dp[i**2] = 1
            j = 1
            while j**2 < i:
                dp[i] = min(dp[i], dp[i-j**2] + 1)
                j +=1
        return dp[n]
# 题三：分割回文串
class Solution:
    """
    @param s: A string
    @return: An integer
    """
    def minCut(self, s):
        if not s or len(s) == 0:
            return 0
        
        m = len(s)
        res = [0 for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            res[i] = sys.maxsize
            for j in range(i):
                #if (j, i - 1) not in palin_index:
                if not self.isPalin(s, j, i-1):
                    continue
                res[i] = min(res[i], res[j] + 1)
        
        return res[m] - 1
    def isPalin(self, s, start, end):
        if s[start : end+1] == s[start : end+1][::-1]:
            return True 
        return False
