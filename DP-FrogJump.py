# x轴上有n个石头，从0，到n-1, 青蛙初始在0处，给定数组A[i],已知青蛙在i处最多向右跳A[i]下，问给定数组A，青蛙能否跳到n-1？
# 动态规划-存在型问题
#解释： 题设 - 在i处，最多向右跳A[i], 重点在 -- 最多
#      这就决定了 不能单纯找f[n-1]与f[n-2],f[n-3]的关系，也就是说跳到n-1之前的那块石头是不确定的，所以应该设定j， 
#      j在（1，n-1）之间，j是青蛙跳到n-1石头之前的那块石头，但是这样的石头不只一个，所以要 for j in range(1, n)
#      但是怎么确定f[j]是否为True呢？ 通过i，也就是i是跳到j的前一块石头，如果 f[i]是true,并且 A[i] >= j-i, 
#      也就是从i到j的距离小于A[i]，那么f[j]就是true。
#难点： 不是从f[n-1],f[n]这样循环，这里迭代步长不确定，通过枚举j来实现迭代。 
#关键：
#    创建bool数组，初始化f[0] =True, f[j] = False
#    判断条件：f[i] == True 和 A[i] >= j-i

class Solution:
    def is_jump(self, A):
        n = len(A)
        f = [bool]*n
        f[0] = True
        for j in range(1,n):
            f[j] = False
            for i in range(0,j):
                if f[i] == True and A[i] +i >= j:
                    f[j] = True
                    break
        return f[n-1]
   

if __name__ == '__main__':
    solution = Solution()
    print(solution.is_jump([3,2,1,0,4]))
