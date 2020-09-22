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
