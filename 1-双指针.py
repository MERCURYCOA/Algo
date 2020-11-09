# 模版一：窗口类指针， j的循环移动，要把条件写到while后面
# 本质是对两层for循环的改进
# 内层不需要回退
for i in range(n):
  while j < n and i < j and j前进的条件：
    j += 1
    更新j状态
   
  if j停下的条件：
    return/ break 
  更新i的状态

# ============================================
# 一： 同向双指针
# 1 滑动窗口内数的和
# 滑动窗口的节约时间的点是只需要计算1次第一个窗口内的和， 窗口向后移动时，只需减去最前面的元素然后假设后面一个新元素，如果每个窗口都计算窗头到窗尾的和，就失去窗口的意义了。
# 步骤： 1， 2个指针一开始就r放到窗口头尾，2， 计算第一个窗口的和  3 向后滑动窗口 加减元素
class Solution:
    """
    @param nums: a list of integers.
    @param k: length of window.
    @return: the sum of the element inside the window at each moving.
    """
    def winSum(self, nums, k):
        if not nums:
            return []
        n = len(nums)
        res = []
       
        l, r = 0, k-1
        sum_ = sum(nums[l : k])
        res.append(sum_)
        while r < n-1:  # 如果r < n, 会出界
            sum_ -= nums[l]
            l+= 1 
            r += 1
            sum_ += nums[r]
            res.append(sum_)
            
        return res 
# 2  移动零 
# 先移动i，i符合条件停下后，让j=i，然后让j向后，直到符合条件停下
class Solution:
    """
    @param nums: an integer array
    @return: nothing
    """
    def moveZeroes(self, nums):
        if not nums:
            return []
        n = len(nums)
        i, j = 0, 0
        
        while i < n and j < n:
            while i < n and nums[i] != 0:
                i += 1 
            if i < n and nums[i] == 0:
                j = i
                while j < n and nums[j] == 0:
                    j += 1 
                if j < n and nums[j] != 0:
                    nums[i], nums[j] = nums[j], nums[i]
                    i += 1 
                    j += 1 
        return nums
# 二：对向双指针
#1 有效回文串
class Solution:
    """
    @param s: A string
    @return: Whether the string is a valid palindrome
    """
    def isPalindrome(self, s):
        if not s:
            return True 
        s = s.lower()
        print(s)
        n = len(s)
        i, j = 0, n-1 
        while i < j and i < n and j >= 0:
            while  i < j and i < n and s[i].isalpha() == False and s[i].isdigit() == False:  # 用and不同or, 因为必须同时符合不是字母不是数字才能略过
                i += 1 
            
            while i < j and j >= 0 and s[j].isalpha() == False and s[j].isdigit()== False:
                j -= 1 
            
            if i < j and s[i] != s[j]: # 这里不可以用if else , 只能用单if 判断是否不相等， 返回false, 错误写法见下面
                return False 
            i += 1 
            j -= 1 
        return True
# 错误写法：        
if i < j and s[i] == s[j]:
    i += 1 
    j -= 1    
else:
    return False  # ！！！！！！！！千万慎用else: False, 这样太冒险兰，你怎么直到else里面包含什么， 一定要明确的条件下return False
                  # 这里问题出在 当i = j时， a b a， 最后i, j都移动到b， 此时i = j 就包含在else里，返回false.这是不对的， 当i = j  时，应该i, j继续对向移动才对。
# 2. 旋转字符
# 3步翻转 # offset说的是从后向前  # 要求额外空间O(1)
class Solution:
    """
    @param str: An array of char
    @param offset: An integer
    @return: nothing
    """
    def rotateString(self, s, offset):
        if not s:
            return ""
        n = len(s)
        if offset <= 0:
            return s
        offset = offset % n
        s = self.reverse(s, 0, n - offset - 1)
        s = self.reverse(s, n - offset, n-1)
        s = self.reverse(s, 0, n-1)
        
        return s
    def reverse(self, s, start, end):
        i, j = start, end 
        while i < j:
            s[i], s[j] = s[j], s[i]
            i += 1 
            j -= 1 
        return s
 # 旋转字符II:
# 没有要求额外空间O(1)
class Solution:
    """
    @param str: An array of char
    @param left: a left offset
    @param right: a right offset
    @return: return a rotate string
    """
    def RotateString2(self, str, left, right):
        # write your code here
        str_len = len(str)
        offset = left - right		#计算总偏移量
        flag = 1 if(offset >= 0) else -1
        offset = abs(offset) % str_len
        if flag >= 0:
            A = str[:offset]		#截取offset右边部分  # 因为可以有额外空间，所以可以截取
            B = str[offset:]		#截取offset左边部分
        else:
            A = str[:str_len - offset]		#截取右边算起offset位置处左边部分
            B = str[str_len - offset:]		#截取其右边部分
        results = B + A
        return results


# Two Sum 系列
# 1. Two Sum
# 方法一：two pointers 
# 注意一般two pointers需要排序数组，如果需要返回坐标组，不能直接排序，这样会打乱下标，应该通过二元组储存原下标，之后再排序。
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        if not numbers:
            return [-1, -1]
        A = [(number, index) for index, number in enumerate(numbers)]
        A = sorted(A)
        i, j = 0, len(A) - 1 
        while i < j:
            if i < j and A[i][0] + A[j][0] == target:
                return  sorted([A[i][1], A[j][1]])
            elif i < j and A[i][0] + A[j][0] > target:
                j -= 1 
            elif i < j and A[i][0] + A[j][0] < target:
                i += 1 
        return [-1, -1]
# 方法二：dict
class Solution:
    """
    @param numbers: An array of Integer
    @param target: target = numbers[index1] + numbers[index2]
    @return: [index1 + 1, index2 + 1] (index1 < index2)
    """
    def twoSum(self, numbers, target):
        if not numbers:
            return [-1, -1]
        dict = {}
        for index, num in enumerate(numbers):
            if target - num in dict:
                return [dict[target-num], index]   # 注意第一个位置是dict[target-num]， dict[num]还没放进去
            if num not in dict:
                dict[num] = index 
        return [-1, -1]

# 2. 两数之和 - 不同组成  
# 关键略过重复的元素， 但又不能用set()去重 # 1, 2, 3, 4, 7, 11, 11  target = 22 , 用set 会把第2个11去掉
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSum6(self, nums, target):
        count = 0
        if not nums:
            return count
        nums = sorted(nums)
        
        # 1, 1, 2, 45, 46, 46
        i, j = 0, len(nums)-1
        while i < j:
            if i > 0 and nums[i] == nums[i-1]:
                i += 1 
            else:
                if i < j and nums[i] + nums[j] > target:
                    j -= 1 
                elif i < j and nums[i] + nums[j] < target:
                    i += 1 
                elif i < j and nums[i] + nums[j] == target:
                    count += 1 
                    i += 1 
                    j -= 1
        return count
#2. 构建data structure ,2个方法，add, find,可以加进去数，也可以返回是否存在一对数的和为value
# 用dictionary最快
class TwoSum:
    """
    @param number: An integer
    @return: nothing
    """
    def __init__(self):     # 构建类， init一个map
        self.map = {}
    
    def add(self, number):
        # write your code here
        if number in self.map:
            self.map[number] += 1 
            
        else:
            self.map[number] = 1 
            
            
    """
    @param value: An integer
    @return: Find if there exists any pair of numbers which sum is equal to the value.
    """
    def find(self, value):
        # write your code here
        for num in self.map:
            if value-num in self.map and (value-num != num or self.map[num]>1):
                return True
        return False   # 注意False的位置，与for并行。在for循环出来之后还没return true的一定return false。 
######
# 对于3 sum ,什么时候用2 sum函数什么时候用3指针（1定2动） -- 3 sum target 用sum, 可以调用2 sum ,传入target参数。 3 sum closest用3指针，因为你没办法用target做参数调用2 sum， 强行用会非常复杂， 本来closest问题就要遍历比较的，所以要用3指针。
      # 4. 3Sum  a + b + c = 0, 找到所有的[a, b, c] -> 不能重复
# 转化为 b + c = -a
# 第一次做
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        # write your code here
        if not numbers or len(numbers) == 0:
            return []
        
        n = len(numbers)
        numbers = sorted(numbers)
        res = []
        
        for i in range(n-2):
            if i>0 and numbers[i] == numbers[i-1]:
                continue
            target = -numbers[i]
            self.twoSum(target, numbers, i+1, res)  # 因为twoSum函数中已经将答案append到res, 这里不要让twoSum等于什么，直接传入调用twoSum,传入res
        return res
        
    def twoSum(self, target, numbers, start_index,res):
        left, right = start_index, len(numbers)-1
        while left < right:                                     # 3种情况，第一种情况内又考虑重复，一定要思路清晰
            if numbers[left] + numbers[right] == target:
                res.append([-target, numbers[left], numbers[right]])  # 加入的是具体的数，不是index
                left += 1
                right -= 1 
            
                while left < right and numbers[left] == numbers[left-1]: #因为res不能重复，所以这里只需判断numbers[left] + numbers[right] == target 情况下后面的left+,right-1是否是否与left,right位置的数相等即可，其他两种情况不需要做这个判断，反正都会略过
                    left += 1     
                while left < right and numbers[right] == numbers[right+1]:
                    right -= 1 
            elif numbers[left] + numbers[right] > target:
                right -= 1 
                
            else:
                left += 1 
# 第二次做
# two sum函数使用index的原因
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, numbers):
        if not numbers:
            return []
        self.res = []
        numbers = sorted(numbers)
        # -4, -1, -1, 0, 1, 2
        # -1 : 1 
        # -4, -1, 0, 1, 2  //   target 1
        for i in range(len(numbers)):
            if i > 0 and numbers[i] == numbers[i-1]:
                continue
            self.twoSum(i, numbers, 0 - numbers[i])
        return self.res
    
    def twoSum(self, index, numbers, target): 
        i, j = index+1, len(numbers)-1   # 这里必须从index下一个元素开始，不能i, j = 0, len(numbers)-1， 因为res里不能有重复的。如果每次给出一个target，都从头开始找twoSum = -target, 就会重复。例如： [-1,  0, 1]的结果就成里[[[-1,  0, 1],[[-1,  0, 1], [[-1,  0, 1]]]]]
        while i < j:
            if i > index+1 and numbers[i] == numbers[i-1]:
                i += 1 
            else:
                if i < j and numbers[i] + numbers[j] > target:
                    j -= 1 
                elif i < j and numbers[i] + numbers[j] < target:
                    i += 1 
                elif i < j and numbers[i] + numbers[j] == target:
                    self.res.append(sorted([0-target, numbers[i], numbers[j]]))
                    i += 1 
                    j -= 1 
# 5：能组成三角形的个数
# 第2次做不出的原因是：S[left] + S[right] <= S[i]时不知道该怎么办。让left向右移动即可， 不用担心left前面的再跟end后面的数，这些在ans += right - left已经算进去了， 当前left已经跟所有可能值都算到ans里了，过去的不用再考虑了。
# 注意： 算到ans里的是left 到right中间的元素，不是right到i之间的。绕不过这个弯，就没办法知道left， right怎么移动。想法应该是left 和 right之间的数都比left大，所以如果S[left] + S[right] > S[i]，那么left,right之间的所有数都可以，而不是让right跟right与i之间的数比较。
class Solution:
    """
    @param S: A list of integers
    @return: An integer
    """
    def triangleCount(self, S):
        # write your code here
        S.sort()
        
        ans = 0
        for i in range(len(S)-1, 1,-1):
            left, right = 0, i - 1
            while left < right:
                if S[left] + S[right] > S[i]:
                    ans += right - left  # 如果 S[left], S[right], S[i]能组成三角形，那么S[left]和S[right]中间的数都能和S[right]，S[i]组成，因为他们都大于S[left]
                    right -= 1
                else:
                    left += 1  # 移动最小的边，因为条件要求和大于S[i], 如果要求小于某值，就移动最大的值
        return ans            
# 6: 跟5同思路  two sum less than or equal to target  // two sum greater than target
def twoSumGreater(self, nums, target):
        nums = sorted(nums)
        count = 0
        i, j = 0, len(nums)-1 
        while i < j:
            if nums[i] + nums[j] <= target:
                i += 1
            else:
                count += j - i 
                j -= 1
        return count
def twoSumLess(self, nums, target):
        nums = sorted(nums)
        count = 0
        i, j = 0, len(nums)-1 
        while i < j:
            if nums[i] + nums[j] <= target:
                count += j - i 
                i += 1
            else:
                j -= 1
        return count
# 7 two sum closest
import sys
class Solution:
    """
    @param nums: an array of integer
    @param target: An integer
    @return: An integer
    """
    def twoSumClosest(self, nums, target):
        nums = sorted(nums)
        i, j = 0, len(nums)-1 
        min_diff = sys.maxsize
        while i < j:
          if nums[i] + nums[j] < target:
                diff = abs(nums[i] + nums[j] - target)
                min_diff = min(diff, target - nums[i] - nums[j])
                i += 1
            else:
                diff = abs(nums[i] + nums[j] - target)
                min_diff = min(diff, nums[i] + nums[j] - target)
                j -= 1

        return min_diff
      
 # 错误做法：！！！！！！！严禁在i,j移动之后对nums[i], nums[j]做判断或操作，一定是先操作，判断， 最后移动i, j. 先移动i, j是什么鬼！！！！！！！！
            if nums[i] + nums[j] > target:
                j -= 1 
            elif nums[i] + nums[j] < target:
                i += 1 
            elif nums[i] + nums[j] == target:
                return 0 
            if i < j:
                diff = abs(nums[i] + nums[j] - target)
                min_diff = min(min_diff, diff)
        return min_diff
# 8 three sum closest
# 别特么想那么复杂，复杂问题简单化， 你tm老是简单问题复杂化是怎么回事？？？？？？？？
# 固定1个，移动2个
import sys
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @param target: An integer
    @return: return the sum of the three integers, the sum closest target.
    """
    def threeSumClosest(self, numbers, target):
        if not numbers:
            return 0
        numbers = sorted(numbers)
        res = sys.maxsize
        for i in range(len(numbers) - 2):
            left, right = i + 1, len(numbers)-1   # left从i开始
            while left < right:
                sum_ = numbers[i] + numbers[left] + numbers[right]
                if abs(sum_ - target) < abs(res - target):
                    res = sum_     # 记住返回的是sum 不是差值
                if sum_ < target:
                    left += 1 
                elif sum_ > target:
                    right -= 1 
                elif sum_ == target:
                    return sum_ 
        return res
      
 # 9： 4 sum
# 固定2个，移动2个
# 注意4个数都有排除重复， i，j， left， right都要排除重复
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, numbers, target):
        # write your code here
        if len(numbers) <4:
            return []
        numbers.sort()
        res = []
        for i in range(len(numbers)-3):
            if i and numbers[i] == numbers[i-1]:  #i去重
                continue
            for j in range(i+1,len(numbers)-2): # j起始点是i+1, 不是0
                if j != i+1 and numbers[j] == numbers[j-1]:  # j 去重, j可以跟i相等, 但j不可以等于前一个j
                    continue
                left = j + 1 
                right = len(numbers)-1
                while left < right and left < len(numbers)-1:
                    if numbers[i] + numbers[j] + numbers[left] + numbers[right] == target:
                        res.append([numbers[i], numbers[j],numbers[left], numbers[right]])
                        right -= 1 
                        left += 1 
                        while left < right and numbers[left] == numbers[left-1]:           #left去重
                            left += 1 
                        while left < right and numbers[right] == numbers[right+1]:          # right去重
                            right -= 1 
                    elif numbers[i] + numbers[j] + numbers[left] + numbers[right] > target:
                        right -= 1 
                    else:
                        left += 1 
        return res            
# 第2次做：用two sum
# 不难， 但是细节非常重要  1， 每层都有去重， 内层i, j对向，记得j也要去重
# 加到res里的是start,end还是start+1， end-1上的位置，要清楚
class Solution:
    """
    @param numbers: Give an array
    @param target: An integer
    @return: Find all unique quadruplets in the array which gives the sum of zero
    """
    def fourSum(self, numbers, target):
        if not numbers:
            return []
        self.res = []
        numbers = sorted(numbers)
        for start in range(len(numbers)-3):
            if start > 0 and numbers[start] == numbers[start-1]:
                continue                     # 每层去重放在层首
            end = len(numbers)-1 
            while start+2 < end:          # 注意范围
                if end < len(numbers)-1 and numbers[end] == numbers[end+1]:
                    end -= 1
                else:
                    self.twoSum(numbers, start, end, target - numbers[start] - numbers[end])
                    end -= 1
        return self.res
        
        
    def twoSum(self, numbers, index_start, index_end, target):
        i, j = index_start+1, index_end-1     # i, j起始点在strat, end内侧，记得最后加到res里的是start, end位置的元素
        while i < j: 
            
            if numbers[i] + numbers[j] > target:
                j -= 1 
            elif numbers[i] + numbers[j] < target:
                i += 1 
            elif numbers[i] + numbers[j] == target:
                self.res.append(sorted([numbers[index_start], numbers[i], numbers[j], numbers[index_end]]))
                i += 1
                j -= 1
                while i < j and numbers[i] == numbers[i-1]:  # 这里排除i,j的重复要用while， 且放在 if ==target里面
                    i += 1 
                while i < j and numbers[j] == numbers[j+1]:
                    j -= 1 
# 指针不是为了让你遍历，而是为了将符合某条件的中间的一些值成批量删掉或加上，这就是用指针的意义，加快运算
# 双指针 #dictionary
# 排好序用双指针更快，没有排序要么先排序要么用dictionary

# 题一：将数组中重复的数移到后面，返回unique数的个数
# 法1： 双向指针 nlogn, no extra space
def deduplication(nums):
        
        if len(nums) < 2: return len(nums)
        
        nums.sort()
        lo, hi = 0, len(nums)-1
        while lo < hi:
            while lo < hi and nums[lo] != nums[lo+1]:   # 左指针必须遇到非重复数组才能前进，因为左边存的是unique的数
                lo += 1 
            while lo < hi and nums[hi] == nums[hi-1]:  # 右边可以重复
                hi -= 1
            
            if lo < hi:
                nums[lo], nums[hi] = nums[hi], nums[lo]
                lo += 1 
                hi -= 1
                
        return lo + 1  # print(nums)

# 法二：set存访问过的数， 时间 O(n) 空间 O(n)
def deduplication(nums):
               # write your code here
        left, right = 0, len(nums) - 1
        visited = set()
        while left <= right:
            if nums[left] not in visited:   # 左指针必须没有访问过才能前进
                visited.add(nums[left])
                print(visited)
                left += 1
            else:
                nums[left], nums[right] = nums[right], nums[left]
                right -= 1
        
        return right + 1  # 这里要注意哪个指针代表uniqur数的结束
                
       # print(nums)             


            
# 题五：差值为target,返回这一对数字[num1, num2], 且num1 < num2
# 求差值， 用同向双指针， 
def diffTarget(A, target):
    if not A or not target or len(A) == 0:
        return -1
    A.sort()
    i, j = 0,1
    res = []
    while i < j and j < len(A):
        if A[j] - A[i] == target:
            res.extend([A[i], A[j]])
            break
        elif A[j] - A[i] < target: #右减左 小于target， 右指针向前，找更大的数
            right -= 1 
        else:                       #找到第一个右减左 大于target， 右停下，左向前，缩小当前差值
            left += 1 
    return res
print(diffTarget([2, 7, 15, 24], 5))



# 题七： 数组元素小于k在左边，大于等于k在右边
# 快速选择
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partitionArray(self, nums, k):
        # write your code here
        if not nums:
            return 0
            
        left, right = 0, len(nums)-1
        
        while left < right:
            while left < right and nums[left]  < k:
                left += 1 
                
            while left < right and nums[right] >= k:
                right -= 1 
                
            if left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1 
                right -= 1 
        if nums[left] < k:  #这里的判断很关键，细节处理一定要想清楚分好类
            return left+1
            
        return left

# 题八： sort colors

# 三指针，分成3部分： left, i, right。 跟right交换之后，i不前进； 跟left交换，i和left一起前进；i==1， i前进
class Solution:
    """
    @param nums: A list of integer which is 0, 1 or 2 
    @return: nothing
    """
    def sortColors(self, nums):
        # write your code here
        if not nums:
            return []
            
        left, right = 0, len(nums)-1
        i = 0
        while i <= right:       #这里有等号，因为i停在right停的位置可能是0，1，还得进去循环判断一下
            if nums[i] == 0:
                nums[i],nums[left] = nums[left], nums[i]  
                left += 1 
                i += 1  #跟左边交换之后，i可以前进，因为左边的不是1，就是0， i走过的地方不可能有2
            elif nums[i] == 2:
                nums[i], nums[right] = nums[right], nums[i]
                right -= 1      # 这里right交换后i不前进，因为换过来的数可能还是2，所以只让right向左走，一直找到不是2的数，跟i交换，这样在下个循环i 可以前进  #这就是为什么i走过的地方不可能有2
            else:
                i += 1  #只有i前进，因为left要坚守0的真谛，i一旦发现0，就跟left换，left前进， 也就是left永远在左边第一个不是0 的数的位置
                
        return nums
# 题九： color sort II  k种颜色分成k部分
# nlogk, 对颜色进行二分  [3,2,2,1,4], k=4, 找到颜色的中间是2，九把数组分成大于2和小于等于2的， [2,2,1] [3,4] 再对两个子数组递归
# 需要记录颜色，也就是k的开始和结束， index的开始和结束

class Solution:
    """
    @param colors: A list of integer
    @param k: An integer
    @return: nothing
    """
    def sortColors2(self, colors, k):
        # write your code here
        if not colors or len(colors) == 0:
            return []
            
        self.colorsPatition(colors, 0, len(colors)-1, 1, k)
        
    def colorsPatition(self, colors, index_left, index_right, color_start, color_end):
        if color_start == color_end:
            return
        if index_left >= index_right:
            return
        
        left, right = index_left, index_right  # 记下开始和结束，后面递归要用的
        
        color_mid =  (color_end + color_start)//2   #这里的color_mid就是（1+k）//2, 不是index，数字就代表颜色
        while left <= right:                            #后面递归要用到停止的left和right的位置，作为下一步递归的开始或停止的地方，所以用left<= right,他们会交错，left停在较大值的第一个，right停在较小值的最后一个，
                                                        # 不用额外判断left=right的地方的值是大于还是小于color_mid. 如果用left<right,还有判断他们相遇的地方与color_mid的大小，然后才能判断下一层递归的起点或终点
            while left <= right and colors[left] <= color_mid:
                left += 1 
                
            while left <= right and colors[right] > color_mid:
                right -= 1 
            if left <= right:
                colors[left], colors[right]= colors[right], colors[left]
                left += 1 
                right -= 1 
    
        self.colorsPatition(colors, index_left,right, color_start, color_mid)
        self.colorsPatition(colors,left, index_right, color_mid+1, color_end)
        
#====================================
# 双指针+dict

# 题十： 最长无重复字符的子串
# 方法一： set
class Solution:
    """
    @param s: a string
    @return: an integer
    """
    def lengthOfLongestSubstring(self, s):
        unique_chars = set([])
        j = 0
        n = len(s)
        longest = 0
        for i in range(n):
            while j < n and s[j] not in unique_chars:
                unique_chars.add(s[j])
                j += 1
            longest = max(longest, j - i)
            unique_chars.remove(s[i])
            
        return longest
    
 # 方法二：用dict
# 记录元素和元素的index到字典，遇到重复元素让左指针直接指向当前重复元素前一次出现的index+1.例如：a,b,c,b  第二个b出现时，让左指针指向dict[b]+1,dict[b]存的是第一个b的下标，+1就是左指针指向c。
class Solution:
    """
    @param s: a string
    @return: an integer
    """
    def lengthOfLongestSubstring(self, s):
        # write your code here
        if not s:
            return 0
            
        dict = {}
        longest = 0
        i = 0
        for j in range(len(s)):
            if s[j] in dict:
                i = max(i, dict[s[j]]+1)

            dict[s[j]] = j
            longest = max(longest, j-i+1)
        return longest

# 题十一： 最小子串覆盖
class Solution:
    """
    @param source : A string
    @param target: A string
    @return: A string denote the minimum window, return "" if there is no such a string
    """
    def minWindow(self, source , target):
        s = {x:0 for x in source}
        t = {x:0 for x in target}
        for x in target:
            t[x] += 1
        j = 0
        res = ""
        min_len = len(source)
        for i in range(len(source)):
            while j < len(source):
                if self.contain(s, t):
                    break
                s[source[j]] += 1 
                j += 1 
                
            if self.contain(s,t) and min_len >= j-i:
                min_len = j-i
                res = source[i:j]
            s[source[i]] -= 1   #i更新过程：从0开始，找到第一个覆盖target的子串，i向后移动，找到当前最小覆盖，当i移动到第一个target里的元素被剔除时停下，也就是当前i到j直接无法覆盖target时停下。
                                # 这时新的一轮查找开始，j向后移动直到找到覆盖target的子串，然后i向后移动，找到当前最短覆盖子串
        return res
    
    
    def contain(self, s, t):
        for key, value in t.items():
            if key not in s:
                return False
            if s[key] - value < 0:
                return False 
                
        return True
 # 题三： minimum subarray , 和大于S的最小子数组  # 这里求的是长度最小，要用同向双指针
import sys
class Solution:
    """
    @param nums: an array of integers
    @param s: An integer
    @return: an integer representing the minimum size of subarray
    """
    def minimumSize(self, nums, s):
        if not s:
            return -1
            
        min_size = sys.maxsize
        sum, j = 0, 0
        for i in range(len(nums)):
            while j < len(nums) and sum < s: # j循环的条件 - while
                sum += nums[j]
                j += 1 
            if sum >= s:                    # j停下的条件 - if
                min_size = min(min_size, j-i)
            sum -= nums[i]
        if min_size == sys.maxsize:
            return -1
        return min_size
      
# 题十二：接雨水 
#本质4指针相向而行
class Solution:
    """
    @param heights: a list of integers
    @return: a integer
    """
    def trapRainWater(self, heights):
        if not heights:
            return 0 
        res = 0    
        left, right = 0, len(heights)-1  # 查找指针
        lefthighest = heights[left]  # 记录最左边/ 最右边最高海拔的指针
        righthighest = heights[right]
        
        while left < right:
            if lefthighest < righthighest: # 谁低舍弃谁
                left += 1 
                if lefthighest > heights[left]:  
                    res += lefthighest - heights[left]
                else:
                    lefthighest = heights[left]
            else:
                right -= 1 
                if righthighest > heights[right]:
                    res += righthighest - heights[right]
                else:
                    righthighest = heights[right]
                    
        return res
# 题十三：搜索区间 - 连续使用二分法
# 先二分找到target第一次出现的位置leftBound, 再二分查找leftBound到结尾，找到target最后一次出现的位置
class Solution:
    """
    @param A: an integer sorted array
    @param target: an integer to be inserted
    @return: a list of length 2, [index1, index2]
    """
    def searchRange(self, A, target):
        if not A:
            return [-1,-1]
            
        start, end = 0, len(A)-1
        while start + 1 < end:
            mid = (start + end)//2 
            if A[mid] < target:
                start = mid 
            else:
                end = mid 
        if A[start] == target:
            leftBound = start
        elif A[end] == target:
            leftBound = end 
        else:
            return[-1, -1] 
        
        start, end = leftBound, len(A)-1 
        while start + 1 < end:
            mid = (start + end) // 2 
            if A[mid] <= target:
                start = mid 
            else:
                end = mid 
        if A[end] == target:
            return [leftBound, end]
        else:
            return [leftBound, start]
# 题十四： 排序数组，找到第一个小于等于target的数

def findFirstLess(self, presum, target):
        m = len(presum)
        if presum[m-1] < target:
            return m
        
        start, end = 0, m - 1
        while start + 1 < end:
            mid = (start + end) / 2
            if target <= presum[mid]:
                end = mid 
            else:
                start = mid
        
        if presum[end] < target: # 前面是二分的模版， 最后的判断条件是根据题意变化的。走到这一步，start肯定是小于等于target的，end不确定，所以先判断presum[end]跟target的大小
            return end + 1
        if presum[start] < target: # 如果presum[end] > target, 再判断presum[start]与target的大小
            return start + 1
        return 0   # target太小，返回最小的位置
