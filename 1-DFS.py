# 什么时候用 DFS? • 求所有方案时
# 怎么解决DFS?
# 不是排列就是组合

# 通用的DFS时间复杂度计算公式，O(答案个数 * 构造每个答案的时间)
# DFS 时间复杂度2^n, 组合问题复杂度与n!相关
# DFS 有 数组的， 字符串的，（组合问题，排列问题）和二叉树的遍历
# 模版必背

# 数组dfs题型一：subsets 是数组深度搜索的基础
# 组合问题
# 题一：subsets I 
# 核心：递归
# 作为模版记住

# 方法一：
class Solution:
    """
    @param nums: A set of numbers
    @return: A list of lists
    """
    def subsets(self, nums):
        nums = sorted(nums)
        res = []
        self.dfs(nums, 0, [], res)
        return res
        
    def dfs(self, nums, index, S, res):
        res.append(list(S))
        
        for i in range(index, len(nums)):
            S.append(nums[i])
            self.dfs(nums, i + 1, S, res)  # 每一层返回之后，将当前S的最后元素pop出来， 一定要记得，这时还在上一层递归的for循环里，如果i+1<len(nums),  再次进入 for循环，如果进不去，当前dfs返回
            S.pop()
# [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]

# 方法二：
class Solution:
    def subsets(self, nums):
        self.results = []
        self.search(sorted(nums), [], 0)
        return self.results 
    
    def search(self, nums, S, index):
        if index == len(nums):
            self.results.append(list(S))
            return
        
        S.append(nums[index])
        self.search(nums, S, index + 1)  # 从0位置开始，先全都加进来，然后pop最后一个，看是否还能加进来， 一直这样pop，然后搜索能加进来,知道S pop成空时，移动到1位置，从1位置开始，向后每个元素加进来，再从最后pop
        S.pop()
        self.search(nums, S, index + 1)

solution = Solution()
print(solution.subsets([1,2,3]))
# [[1, 2, 3], [1, 2], [1, 3], [1], [2, 3], [2], [3], []]

# 题二：subsets II  去重
class Solution:
    """
    @param nums: A set of numbers.
    @return: A list of lists. All valid subsets.
    """
    def subsetsWithDup(self, nums):
        nums = sorted(nums)
        res = []
        self.dfs(nums, 0, [], res)
        return res
        
    def dfs(self, nums, index, S, res):
        res.append(list(S))
        
        for i in range(index, len(nums)):
            if i != index and nums[i] == nums[i-1]:  # 判断当前i，如果i不是起点，且num[i] = nums[i-1]说明当前为重复元素，需要跳过去
                continue
            S.append(nums[i])
            self.dfs(nums, i + 1, S, res) 
            S.pop()

# 数组深度搜索题型二：combination sum 子数组和

# 与subsets题的区别
# 1， 多了限制，sum
# 2， sum可能有重复，去重
# 3， subsets每个元素只能选一次，combination sum每个元素可以重复选。所以搜索的时候从index开始而不是index+1

# 题三：[2,3,6,7]找到所有和为7的数， 可重复使用
# 组合问题
# 套用susets模版，只是return条件改成sum(S) == target
# 剪枝 当前sum(S)+num[i]>target时，break 不用往后看了，后面一定也大于target
class Solution:
    """
    @param candidates: A list of integers
    @param target: An integer
    @return: A list of lists of integers
    """
    def combinationSum(self, num, target):
        if not num or target == None:
            return []
        num= sorted(num)    
        self.res = []
        self.dfs(num, 0, [], target)
        return self.res
        
    def dfs(self, num, index, S, target):
        if sum(S) == target:
            self.res.append(list(S))
        
        for i in range(index, len(num)):
            if i!= 0 and num[i] == num[i-1] and i>index:    # 去重   # 本题也可以用num = sorted(list(set(num))) 去重，但是combination sumII不可以直接用set去重，因为本题元素可重复用，II中元素不可以重复用。
                continue
            if sum(S) + num[i] > target:  # 剪枝
                break
            S.append(num[i])
            self.dfs(num, i, S, target)
            S.pop()

         
# 题四： calculate sumII  与I区别，原candidates不需要去重，只需排序，但是求出来的result需要去除， 比如 1' 1'' 1''' 选谁， 标准是需要一个1，选1', 需要2个1， 选1', 1''
# 组合问题
# 方法一：subsets模版
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, num, target):
        if not num or target == None:
            return []
        num = sorted(num)    
        self.res = []
        self.dfs(num, 0, [], target)
        return self.res
        
    def dfs(self, num, index, S, target):
        if sum(S) == target:
            self.res.append(list(S))
        
        for i in range(index, len(num)):
            if i!= 0 and num[i] == num[i-1] and i>index:    # 去重 # 不可以直接用set去重，因为本题的元素不能重复用，假设有2个1，那么1最多用2次，如果用set去重，就变成只有1个1了
                                                            # 这里去重 去的是同一个index（即相同offset），就是从1开始的S，向后移动时，又碰到一个1，就不可以从再算一次从1开始的S
                                                            # 但是，在同一offset下，求S的和，是不用去重的。S=[1,1,2]这里本来就要2个1，所以不用去掉第2个1，去掉就不对了
                                                             #符合上述if判断条件， continue，继续向后查看， 注意不是break, 因为后面的可能有2，3...
                continue
            if sum(S) + num[i] > target:
                break
            S.append(num[i])
            self.dfs(num, i+1, S, target)
            S.pop()
 
# 方法二: 引入remainTarget = target - 当前sum(S)， 当remainTarget=0时，就让S加入res。
           
class Solution:
    """
    @param num: Given the candidate numbers
    @param target: Given the target number
    @return: All the combinations that sum to target
    """
    def combinationSum2(self, candidates, target):
        results = []
    
        # 集合为空
        if len(candidates) == 0:
            return results
        
        # 利用set去重后排序
        candidatesNew=sorted(candidates)
        # dfs
        self.dfs(candidatesNew, 0, [], target, results)
    
        return results

    def dfs(self, candidatesNew, index, current, remainTarget,  results):
        # 到达边界
        if remainTarget == 0:
            results = results.append(list(current))
            return results

        # 递归的拆解：挑一个数放入current
        for i in range(index, len(candidatesNew)):
            if i!= 0 and candidatesNew[i] == candidatesNew[i-1] and i>index:  
                continue
           
            # 剪枝
            if remainTarget < candidatesNew[i]:
                break

            current.append(candidatesNew[i])
            self.dfs(candidatesNew, i+1, current, remainTarget - candidatesNew[i], results)
            current.pop()

            
# 题五：分割回文串 （组合问题）
# 字符串dfs - 想成字符与字符中间有分割线，a| b| b|a, 假设分割线1,2,3， 对字符串abba的分割就是对1,2,3的任意组合，所以是dfs
# 再一次理解dfs, （第1层）对abba （i可以取1，2，3，4）， i=1时就是在第一个元素a后分割，将‘a’加入stringlist，然后将其后元素bba作为s进行dfs递归；（第2层）对bba（i可以取1,2,3），i=1时就是在元素b后分割，将‘b’加入stringlist, 然后将其后元素ba作为s进行dfs递归；
# （第3层）对ba(i可以取1，2)，i=1时就是在b后分割， 将‘b’加入stringlist, 将其后元素a进行dfs递归； （第4层）对a（i只能取1），i=1时就是在a出分割，将‘a’加入stringlist, 将其后“”进行dfs递归；（最后一层）就到了递归的出口,将当前stringlist=['a', 'b', 'b', 'a']加入res
# 从最后一层返回到第4层， i=1的循环还没完，还有stringlist.pop(), stringlist变成['a', 'b','b'], i只有一个值，刚刚取过了，这一层再返回第3层, 当前层i=1的循环还没完，还有stringlist.pop(), stringlist变成['a', 'b']，i向后走到2， i=2就是在第二个元素a后面分割，这一层加到stringlist里面的本来应该是'ba',但是‘ba’不是回文，所以不能加进stringlist,i无法再进for循环，i不能等于3，也就无法走到递归出口，无法return,也就不用pop然后返回第2层；
# 第2层i=1的循环还没走完，stringlist.pop(),stringlist变成['a'], 继续for循环，i=2, 就是在bba的第2个元素后分割，也就是‘bb’,判断'bb'是回文，加入stringlist,stringlist变成['a', 'bb'],将其后元素继续dfs递归，a作为s进入递归，判断‘a’是回文，将‘a’加入stringlist, 变成['a', 'bb', 'a'], 将a后面的""放进dfs递归，找到出口，将当前stringlist放到res里。返回当前层i =2, stringlist.pop(), 变成['a', 'bb']。返回当前层i=1, stringlist.pop(), 变成['a']
# 第1层的i=1还没走完，stringlist.pop(),stringlist变成[]; i=2,就是在第2个元素分割即‘ab’, 不是回文，不加入stringlist,也不用pop; i=3, 分割aab, 不是回文; i=4, abba, 是回文， stringlist=['abba'],将其后""加入dfs递归，找到出口，将当前stringlist加入res， 返回当前层i=4， stringlist.pop(), stringlist = [] ， 完结。

class Solution:

    def partition(self, s):
        results = []
        self.dfs(s, [], results)
        return results
    
    def dfs(self, s, stringlist, results):
        if len(s) == 0:         # 递归的出口   # 因为s起点在一点点向后移动，s的长度在缩短，说明分割线在向后移动，当len(s)=0时，说明已经分割完了，且每个部分都是回文
            results.append(list(stringlist))
            return
            
        for i in range(1, len(s) + 1):  # 从1开始，因为要取得前i个字符，从0开始就取不到了
            prefix = s[:i]
            if self.is_palindrome(prefix): # 只有回文才能继续
                stringlist.append(prefix)
                self.dfs(s[i:], stringlist, results)  # 这里的offset是s[i:]，起点从i开始
                stringlist.pop()

    def is_palindrome(self, s):
        return s == s[::-1]
# 题六：全排列  （排列问题）
# 无重复元素
# []->[1]->[1,2]->[1,2,3]return->[1,2] -> [1] -> [1,3]->[1,3,2]return->[1,3]->[1]->[]->[2]->[2,1]->[2,1,3]return->[2,1]->[2,3]->[2,3,1]return->[2,3]->[2]->[]->[3]->[3,1]->[3,1,2]return->[3,1]->[3]->[3,2]->[3,2,1]return

class Solution:
    """
    @param: nums: A list of integers.
    @return: A list of permutations.
    """
    def permute(self, nums):
        if not nums:
            return [[]]
            
        res = []
        self.dfs(nums, [], set(), res)
        return res 
        
    def dfs(self, nums, permutation, visited, res):
        if len(permutation) == len(nums):  # 位置占满，加到res， 
            res.append(list(permutation))
            return 
        for num in nums:  # 每个位置可能取到所有的元素，所以要对全nums进行for 循环
            if num in visited:  # 前面取过的，在当前位置不能再取
                continue 
            
            permutation.append(num)
            visited.add(num)
            self.dfs(nums, permutation, visited, res)
            permutation.pop()  # pop最后一个元素
            visited.remove(num)  # remove当前的num # 注意这pop和remove的一定是删除的同一个元素，只是函数不同
            
# 题七：全排列 （有重复元素）
# 关键：怎么处理重复元素。如果用题六的解法，只用一个set，显然无法满足重复元素排列。比如[1,1,2], 用set就会把第二个1跳过，是不对的。
# 正确的方法是，用True/False 数组表示每个元素是否在当前排列被访问，假设【1，1，2】, boolean数组[False, False, False] 两个1不会被互相影响，前面的1被访问，变成true，第2个1还是false,还可以用
# 但是boolean数组无法解决 offset向后移，碰到相同元素的问题。这时就单独判断 [nums[i] == nums[i - 1] and self.visited[i - 1]

class Solution:
    """
    @param: :  A list of integers
    @return: A list of unique permutations
    """

    def permuteUnique(self, nums):
        self.results = []
        self.visited = {i: False for i in range(len(nums))}
        self.dfs([], sorted(nums))
        return self.results
        
    def dfs(self, path, nums) :
        if len(path) == len(nums) :
            self.results.append(path[:])
            return
        
        for i in range(len(nums)) :
            if self.visited[i] :
                continue
            
            if i != 0 and nums[i] == nums[i - 1] and self.visited[i - 1]:
                continue
            
            self.visited[i] = True
            path.append(nums[i])
            self.dfs(path, nums)
            path.pop()
            self.visited[i] = False
# 题八：字符串解码 expression expanding
#字符串递归
class Solution:
    """
    @param s: an expression includes numbers, letters and brackets
    @return: a string
    """
    def expressionExpand(self, s):
        if not s:
            return ""
        self.right = 0
        return self.dfs(s, 0, '')
        
    def dfs(self, s, index, res):
        while index < len(s):
            if s[index] == ']': 
                self.right = index
                return res
            elif s[index].isdigit():
                num = ''
                while s[index] != '[':   #这里把‘[’跳过去了，所以外面if条件里不会遇到‘[’
                    num += s[index]     # 数字在这里也是字符，所以要把多位数找全
                    index += 1 
                res += int(num) * self.dfs(s, index+1, '')
            else:
                res += s[index]
            index = 1 + max(index, self.right)
        return res
# 二叉树3种遍历+迭代器 （非递归，stack）  -- 必背
# 题九 二叉树迭代器
# 方法一：
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
        while root:      # 初始化就可以将root开始的所有左儿子加入stack
            self.stack.append(root)
            root = root.left

    """
    @return: True if there has next node, or false
    """
    def hasNext(self):
        return bool(self.stack)
    """
    @return: return next node
    """
    def next(self):   # 这里不可以用while stack, 判断stack是否存在已经在hasNext判断了，如果这里用while， 就不是迭代器一步一走了，就停不下来了
        res = node = self.stack.pop()
        if node.right:          # 判断node.right是否存在，存在就把node.right这一支的left都加进stack
            node = node.right
            while node:
                self.stack.append(node)
                node = node.left
       
        return res  # 注意：这里返回的是当前pop出的节点，node作为指针，已经从351行开始指向其他节点了，所以如果return node 就不对了，所以让res固定当前pop出的那个节点
# 方法二： “一路向左” - 将向左找到所有左儿子写成一个函数，因为这样过程被重复使用了 

class BSTIterator(object):
    def __init__(self, root):
        self.stack = []
        self.pushLeft(root)

    def next(self):
        node = self.stack.pop()
        self.pushLeft(node.right)
        return node.val

    def hasNext(self):
        return self.stack != []
    
    def pushLeft(self, node):
        while node:
            self.stack.append(node)
            node = node.left
            
            
# 题十：前序遍历
# 题十一： 中序遍历
# 题十二： 后序遍历  压入栈内的是（node, time）， 第一层被弹出是需要把右边的孩子加入stack,第二次弹出是把自己加到res里，所以当time=1时，就不要再向右找了，之前找过了，直接将node加入res就可以了， 不然就死循环了
