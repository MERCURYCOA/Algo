# 题一：字符串查找
# 方法一： O(n*m)
class Solution:
    """
    @param source: 
    @param target: 
    @return: return the index
    """
    def strStr(self, source, target):
        if not target:
            return 0 
        if not source:
            return -1 
            
        BASE = 10**6
        target_hash = 0 
        for x in target:
            target_hash = (target_hash *31 + ord(x)) % BASE 
        source_hash = 0 
        for i in range(len(source)):
            source_hash = (source_hash *31 + ord(source[i])) % BASE
            if i < len(target)-1:
                continue 
            if source_hash == target_hash:
                return i - len(target) + 1
            source_hash = (source_hash - ord(source[i-len(target)+1])*(31**(len(target)-1))) % BASE 
            if source_hash < 0:
                source_hash += BASE 
                
        return -1
# 方法二： 优化方法一中字符串比较的时间，字符串的比较用O(m),整数比较O(1),所以想到用哈希函数将string转换成整数
# 关键：会写hash function
class Solution:
    """
    @param source: 
    @param target: 
    @return: return the index
    """
    def strStr(self, source, target):
        if not target:
            return 0 
        if not source:
            return -1 
            
        BASE = 10**6   
        target_hash = 0 
        for x in target:
            target_hash = (target_hash *31 + ord(x)) % BASE   # 每一步都取模，因为怕31的高次幂超出int边界
        source_hash = 0 
        for i in range(len(source)):
            source_hash = (source_hash *31 + ord(source[i])) % BASE  # hash('abc') = a *31^2 + b *31 + c
            if i < len(target)-1:
                continue 
            if source_hash == target_hash:
                return i - len(target) + 1
            source_hash = (source_hash - ord(source[i-len(target)+1])*(31**(len(target)-1))) % BASE   
                                          # hash('abc'-'a') = a *31^2 + b *31 + c - a * 31^2   这里把a减掉，下一个循环加上source[i]，记得每一步都要取模
            if source_hash < 0:
                source_hash += BASE   # 26行，正数 - 正数， 是有可能变成负数的，因为取BASE的模，所以加一个BASE变正数
                
        return -1
