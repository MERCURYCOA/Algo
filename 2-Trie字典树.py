# 什么时候用trie树： 1 需要一个一个字母进行遍历  2 需要前缀特性  3 需要省空间
# 题一：实现Trie
# 注意：实例化一个TrieNode包括它本身，他的children,和is_word。在字典里的映射不是TrieNode:{children},而是字符：TrieNode,例如：‘a’:TrieNode({children},is_word)。
class TrieNode:
    
    def __init__(self):
        self.children = {}
        self.is_word = False
    
    
class Trie:
    
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()  # {'a'： TrieNode({}, False)}
            node = node.children[c]   #####千万注意，逻辑要清楚， c不能在if内，要与if并行，因为无论如何c都要加入
        
        node.is_word = True

    def find(self, word):
        node = self.root
        for c in word:
            node = node.children.get(c)   # 找到当前字符例如‘a’所对于的TrieNode
            if node is None:
                return None
        return node
        
    def search(self, word):
        node = self.find(word)
        return node is not None and node.is_word

    def startsWith(self, prefix):
        return self.find(prefix) is not None
# 题二： 单词的添加与查找 与题一不同，‘.’可以匹配任意字符
# 想到递归，想到对find(node, subword)进行递归，这样并不容易实现。正确的是，word作为参数在每层传入， 同时用index进行递归，向前推进
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class WordDictionary:
 
    def __init__(self):
        self.root = TrieNode()
    
    def addWord(self, word):
        node = self.root
        for x in word:
            if x not in node.children:  # 注意：x是key， value是x对应的TrieNode， children字典内部是{'a'： TrieNode({}, False)}
                node.children[x] = TrieNode()
            node = node.children[x]
        node.is_word = True

    def find(self, node, word, index):
        if node is None:
            return False 
            
        if index >= len(word):      # 一条链上有’abc‘, word=‘ab'，node到达b节点时，word已经查看完了，这时必须看b节点是不是is_word, 如果不是，说明word只是这条链上的子串。
            return node.is_word
        
        char = word[index]   # 这里就不能用for循环了。要注意，for/ while循环都是枚举，迭代，跟递归不能混在一起用。这里是查看当前index的字符。
        if char != '.':
            return self.find(node.children.get(char), word, index+1)
        for c in node.children:   
            if self.find(node.children[c], word, index+1):  # 注意这里children字典里存的是字符到该字符所代表的TrieNode的映射。不能直接将c放到node参数位置，应该查看c在字典中的映射
                return True
        return False
        
    def search(self, word):
        if not word:
            return False
        return self.find(self.root, word, 0)
    
# 题三： 查找树服务

"""
Definition of TrieNode:
class TrieNode:
    def __init__(self):
        # <key, value>: <Character, TrieNode>
        self.children = collections.OrderedDict()
        self.top10 = []
"""
class TrieService:

    def __init__(self):
        self.root = TrieNode()

    def get_root(self):
        # Return root of trie root, and 
        # lintcode will print the tree struct.
        return self.root

    # @param {str} word a string
    # @param {int} frequency an integer
    # @return nothing
    def insert(self, word, frequency):
        # Write your your code here
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            node.top10.append(frequency)
            node.top10.sort(reverse=True) # 也可以不每次都sort，因为之前的已经都是sorted过的，就从后向前比较，交换，类似插入排序那样，插到对的位置，
            
            if len(node.top10) > 10:
                node.top10.pop()
        
         
  # 题四： 字典树 序列化与反序列化
"""
Definition of TrieNode:
class TrieNode:
    def __init__(self):
        # <key, value>: <Character, TrieNode>
        self.children = collections.OrderedDict()
"""


class Solution:

    '''
    @param root: An object of TrieNode, denote the root of the trie.
    This method will be invoked first, you should design your own algorithm 
    to serialize a trie which denote by a root node to a string which
    can be easily deserialized by your own "deserialize" method later.
    '''
    def serialize(self, root):
        # Write your code here
        if root is None:
            return ""
        data = ""
        
        for k, v in root.children.items():  # 递归
            data += k + self.serialize(v)
        
        return '<%s>' % data    # 注意记住这种写法，不需要对每个char都补上‘<''>'
            


    '''
    @param data: A string serialized by your serialize method.
    This method will be invoked second, the argument data is what exactly
    you serialized at method "serialize", that means the data is not given by
    system, it's given by your own serialize method. So the format of data is
    designed by yourself, and deserialize it here as you serialize it in 
    "serialize" method.
    '''
    def deserialize(self, data):  # 树的层级便利（list, pop ）
        # Write your code here

        if data is None or len(data) == 0:
            return None 
        root = TrieNode()
        current = root
        path = []

        for c in data:
            if c == '<':
                path.append(current)
            elif c == '>':
                path.pop()
            else:
                path[-1].children[c] = TrieNode()
                current = path[-1].children[c]
        return root
        
  
