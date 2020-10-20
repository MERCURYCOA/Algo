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
            node = node.children[c]
        
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
        
  
