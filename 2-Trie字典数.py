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
                node.children[c] = TrieNode()  # {'a', TrieNode({}, False)}
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
