from AVLTree import AVLTree, Node
from RedBlackTree import RedBlackTree
import re
from sys import exit
import numpy as np

file = open('glove.6B.50d.txt', 'r')
print('Select one of the trees to use\n1.AVL : AVL Tree\n2.RB : Red Black Tree\n')
select = input('Your selection is : ')
print('\n')

if select == 'AVL' or select == 'AVL Tree':
    tree = AVLTree()
elif select == 'RB' or select == 'Red Black Tree':
    tree = RedBlackTree()
else:
    print("'%s' not a valid selection" % select)
    exit()

for line in file:
    words = line.split()
    if re.match(r'[^\w]', words[0]):
        continue
    if select == 'AVL' or select == 'AVL Tree':
        node = Node(words)
        tree.insert(node)
    else:
        tree.insert(words)


def count_nodes(tree):
    return len(tree)


# Compute the "maxDepth" of a tree -- the number of nodes
# along the longest path from the root node down to the
# farthest leaf node
def maxDepth(node):
    if node is None:
        return 0

    else:

        # Compute the depth of each subtree
        lDepth = maxDepth(node.left)
        rDepth = maxDepth(node.right)

        # Use the larger one
        if lDepth > rDepth:
            return lDepth + 1
        else:
            return rDepth + 1


# global variable prev - to keep track
# of previous node during Inorder
# traversal
prev = None
node_keys = []

# function to check if given binary
# tree is BST
def isbst(root):
    # prev is a global variable
    global prev
    prev = None
    return isbst_rec(root)


# Helper function to test if binary
# tree is BST
# Traverse the tree in inorder fashion
# and keep track of previous node
# return true if tree is Binary
# search tree otherwise false
def isbst_rec(root):
    # prev is a global variable
    global prev

    # if tree is empty return true
    if root is None:
        return True

    if isbst_rec(root.left) is False:
        return False

    # if previous node'data is found
    # greater than the current node's
    # data return fals
    if prev is not None and prev.get_key() > root.get_key():
        return False

    # store the current node in prev
    prev = root
    node_keys.append(prev.get_key())
    return isbst_rec(root.right)

def similarity(filename,tree):
    print('Word Similarities:')
    with open(filename,'r') as word_pair_file:
        for line in word_pair_file:
            word_pairs = line.split()
            node1 = tree.search(word_pairs[0])
            node2 = tree.search(word_pairs[1])
            if node1 is None or node2 is None:
                print('%s %s Embedding Not Found'%(word_pairs[0],word_pairs[1]))
                continue
            embedding1 = np.array(node1.get_embedding(), dtype=np.float)
            embedding2 = np.array(node2.get_embedding(), dtype=np.float)
            similarity_score = np.divide(np.dot(embedding1,embedding2),np.linalg.norm(embedding1)*np.linalg.norm(embedding2))
            print('{0} {1} {2}'.format(word_pairs[0], word_pairs[1], similarity_score))


# To test the various methods
print("tree count", count_nodes(tree))
print("tree height", maxDepth(tree.get_root()))
if isbst(tree.get_root()):
    with open('Words_In_Tree.txt','w') as out_file:
        for word in node_keys:
            out_file.write('%s\n' % word)
    out_file.close()
similarity('word_pair.txt',tree)

