import math
import random
from past.builtins import raw_input
from pygments.util import xrange


################### Method to read dataset ##########################
# Method to read dataset
def readData(path):
    with open(path) as file: lines = file.readlines()
    data = []
    for line in lines: data.append(line.strip().split(','))
    # The first line of data is header, and the following lines are data.
    header = data[0]
    data = data[1:]
    return header, data


# Method to get attributes from the input dataset, including # of attributes, # of instances,
def getAttributes(data):
    return len(data[0]), len(data)


################ Methods to select the attribute with most Information Gain ###################################
# Method to get counts of each case:
# Let x denotes explanatory variable, and y denotes resposne variable.
# We need to get the counts of (X, Y): (True, True), (True, False), (False, True), (False, False);
def getCounts(data):
    if (not data or len(data) == 0):
        return [[0, 0, 0, 0]]
    nrow = len(data)
    ncol = len(data[0])
    counts = [[0 for i in range(4)] for i in range(ncol)]

    for col in range(ncol):
        for row in range(nrow):
            num = data[row][col]
            if (num == '1'):
                if (data[row][-1] == '0'):
                    counts[col][1] += 1
                else:
                    counts[col][0] += 1
            else:
                if (data[row][-1] == '1'):
                    counts[col][2] += 1
                else:
                    counts[col][3] += 1
    return counts


# Given data, calculate the entropy of the response variable.
def calEntropy(numT, numF):
    total = numT + numF
    if (total == 0): return 0
    pT = percentage(numT, total)
    pF = percentage(numF, total)
    return abs(-pT * log2(pT) - pF * log2(pF))


# Calculate the log2 value of a given number num
def log2(num):
    if (num == 0): return 0
    return math.log(num, 2)


# Calculate the percentage of True given the number of true and the number of total
def percentage(numT, total):
    if (total == 0): return 0
    return float(numT) / total


# Get the index of atttribute that maximizes the Information Gain from a counts matrix
def maximizedIG(counts):
    numAttr = len(counts)
    maxId = 0
    minH = 1
    for i in range(numAttr - 1):
        count = counts[i]
        temp = calH(counts, i)
        if (temp < minH):
            maxId, minH = i, temp
    return maxId


# Caldulate the H(Y|X) of of the i-th variable
def calH(counts, i):
    count = counts[i]
    TT = count[0]
    TF = count[1]
    FT = count[2]
    FF = count[3]
    return calEntropy(TT, TF) * percentage(TT, TT + TF) + calEntropy(FT, FF) * percentage(FF, FT + FF)


# Divide the original dataset into two subset according to the value of an attribute i
def divMatrix(data, maxId):
    dataOne = []
    dataTwo = []
    for i in range(len(data)):
        row = data[i][:]
        val = row[maxId]
        del row[maxId]
        if (val == '0'):
            dataOne.append(row)
        else:
            dataTwo.append(row)
    return dataOne, dataTwo


################ Methods to build up trees ###################################
# To implement a tree, we need a class of TreeNode
class TreeNode(object):
    def __init__(self, attrName, attrVal, nodeData, leftChild, rightChild, level, outputClass, label):
        self.attrName = attrName
        self.attrVal = attrVal
        self.nodeData = nodeData
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.level = level
        self.outputClass = outputClass
        self.label = label

    def toString(self):
        if (self.level == 0): return ""
        s = ''
        if (self.level > 0):
            for i in range(self.level - 1):
                s = s + '|'
        s = s + self.attrName + " = " + self.attrVal + ":"
        if (self.leftChild == None and self.rightChild == None):
            s = s + self.outputClass
        return s


# ID3 Method to build up a Decision tree
def ID3buildTree(data, header, level, attrName, attrVal):
    # print 'calling buildeTree(data, header, ', level, ',', attrVal,')'
    # Find the attribute with most information gain
    counts = getCounts(data)
    # Stop building tree if reaching the leaf: only single class included or no more attributes to divide
    yDist = counts[len(counts) - 1]
    dataEntropy = calEntropy(yDist[0] + yDist[1], yDist[2] + yDist[3])

    if (dataEntropy == 0 or dataEntropy == -0 or len(counts) == 1):
        outputClass = "0"
        if (yDist[0] > yDist[3]):
            outputClass = "1"
        return TreeNode(attrName, attrVal, data, None, None, level, outputClass, None);

    maxId = maximizedIG(counts)
    # Divide the data into two sub data according to value of the attribute with most information gain
    dataOne, dataTwo = divMatrix(data, maxId)

    # Two sub data are used as the left child and right child of node
    left = ID3buildTree(dataOne, header, level + 1, header[maxId], "0")
    right = ID3buildTree(dataTwo, header, level + 1, header[maxId], "1")
    # Return the node
    node = TreeNode(attrName, attrVal, data, left, right, level, None, None)
    return node

# Random Method to build up a Decision tree
def RandombuildTree(data, header, level, attrName, attrVal):
    # print 'calling buildeTree(data, header, ', level, ',', attrVal,')'
    # Find the attribute with most information gain
    counts = getCounts(data)
    # Stop building tree if reaching the leaf: only single class included or no more attributes to divide
    yDist = counts[len(counts) - 1]
    dataEntropy = calEntropy(yDist[0] + yDist[1], yDist[2] + yDist[3])

    if (dataEntropy == 0 or dataEntropy == -0 or len(counts) == 1):
        outputClass = "0"
        if (yDist[0] > yDist[3]):
            outputClass = "1"
        return TreeNode(attrName, attrVal, data, None, None, level, outputClass, None);

    Natt = len (data[0])
    RanId = random.randrange(Natt)
    # Divide the data into two sub data according to value of the attribute with most information gain
    dataOne, dataTwo = divMatrix(data, RanId)

    # Two sub data are used as the left child and right child of node
    left = RandombuildTree(dataOne, header, level + 1, header[RanId], "0")
    right = RandombuildTree(dataTwo, header, level + 1, header[RanId], "1")
    # Return the node
    node = TreeNode(attrName, attrVal, data, left, right, level, None, None)
    return node

# Method to print a Decision Tree
def printTree(node):
    if not node: return
    print(node.toString())
    printTree(node.leftChild)
    printTree(node.rightChild)

def labelTree(root):
    queue = [root]
    label = 1
    while len(queue) != 0:
        size = len(queue)
        for i in range(size):
            node = queue[0]
            node.label = label
            del queue[0]
            label = label + 1
            if (node.leftChild): queue.append(node.leftChild)
            if (node.rightChild): queue.append(node.rightChild)
    return

def traverse(node):
    ## stop iteration at leaf node
    if (node.leftChild == None and node.rightChild == None): return 1, 1

    leftTot, leftLeaf = traverse(node.leftChild)
    rightTot, rightLeaf = traverse(node.rightChild)

    return leftTot + rightTot + 1, leftLeaf + rightLeaf

def prunning(node, pruneFactor):
    # the total # of internal nodes in the tree
    totNode, totLeaf = traverse(node)
    numToPrune = int(totNode * pruneFactor)
    randomInt = random.sample(xrange(totNode), numToPrune)

    for index in randomInt:
        prunningGivenIndex(node, index)
    return


def prunningGivenIndex(node, index):
    # print "calling prunningGivenIndex(node, ", index
    if (node == None): return

    if index == node.label:
        node.leftChild = None
        node.rightChild = None
        # redefine the output class
        counts = getCounts(node.nodeData)
        response = counts[len(counts) - 1]
        if response[0] > response[3]:
            node.outputClass = '1'
        else:
            node.outputClass = '0'
        return

    prunningGivenIndex(node.leftChild, index)
    prunningGivenIndex(node.rightChild, index)
    return

############################## Accuracy Test ###################################
# To faciliate fast locating the position of key attribute, we create the mapping between attribute name and its index
def mapping(header):
    dict = {}
    for i in range(len(header)):
        dict[header[i]] = i
    return dict

# Given one record, predict its value according to the decision tree
def predict(root, dict, record):
    node = root
    while node.outputClass == None:
        leftAttr, leftAttrVal = node.leftChild.attrName, node.leftChild.attrVal
        idx = dict[leftAttr]
        if (record[idx] == leftAttrVal):
            node = node.leftChild
        else:
            node = node.rightChild
    return node.outputClass

# Given a test set, calculate the accuracy of the decision tree
def accuracy(root, header, testSet):
    if (testSet == None): return 0
    dict = mapping(header)
    total = len(testSet)
    numCorrect = 0
    for i in range(total):
        record = testSet[i]
        predictVal = predict(root, dict, record)
        if (predictVal == record[-1]): numCorrect = numCorrect + 1
    return float(numCorrect) / total

'''
#Give a tree root, calculate the average depth of the leaves
def AvgDepth(node,leafsum=0,sumdepth=0):
    if node.leftChild != None:
        return AvgDepth(node.leftChild,leafsum=0,sumdepth=0)
    if node.rightChild != None:
        return AvgDepth(node.rightChild,leafsum=0,sumdepth=0)
    if node.leftChild == None and node.rightChild == None:
        leafsum = leafsum + 1
        sumdepth = sumdepth + node.level
        return sumdepth/leafsum
'''
def totDepth(node):
    if(node == None): return 0
    if(node.leftChild == None and node.rightChild == None): return node.level
    return totDepth(node.leftChild) + totDepth(node.rightChild)

##################################### Main #####################################

# Read the Training set
#trainSet_Path = raw_input("Training dataset path: ")
#validationSet_Path = raw_input("Validation dataset path: ")
#testSet_Path = raw_input("Test dataset path: ")
trainSet_Path = input("Training dataset path: ")
validationSet_Path = input("Validation dataset path: ")
testSet_Path = input("Test dataset path: ")
pruneFactor = float(input("Pruning factor:"))

# Read dataset
header, trainSet = readData(trainSet_Path)
ncol, nrow = getAttributes(trainSet)

validationHeader, validationSet = readData(validationSet_Path)
testHeader, testSet = readData(testSet_Path)

# Apply ID3 algorithm and build tree
root = ID3buildTree(trainSet, header, 0, None, None)
# print(node.nodeData)
print("")
print("Tree Representation")
print("----------------------------------")
printTree(root)
print("")

# print summary of the tree
totNode, totLeaf = traverse(root)

#AvergDepth = AvgDepth(root,leafsum=0,sumdepth=0)
AveDepth = float(totDepth(root)) / totLeaf
print("ID3 average depth = {0:.02f}".format(AveDepth))
print("")
###################### Pre prunned ######################
print("ID3 Accuracy")
print("----------------------------------")
print("Number of training instances = ", nrow)
print("Number of training attributes = ", ncol - 1)
print("Total number of nodes in the tree = ", totNode)
print("Total number of leaf nodes in the tree = ", totLeaf)
print("Accuracy of the model on the training dataset = {0:.02f}%".format(accuracy(root, header, trainSet) * 100))
print("")

ncol, nrow = getAttributes(validationSet)
print("Number of validation instances = ", nrow)
print("Number of validation attributes = ", ncol - 1)
print("Accuracy of the model on the validation dataset before pruning = {0:.02f}%".format(
    accuracy(root, validationHeader, validationSet) * 100))
print("")

ncol, nrow = getAttributes(testSet)
print("Number of testing instances = ", nrow)
print("Number of testing attributes = ", ncol - 1)
print("Accuracy of the model on the testing dataset before pruning = {0:.02f}%".format(
    accuracy(root, testHeader, testSet) * 100))
print("")


# Apply random algorithm and build tree
root = RandombuildTree(trainSet, header, 0, None, None)
# print(node.nodeData)
# print("")
# print("Tree Representation")
# print("----------------------------------")
# printTree(root)
# print("")
#AvergDepth = AvgDepth(root,leafsum=0,sumdepth=0)
AveDepth = float(totDepth(root)) / totLeaf
print("Random average depth = {0:.02f}".format(AveDepth))

# print summary of the tree
totNode, totLeaf = traverse(root)
###################### Pre prunned ######################
print("Random Accuracy")
print("----------------------------------")
print("Number of training instances = ", nrow)
print("Number of training attributes = ", ncol - 1)
print("Total number of nodes in the tree = ", totNode)
print("Total number of leaf nodes in the tree = ", totLeaf)
print("Accuracy of the model on the training dataset = {0:.02f}%".format(accuracy(root, header, trainSet) * 100))
print("")

ncol, nrow = getAttributes(validationSet)
print("Number of validation instances = ", nrow)
print("Number of validation attributes = ", ncol - 1)
print("Accuracy of the model on the validation dataset before pruning = {0:.02f}%".format(
    accuracy(root, validationHeader, validationSet) * 100))
print("")

ncol, nrow = getAttributes(testSet)
print("Number of testing instances = ", nrow)
print("Number of testing attributes = ", ncol - 1)
print("Accuracy of the model on the testing dataset before pruning = {0:.02f}%".format(
    accuracy(root, testHeader, testSet) * 100))
print("")