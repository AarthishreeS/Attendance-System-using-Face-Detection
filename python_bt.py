import csv
import time
import random

COUNT = [10]

# Define a class for the binary tree nodes
class newNode:
    def __init__(self, row):
        self.data = row
        self.left = self.right = None

# Function to insert nodes in level order
def insertLevelOrder(arr, i, n):
    root = None
    # Base case for recursion
    if i < n:
        # Create a new node for the current element
        root = newNode(arr[i])

        # Recursively insert the left child
        root.left = insertLevelOrder(arr, 2 * i + 1, n)

        # Recursively insert the right child
        root.right = insertLevelOrder(arr, 2 * i + 2, n)

    return root

# Helper function to print the binary tree in 2D
def print2DUtil(root, space):
    # Base case
    if (root == None):
        return

    # Increase distance between levels
    space += COUNT[0]

    # Process right child first
    print2DUtil(root.right, space)

    # Print current node after space count
    print()
    for i in range(COUNT[0], space):
        print(end="                                   ")
    print(root.data)

    # Process left child
    print2DUtil(root.left, space)

def backtrack_search(root, target):
    if root is None or root.data[0] == str(target):
        return root

    if target < int(root.data[0]):
        return backtrack_search(root.left, target)
    else:
        return backtrack_search(root.right, target)


def dfs(node, target):
    if node is None:
        return False
    if node.data[0] == str(target):
        return True
    left = dfs(node.left, target)
    if left:
        return True
    right = dfs(node.right, target)
    if right:
        return True
    return False
    

def bfs(root, target):
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.data[0] == str(target):
            return node
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)
    return None


# Wrapper function for print2DUtil
def print2D(root):
    # Pass initial space count as 0
    print2DUtil(root, 0)


# Function to print tree nodes in InOrder fashion
def inOrder(root):
    if root != None:
        inOrder(root.left)
        #print(root.data,end="            ")
        inOrder(root.right)


# Greedy Best Worst Search function
def GBWSearch(root, target):
    best_node = None
    worst_dist = float('inf')
    pq = [(root, 0)]
    while pq:
        node, dist = pq.pop()
        if int(node.data[0]) == target:
            return node
        if abs(int(node.data[0] )- target) < worst_dist:
            best_node = node
            worst_dist = abs(int(node.data[0]) - target)
        if node.left:
            pq.append((node.left, dist+1))
        if node.right:
            pq.append((node.right, dist+1))
        pq.sort(key=lambda x: abs(int(x[0].data[0] )- target) - x[1], reverse=True)
    return best_node


def hierarchical_search(root, value):
    current_node = root
    while current_node is not None:
        if int(current_node.data[0]) == value:
            return current_node
        elif value < int(current_node.data[0]):
            current_node = current_node.left
        else:
            current_node = current_node.right
    return False


def iterative_deepening_search(root, target):
    depth_limit = 0
    while True:
        result = depth_limited_search(root, target, depth_limit)
        if result is not None:
            return result
        depth_limit += 1


def depth_limited_search(node, target, depth_limit):
    if node is None:
        return None
    if node.data[0] == str(target):
        return node
    if depth_limit == 0:
        return None
    left_result = depth_limited_search(node.left, target, depth_limit - 1)
    if left_result is not None:
        return left_result
    right_result = depth_limited_search(node.right, target, depth_limit - 1)
    if right_result is not None:
        return right_result
    return None

# Exponential search function for binary search tree
def exponential_search(root, target):
    if root is None:
        return None

    # Check if the root node is the target node
    if int(root.data[0]) == target:
        return root

    # Determine the range for exponential search
    i = 1
    while root.left is not None and int(root.left.data[0]) <= target:
        root = root.left
        i *= 2

    # Perform binary search on the determined range
    return binary_search(root, target, i // 2)

def create_random_tree(min_val, max_val, size):
    # Helper function to create a random binary search tree
    def insert_node(root, val):
        if not root:
            return val
        elif val < root.data[0]:
            root.left = insert_node(root.left, val)
        else:
            root.right = insert_node(root.right, val)
        return root

    tree = None
    for _ in range(size):
        val = random.randint(min_val, max_val)
        tree = insert_node(tree, val)
    return tree


def calculate_tree_height(root):
    if not root:
        return -1
    return 1 + max(calculate_tree_height(root.left), calculate_tree_height(root.right))


def crossover(parent1, parent2):
    # Helper function to perform crossover between two trees
    def copy_node(node):
        if not node:
            return None
        new_node = Node(node.val)
        new_node.left = copy_node(node.left)
        new_node.right = copy_node(node.right)
        return new_node

    # Choose a random node from each parent
    nodes1 = get_all_nodes(parent1)
    nodes2 = get_all_nodes(parent2)
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)

    # Swap the subtrees
    temp = copy_node(node1.left)
    node1.left = copy_node(node2.left)
    node2.left = temp

    temp = copy_node(node1.right)
    node1.right = copy_node(node2.right)
    node2.right = temp

    return parent1, parent2


def mutate(tree, mutation_rate):
    # Helper function to perform mutation on a tree
    def mutate_node(node):
        if not node:
            return None
        if random.random() < mutation_rate:
            node.left, node.right = node.right, node.left
        mutate_node(node.left)
        mutate_node(node.right)
        return node

    return mutate_node(tree)


def get_all_nodes(root):
    # Helper function to get a list of all nodes in a tree
    if not root:
        return []
    return [root] + get_all_nodes(root.left) + get_all_nodes(root.right)


def get_node_with_value(root, target_value):
    # Helper function to get the node with a target value in a binary search tree
    if not root:
        return None
    if root.val == target_value:
        return root
    elif root.val < target_value:
        return get_node_with_value(root.right, target_value)
    else:
        return get_node_with_value(root.left, target_value)


def morris_search(root, val):
    curr = root
    while curr:
        if int(curr.data[0]) == val:
            return curr
        elif int(curr.data[0]) < val:
            curr = curr.right
        else:
            # Find the predecessor
            pred = curr.left
            while pred.right and pred.right != curr:
                pred = pred.right

            if pred.right is None:
                # Set the right child of predecessor to current
                pred.right = curr
                curr = curr.left
            else:
                # Restore the tree
                pred.right = None
                curr = curr.right
    return None


# Binary search function for binary search tree
def binary_search(root, target, size):
    left, right = 0, size
    while left <= right:
        mid = left + (right - left) // 2
        if int(get_node(root, mid).data[0]) == target:
            return get_node(root, mid)
        elif int(get_node(root, mid).data[0]) < target:
            left = mid + 1
        else:
            right = mid - 1
    return None


# Helper function to get the node at a particular index
def get_node(root, index):
    if root is None:
        return None
    if index == 0:
        return root
    if index % 2 == 0:
        return get_node(root.left, (index // 2) - 1)
    else:
        return get_node(root.right, index // 2)

# Main program
if __name__ == '__main__':
    # Create an empty list to store the data from the CSV file
    store_row = []

    # Open the CSV file
    with open(r'Attendance\Attendance_24-04-2023.csv') as file_obj:
        # Create a CSV reader object
        reader_obj = csv.reader(file_obj)

        # Iterate over each row in the CSV file
        rown = 0
        for row in reader_obj:
            if rown == 0:
                rown = rown + 1
                continue
            store_row.append(row)
            rown = rown + 1

        #print(store_row)

        # Get the length of the list
        n = len(store_row)

        # Insert each row into the binary tree
        root = None
        root = insertLevelOrder(store_row, 0, n)
        inOrder(root)
        
        print2D(root)

        start_time = time.time()
        target_node = 2438
        result = dfs(root, target_node)
        end_time = time.time()
      # print(f"\n\n\nTarget node {target_node} was found using DFS - Depth first search : {result}\n")
        print(f"\n\nTime taken using DFS - Depth first search : {end_time - start_time} seconds\n")

        target_node = 2438
        start_time = time.time()
        found = bfs(root, target_node)
        end_time = time.time()
       # print(f"Target node {target_node} was found using DFS - Depth first search : {found}\n")
        print(f"Time taken using BFS - Breadth first search : {end_time - start_time} seconds\n")

        start_time = time.time()
        result = backtrack_search(root, target_node)
        end_time = time.time()
        found = result.data[0]
     #   print(f"Target node {target_node} was found using backtrack_search: {found}\n")
        print(f"Time taken using backtrack_search : {end_time - start_time} seconds\n")

        start_time = time.time()
        result = GBWSearch(root, target_node)
        end_time = time.time()
       # print(f"Target node {target_node} was found using GBW Search: {result}\n")
        print(f"Time taken using GBW Search : {end_time - start_time} seconds\n")

        start_time = time.time()
        result = hierarchical_search(root, target_node )
        end_time = time.time()
      #  print(f"Target node {target_node} was found using hierarchical search: {result}\n")
        print(f"Time taken using hierarchical search : {end_time - start_time} seconds\n")

        start_time = time.time()
        result = iterative_deepening_search(root,target_node )
        end_time = time.time()
      #  print(f"Target node {target_node} was found using iterative deepening search: {result}\n")
        print(f"Time taken using iterative deepening search : {end_time - start_time} seconds\n")


        start_time = time.time()
        result = exponential_search(root, target_node)
        end_time = time.time()
       # print(f"Target node {target_node} was found using exponential search: {result}\n")
        print(f"Time taken using exponential search : {end_time - start_time} seconds\n")


