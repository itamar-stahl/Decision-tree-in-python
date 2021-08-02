from collections import Counter
from itertools import combinations


class Node:
    def __init__(self, data, pos=None, neg=None):
        self.data = data
        self.positive_child = pos  # aka right child
        self.negative_child = neg  # aka left child

    def is_leaf(self):
        return self.positive_child is None and self.negative_child is None


class Record:
    """
    Holds a representation of an illness (string) and its symptoms (list of strings)
    """

    def __init__(self, illness, symptoms):
        self.illness = illness
        self.symptoms = symptoms


class Queue:
    """
    An object which returns an increasing series of numbers (0, 1, ..., max):
    one element in each call
    """

    def __init__(self, max):
        self.curr = 0
        self.max = max

    def get_curr(self):
        curr = self.curr
        self.curr += 1
        if self.curr == (self.max):
            self.curr = 0
        return curr


class Diagnoser:
    """
    Main class. Responsible for diagnosing illnesses by their symptoms,
    by using a decision tree.
    """

    def __init__(self, root):
        self.root = root
        self.curr_node = root
        self.paths_to_ill_lst = []

    def diagnose(self, symptoms):
        """
        Receives a list of symptoms, and returns a diagnosis.
        """
        if self.curr_node.is_leaf():  # base case
            diagnosis = self.curr_node.data
            self.curr_node = self.root  # reset
            return diagnosis

        if self.curr_node.data in symptoms:
            self.curr_node = self.curr_node.positive_child
        else:
            self.curr_node = self.curr_node.negative_child
        return self.diagnose(symptoms)

    def calculate_success_rate(self, records):
        """
        Gets labeled records data, and estimates the success rate of the tree
        """
        records_num = len(records)
        diagnosed_records_num = 0
        for record in records:
            if self.diagnose(record.symptoms) == record.illness:
                diagnosed_records_num += 1
        return (diagnosed_records_num / records_num)

    def all_illnesses(self):
        """
        Returns a list of all the leaves (illnesses) in the
        the tree, most frequent first.
        """
        all_leaves = self.all_illnesses_helper()
        self.curr_node = self.root
        return [i[0] for i in Counter(all_leaves).most_common()]

    def all_illnesses_helper(self):
        """
        Recursive helper function for all_illnesses()
        """
        result = []
        if self.curr_node.is_leaf():
            result.append(self.curr_node.data)
        else:
            pos = self.curr_node.positive_child
            self.curr_node = self.curr_node.negative_child
            result.extend(self.all_illnesses_helper())
            self.curr_node = pos
            result.extend(self.all_illnesses_helper())
        return result

    def most_rare_illness(self, records):
        """
        Returns the rarest diagnosis in the tree for the given records
        """
        diagnoses_lst = {}

        for record in records:
            diagnosis = self.diagnose(record.symptoms)
            if diagnosis in diagnoses_lst:
                diagnoses_lst[diagnosis] += 1
            else:
                diagnoses_lst[diagnosis] = 1
        for ill in self.all_illnesses():
            if ill not in diagnoses_lst.keys():
                diagnoses_lst[ill] = 0

        most_rare = min(diagnoses_lst, key=diagnoses_lst.get)
        return most_rare

    def paths_to_illness(self, illness):
        """
        Returns a list of possible paths to the given illness
        in the tree. Path represented by booleans array, where True
        indicate going down to the positive child, and False to the left child. 
        """
        self.paths_to_illness_helper(self.root, illness, [])
        paths = self.paths_to_ill_lst
        self.paths_to_ill_lst = []
        return paths

    def paths_to_illness_helper(self, curr_node, illness, path):
        """"
        Helper function to paths_to_illness
        """
        if curr_node.is_leaf():  # base case
            if curr_node.data == illness:
                self.paths_to_ill_lst.append(path)
            return

        self.paths_to_illness_helper(
            curr_node.negative_child, illness, path + [False])
        self.paths_to_illness_helper(
            curr_node.positive_child, illness, path + [True])


def parse_data(filepath):
    """
    Reads the data (illnesses, symptoms) from a file,
    and returns a list of Record objects.
    """
    with open(filepath) as data_file:
        records = []
        for line in data_file:
            words = line.strip().split()
            records.append(Record(words[0], words[1:]))
        return records


def build_tree(records, symptoms):
    """"
    A decision tree builder.
    Gets:
    1) List of records
    2) List of symptoms(strings), where the i'th symptom is the query
    of the nodes in the i'th depth of the tree.
    """

    root = Node(symptoms[0], None, None)
    # installing a queue in length of the leaves' amount:
    queue = Queue(2**len(symptoms))
    # build the tree with the queue indexes as leaves
    build_tree_skeleton(symptoms, 0, queue, root)
    diagnoser = Diagnoser(root)

    leafs = [[leaf, [], {}] for leaf in diagnoser.all_illnesses()]
    # installing a list of [leaf_num, [path to it],
    #  {frequency list of diagnosis}] for:
    fit_ill_to_leaf(root, diagnoser, leafs, records)

    return root


def build_tree_skeleton(symptoms, depth, queue, root=None):
    """
    Builds a tree with a depth of len(symptoms),
    where the i'th symptom is the query
    of the nodes in the i'th depth of the tree.
    """
    if depth == len(symptoms):  # base case. Gives a unique name to each leaf:
        return Node(queue.get_curr(), None, None)
    else:
        if depth == 0:
            curr_node = root
        else:
            curr_node = Node(symptoms[depth], None, None)

        curr_node.negative_child = build_tree_skeleton(
            symptoms, depth + 1, queue)
        curr_node.positive_child = build_tree_skeleton(
            symptoms, depth + 1, queue)

        return curr_node


def fit_ill_to_leaf(root, diagnoser, leafs, records):
    """
    For each leaf in the tree, set a diagnosis according to the records:
    The most frequent diagnosis for the given symptoms path in the records.
    """
    for i in range(len(leafs)):
        leafs[i][1] = diagnoser.paths_to_illness(leafs[i][0])[0]
        # fill the path to each leaf in the tree

        for record in records:  # extracts statistics from the records:
            if record.illness not in leafs[i][2]:
                leafs[i][2][record.illness] = 0
            if diagnoser.diagnose(record.symptoms) == leafs[i][0]:
                leafs[i][2][record.illness] += 1

        most_freq = max(leafs[i][2], key=leafs[i][2].get)
        # set the most_freq diagnosis:
        set_ill_to_leaf(most_freq, leafs[i][1], root)


def set_ill_to_leaf(ill, path, curr_node, depth=0):
    """
    Set illness (ill) to leaf.data
    """
    if curr_node.is_leaf():	 # base case
        curr_node.data = ill
        return

    if path[depth]:
        set_ill_to_leaf(ill, path, curr_node.positive_child, depth + 1)
    else:
        set_ill_to_leaf(ill, path, curr_node.negative_child, depth + 1)


def optimal_tree(records, symptoms, depth):
    """
    Chooses the best decision tree (determined by the symptoms' order in the 
    first depth levels) for the given records.
    """
    possible_comb = [list(i) for i in list(combinations(symptoms, depth))]
    # all combination up to the given depth

    possible_trees = []
    optimal_tree = [None, 0.0]

    for i in range(len(possible_comb)):
        possible_trees.append(build_tree(records, possible_comb[i]))
        diagnoser = Diagnoser(possible_trees[i])
        tree_i_rate = diagnoser.calculate_success_rate(records)
        # in each round update the date if necessary:
        if tree_i_rate > optimal_tree[1]:
            optimal_tree = [possible_trees[i], tree_i_rate]

    return optimal_tree[0]
