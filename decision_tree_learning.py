from __future__ import division
from math import log
import sys


class Node(object):
    """
    Nodes of a DecisionTree. Each node has a condition,
    false_child and true_child.

    If is_true or is_false is True, then no consistency is
    guaranteed for any other attributes of the node.

    Each node keeps track of the max_depth and its current depth
    in the tree. If a max_depth is given, the node will stop generating
    children if it is at the max_depth and become a leaf with representing
    the value of the largest representative portion of elements
    in its training set.
    """

    # Attributes
    j = 0  # used only for __str__
    attribute = 0  # used only for __str__
    true_child = None
    false_child = None

    def __init__(self, max_depth, training_set=[], attributes=[], curr_depth=1,
                 is_true=False, is_false=False):
        self.training_set = training_set
        self.attributes = attributes
        self.is_true = is_true
        self.is_false = is_false
        self.curr_depth = curr_depth
        self.max_depth = max_depth
        if not (is_true or is_false):
            self.early_stop_id3()

    def __str__(self):
        if self.is_false:
            return "False"
        elif self.is_true:
            return "True"
        return "Attribute[%i] <= %i, (%s, %s)" % (self.attribute, self.j,
                                                  str(self.true_child),
                                                  str(self.false_child))

    def entropy(self, s):
        """
        Finds the entropy of a set
        """
        size = len(s)
        pos_p = 0
        neg_p = 0
        for ex in s:
            if ex['label'] == 'M':
                pos_p += 1
            else:
                neg_p += 1
        size = len(s)
        pos_entropy = -pos_p/size * log(pos_p/size, 2) if pos_p > 0 else 0
        neg_entropy = -neg_p/size * log(neg_p/size, 2) if neg_p > 0 else 0
        return pos_entropy + neg_entropy

    def find_attr_entropy(self, attribute, j):
        """
        Finds the entropy of an attribute for a given j value
        """
        training_size = len(self.training_set)
        less_eql = []
        greater_than = []
        for example in self.training_set:
            if (example[attribute] <= j):
                less_eql.append(example)
            else:
                greater_than.append(example)
        less_entropy = self.entropy(less_eql)
        greater_entropy = self.entropy(greater_than)
        result = (len(less_eql)/training_size) * less_entropy
        result += (len(greater_than)/training_size) * greater_entropy
        return result

    def add_node_or_make_leaf(self, split_node, true_examples, false_examples):
        """
        Adds the correct child nodes to the current node, or
        turns the current node into a leaf if there are no more
        decisions to be made.
        """
        if len(true_examples) == 0:
            self.is_true = false_examples[0]['label'] == 'M'
            self.is_false = false_examples[0]['label'] == 'B'
        elif len(false_examples) == 0:
            self.is_true = true_examples[0]['label'] == 'M'
            self.is_false = true_examples[0]['label'] == 'B'
        elif split_node[2] == 0.0:
            true_label = true_examples[0]['label']
            false_label = false_examples[0]['label']
            if true_label == false_label:
                self.is_true = true_label == 'M'
                self.is_false = true_label == 'B'
            else:
                self.true_child = Node(self.max_depth,
                                       is_true=true_label == 'M',
                                       is_false=true_label == 'B')
                self.false_child = Node(self.max_depth,
                                        is_true=false_label == 'M',
                                        is_false=false_label == 'B')
                self.j = split_node[1]
                self.attribute = split_node[0]
        else:
            self.true_child = Node(self.max_depth, true_examples,
                                   self.attributes, self.curr_depth + 1)
            self.false_child = Node(self.max_depth, false_examples,
                                    self.attributes, self.curr_depth + 1)
            self.j = split_node[1]
            self.attribute = split_node[0]

    def early_stop_id3(self):
        """
        If the current node is at max_depth, turns the node into a leaf
        Otherwise call the id3 method on this node
        """
        if self.curr_depth >= self.max_depth:
            true_total = 0
            false_total = 0
            for ex in self.training_set:
                if ex['label'] == 'M':
                    true_total += 1
                else:
                    false_total += 1
            self.is_true = true_total >= false_total
            self.is_false = true_total < false_total
        else:
            self.id3()

    def id3(self):
        """
        ID3 algorithm (TDIDT splitting on information gain)
        """
        # set_entropy = self.entropy(s)
        best_node = (0, 0, float('inf'))
        for attribute in self.attributes:
            best_j_val = (0, 0, float('inf'))
            for j in xrange(1, 10):
                attr_entropy = self.find_attr_entropy(attribute, j)
                if (attr_entropy < best_j_val[2]):
                    best_j_val = (attribute, j, attr_entropy)
            if (best_j_val[2] < best_node[2]):
                best_node = best_j_val
        self.condition = lambda ex: ex[best_node[0]] <= best_node[1]
        # print "best_node: " + str(best_node)
        true_examples = [e for e in self.training_set if self.condition(e)]
        # print "true_examples " + str(len(true_examples))
        false_examples = [e for e in self.training_set if not self.condition(e)]
        # print "false_examples " + str(len(false_examples))
        self.add_node_or_make_leaf(best_node, true_examples, false_examples)

    def test(self, example):
        """
        Test the example against this node. If node is a leaf, return
        value of the leaf, otherwise pass the example on to the
        correct child according to the condition.
        """
        if self.is_false:
            return False
        elif self.is_true:
            return True
        else:
            if self.condition(example):
                res = self.true_child.test(example)
                return res
            else:
                res = self.false_child.test(example)
                return res

    def size(self):
        """
        Return the size of the subtree that this node is the root of
        """
        if self.is_true or self.is_false:
            return 1
        return 1 + self.true_child.size() + self.false_child.size()

    def num_leaves(self):
        """
        Return the number of leaves in the subtree that this node
        is the root of
        """
        if self.is_true or self.is_false:
            return 1
        return self.true_child.num_leaves() + self.false_child.num_leaves()


class DecisionTree(object):
    """
    Decision Tree Data structure
    Each node has a condition, false_child and true_child
    Children can either be nodes or leaves (boolean values)
    """

    def __init__(self, training_set, attributes, max_depth=float('inf')):
        self.training_set = training_set
        self.attributes = attributes
        self.root = Node(max_depth, training_set, attributes)

    def eval(self, example):
        """
        Find the value of the example according to the tree
        """
        return self.root.test(example)

    def size(self):
        return self.root.size()

    def num_leaves(self):
        return self.root.num_leaves()


if __name__ == "__main__":
    args = sys.argv[1:]
    train_file = open(args[0], 'r')

    # Put the examples in the training set into a list of dicts
    training_set = []
    for line in train_file:
        example = {}
        vals = line.split(' ')[:-2]
        example['label'] = vals[0]
        for i in vals[1:]:
            attr = int(i[:1])
            example[attr] = int(i[2:])
        training_set.append(example)

    # Create a DecisionTree. Limit it's depth if a max_depth is provided
    if len(args) > 2 and args[2] != 'inf':
        trained_tree = DecisionTree(training_set, list(xrange(1, 10)),
                                    max_depth=int(args[2]))
    else:
        trained_tree = DecisionTree(training_set, list(xrange(1, 10)))

    print "\n\nDecision tree:\n%s" % str(trained_tree.root)
    print "\nSize = %i" % trained_tree.size()
    print "\nNumber of Leaves = %i" % trained_tree.num_leaves()

    train_file.close()
    test_file = open(args[1], 'r')

    test_size = 0
    errors = 0

    # Test each example in the testing set, keeping track of the errors
    for line in test_file:
        example = {}
        vals = line.split(' ')[:-2]
        example['label'] = vals[0]
        for i in vals[1:]:
            attr = int(i[:1])
            example[attr] = int(i[2:])
        malignant = trained_tree.eval(example)
        label = example['label']
        if malignant and label == 'B' or not malignant and label == 'M':
            errors += 1
        test_size += 1

    test_file.close()
    print "\n\nAccuracy:\n%s" % str(1 - errors/test_size)
