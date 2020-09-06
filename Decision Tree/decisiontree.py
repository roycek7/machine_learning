"""
@author: s4625266
"""

import csv
import sys
from math import log2

label = -1  # decision label index
decision = 'yes'  # decision
max_depth = 999  # default max depth of decision tree
delimiter = '\t'  # data file separated by


class Tree:
    def __init__(self):
        self.parent = {}
        self.child = {}

    def __repr__(self):
        return f'{self.parent}'


def read_csv(file) -> tuple:
    """input file reader and returns rows and columns in list"""
    with open(file, newline='') as line:
        data_file = csv.reader(line, delimiter=delimiter)
        data = list(data_file)
        return data[1:], data[:1][0]
    
    
def get_leaf_decision(l_yes, l_no, parent_decision) -> int:
    """returns leaf decision with majority decision if not equal else returns it's parent decision"""
    leaf_value = 0
    if l_yes > l_no:
        leaf_value = 1
    elif l_yes == l_no:
        leaf_value = 1 if parent_decision else 0
    return leaf_value


def calculate_entropy_info(label_yes, label_no, parent_decision, info=False, attribute_dict=None, node=None,
                           data=None, info_gain=None) -> tuple:
    """takes the decision label's and return's information gain / entropy depending on callback parameter"""
    entropy = 0.0
    if label_yes != 0 and label_no != 0:
        entropy = - label_yes * log2(label_yes) - label_no * log2(label_no)
        if info:
            info_gain -= (attribute_dict[node] * entropy) / len(data)
    return info_gain if info else entropy, get_leaf_decision(label_yes, label_no, parent_decision)


def get_label_value(data, rows, columns, node=None) -> tuple:
    """returns tuple with count of total number of decisions"""
    yes, no = 0, 0
    for row in rows:
        if not node:
            if data[row][columns] == decision:
                yes += 1
            else:
                no += 1
        else:
            if data[row][columns] == node:
                if data[row][label] == decision:
                    yes += 1
                else:
                    no += 1
    return yes, no


def get_entropy(data, rows, headers, parent_decision) -> tuple:
    """returns calculated entropy and leaf value"""
    yes, no = get_label_value(data, rows, len(headers))
    entropy, leaf_value = calculate_entropy_info(yes / (yes + no), no / (yes + no), parent_decision)
    return entropy, leaf_value


def get_attribute(data, rows, column, attribute_structure, return_list=False) -> dict or list:
    """returns list of unique attributes of columns if flagged true else returns dictionary of the column attributes
    and the number of decision occurrence"""
    for row in rows:
        node = data[row][column]
        if return_list and node not in attribute_structure:
            attribute_structure.append(node)
        if not return_list:
            if node not in attribute_structure:
                attribute_structure[node] = 1
            else:
                attribute_structure[node] += 1
    return attribute_structure


def get_information_gain(data, rows, columns, headers, parent_decision) -> tuple:
    """returns the maximum information gained by column, its index and leaf value"""
    information_gain = 0.0
    index = -999
    entropy, leaf_value = get_entropy(data, rows, headers, parent_decision)
    if entropy == 0:
        return information_gain, index, leaf_value

    for column in columns:
        info = entropy
        attribute_dict = get_attribute(data, rows, column, attribute_structure={})
        for node in attribute_dict:
            yes, no = get_label_value(data, rows, column, node)
            info = calculate_entropy_info(yes / (yes + no), no / (yes + no), parent_decision, info=True,
                                          attribute_dict=attribute_dict, node=node, data=data,
                                          info_gain=info)[0]
        if info >= information_gain:
            information_gain, index = info, column
    return information_gain, index, leaf_value


def input_decision(root, leaf_value):
    """inserts leaf value as either true or false"""
    root.parent = True if leaf_value == 1 else False


def get_major(parent_decision) -> bool:
    """returns the major parent decision in boolean form"""
    return True if parent_decision[0] > parent_decision[1] else False


def decision_tree_builder(data, rows, columns, headers, maximum_depth, parent_decision, depth=0):
    """builds decision tree in recursive manner by splitting the dataset and with respect to maximum depth based on
    information gain and terminating when information gain is 0 or when we have reached the maximum depth"""
    information_gain, index, leaf_value = get_information_gain(data, rows, columns, headers, parent_decision)

    dt_id3 = Tree()
    if information_gain == 0 or depth == maximum_depth:
        input_decision(dt_id3, leaf_value)
        return dt_id3

    new_dataset_header = columns.copy()
    new_dataset_header.remove(index)
    attr_list = get_attribute(data, rows, index, attribute_structure=[], return_list=True)
    for node in attr_list:
        major_parent = get_major(get_label_value(data, rows, len(headers)))
        new_dataset_split = []
        for row in rows:
            if data[row][index] == node:
                new_dataset_split.append(row)
        tree_builder = decision_tree_builder(data, new_dataset_split, new_dataset_header, headers,
                                             maximum_depth, major_parent, depth + 1)
        dt_id3.parent[headers[index]] = dt_id3.child
        dt_id3.child[node] = tree_builder
    return dt_id3


def get_decision(input_data, leaf_answer):
    """returns decision if boolean decision found else calls function recursively to classify until answer is found"""
    return classify_data(input_data, leaf_answer) if type(leaf_answer.parent) != bool else leaf_answer.parent


def get_leaf(decision_tree, key, input_data):
    """returns internal nodes from decision tree"""
    return decision_tree.parent[key][input_data[key]]


def classify_data(input_data, decision_tree):
    """returns decision to classify input data based on the decision tree"""
    answer = None
    for key in list(input_data.keys()):
        try:
            if key in list(decision_tree.keys()):
                try:
                    answer = decision_tree[key][input_data[key]]
                except AttributeError:
                    answer = get_leaf(decision_tree, key, input_data)
                finally:
                    return get_decision(input_data, answer)
        except AttributeError:
            if key in decision_tree.parent:
                answer = get_leaf(decision_tree, key, input_data)
                return get_decision(input_data, answer)


def accuracy_metric(actual, predicted, correct=0) -> float:
    """calculate accuracy of the dataset comparing the actual decision from the dataset vs the predicted
    from the decision tree"""
    for i in range(len(actual)):
        actual[i] = True if actual[i] == decision else False
        if actual[i] == predicted[i]:
            correct += 1
    return round(correct / len(actual) * 100.0, 2)


def classify_dataset(data, column, dt_id3) -> tuple:
    """classifier accepts training and test dataset, puts the actual decisions in a list and calls the classify_data
    to get predicted decisions"""
    predicted, actual_decision = [], []
    for row in data:
        actual_decision.append(row[label])
        predicted.append(classify_data(dict(zip(column[:label], row)), dt_id3.parent))
    return actual_decision, predicted


def id3(argv, input_depth):
    """main id3 function calls the decision tree builder and prints, also calls other functions to classify
    training and test dataset"""
    try:
        data, column = read_csv(argv[:1][0])
        if len(argv) > 1:
            try:
                if isinstance(int(argv[1]), int):
                    input_depth = int(argv[1])
                    print(f'Max Depth of Decision Tree: {argv[1]}')
                    if input_depth == 0:
                        print(f'\nMax Depth: {input_depth} -> NOT ALLOWED!!!\ni.e max_depth > 0')
                        raise Exception
            except ValueError:
                pass
        decision_tree_id3 = decision_tree_builder(data, [i for i in range(0, len(data))],
                                                  [i for i in range(0, len(column) - 1)],
                                                  column[:label], input_depth, parent_decision=False)
        print('-------------------------------------------------')
        print(f'Decision Tree: --> {decision_tree_id3.__repr__()}')
        print('-------------------------------------------------')
        actual_decision, predicted = classify_dataset(data, column, decision_tree_id3)
        accuracy_training = accuracy_metric(actual_decision, predicted)

        print(f'Training Accuracy on {argv[:1][0]}: {accuracy_training}')

        if len(argv) > 2:
            test_data, test_column = read_csv(argv[label])
            actual_decision, predicted = classify_dataset(test_data, test_column, decision_tree_id3)
            accuracy_test = accuracy_metric(actual_decision, predicted)
            print(f'Test Accuracy on {argv[label]}: {accuracy_test}')
    except Exception as e:
        print(f'{str(e).upper()}\nFollow command:\npython decisiontreeassignment.py <training file> --mandatory '
              f'<maximum depth of tree> --optional <test file> --optional')


if __name__ == '__main__':
    id3(sys.argv[1:], max_depth)
