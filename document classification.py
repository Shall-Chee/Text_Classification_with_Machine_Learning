
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# This cell has the code to load the datasets. You should not need
# to edit this cell unless you want to do the extra credit. If you do
# you should only need to edit normalize_images.
# class CIFARDataset(Dataset):
#   def __init__(self, X, y):
#     self.len = len(X)
#     self.X = torch.FloatTensor(X).cuda()
#     self.y = torch.LongTensor(y).cuda()
  
#   def __len__(self):
#     return self.len

#   def __getitem__(self, idx):
#     return self.X[idx], self.y[idx]


# def normalize_images(X_train, X_valid, X_test):
#   """
#   Normalizes the images based on the means and standard deviations
#   of the training channels. Returns the new normalized images.
#   """
#   # TODO Implement this method for the extra credit experiments
#   # raise NotImplementedError()
#   for i in range(3):
#     mu = np.mean(X_train[:, i])
#     std = np.std(X_train[:, i])
#     X_train[:, i] = (X_train[:, i] - mu) / std
#     X_valid[:, i] = (X_valid[:, i] - mu) / std
#     X_test[:, i] = (X_test[:, i] - mu) / std
  
#   return X_train, X_valid, X_test

# def load_datasets(normalize=False):
#   X_train = np.load('train_images.npy').astype(float)
#   y_train = np.load('train_labels.npy')
#   X_valid = np.load('valid_images.npy').astype(float)
#   y_valid = np.load('valid_labels.npy')
#   X_test = np.load('test_images.npy').astype(float)
#   y_test = np.load('test_labels.npy')

#   if normalize:
#     X_train, X_valid, X_test = normalize_images(X_train, X_valid, X_test)
  
#   train_data = CIFARDataset(X_train, y_train)
#   valid_data = CIFARDataset(X_valid, y_valid)
#   test_data = CIFARDataset(X_test, y_test)
  
#   return train_data, valid_data, test_data

# This is the implementation of the first network architecture. We have
# started it, but you need to finish it. Do not change the class name
# or the name of the data members "fc1" or "fc2"

class FeedForward(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = torch.nn.Linear(3072, 1000)
    # TODO
    # You need to add the second layer's parametersb
    # self.fc2 = None
    self.fc2 = torch.nn.Linear(1000, 10)

  def forward(self, X):
    batch_size = X.size(0)
    # This next line reshapes the tensor to be size (B x 3072)
    # so it can be passed through a linear layer.
    X = X.view(batch_size, -1)
    # TODO
    # You need to pass X through the two linear layers and relu
    # then return the final scores
    # raise NotImplementedError()

    X = self.fc1(X)
    X = F.relu(X)
    X = self.fc2(X)
    # X = torch.sigmoid(X)
    return X

# This is the implementation of the second network architecture. We have
# started it, but you need to finish it. Do not change the class name
# or the name of the data members "conv1", "pool", "conv2", "fc1", "fc2",
# or "fc3".
class Convolutional(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels=3,
                                 out_channels=7,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)
    # TODO
    # You need to add the pooling, second convolution, and
    # three linear modules here
    # self.pool = None
    # self.conv2 = None
    # self.fc1 = None
    # self.fc2 = None
    # self.fc3 = None
    self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
    self.conv2 = torch.nn.Conv2d(in_channels=7,
                                 out_channels=16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=0)
    self.fc1 = torch.nn.Linear(2704, 130)
    self.fc2 = torch.nn.Linear(130, 72)
    self.fc3 = torch.nn.Linear(72, 10)

  def forward(self, X):
    batch_size = X.size(0)
    # TODO
    # You need to implement the full network architecture here
    # and return the final scores
    # raise NotImplementedError()

    X = self.conv1(X)
    X = self.pool(X)
    X = self.conv2(X)
    X = F.relu(X)
    X = X.view(batch_size, -1)
    X = self.fc1(X)
    X = F.relu(X)
    X = self.fc2(X)
    X = F.relu(X)
    X = self.fc3(X)
    X = torch.sigmoid(X)

    return X

# You need to finish implementing this method
# def compute_loss_and_accuracy(network, data_loader):
#   """
#   Given a network, iterate over the dataset defined by the data_loader
#   and compute the accuracy of the model and the average loss.
#   """
#   # This should be used to accumulate the total loss on the dataset
#   total_loss = 0

#   # This should count how many examples were correctly classified.
#   num_correct = 0

#   # This should count the number of examples in the dataset. (Be careful
#   # because it should -not- be the number of batches.)
#   num_instances = 0

#   # The CrossEntropyLoss by default will return the average loss
#   # for the batch. So, when you accumulate the total_loss, make sure
#   # to multiply the loss computed by CrossEntropyLoss by the batch size
#   cross_entropy_loss = torch.nn.CrossEntropyLoss()

#   for X, y in data_loader:
#     # TODO
#     # You need to implement computing the loss and
#     # calculate the number of correct examples.
#     # raise NotImplementedError()

#     output = network(X)
#     loss = cross_entropy_loss(output, y)
#     size = np.asarray(y.size())
#     total_loss += loss.item() * size
#     num_instances += size
#     predicted = torch.max(output.data, 1)[1]
#     num_correct += (predicted ==y).sum().item() 

#   accuracy = num_correct / num_instances * 100
#   average_loss = total_loss / num_instances
#   return accuracy, average_loss

# # You need to finish implementing this method
# def run_experiment(network, train_data_loader, valid_data_loader, optimizer):
#   # This will be a list of the average training losses for each epoch
#   train_losses = []

#   # This will be a list of the average validation losses for each epoch
#   valid_accs = []

#   # This will be a list of the validation accuracies for each epoch
#   valid_losses = []

#   # The CrossEntropyLoss by default will return the average loss
#   # for the batch. So, when you accumulate the total_loss, make sure
#   # to multiply the loss computed by CrossEntropyLoss by the batch size
#   cross_entropy_loss = torch.nn.CrossEntropyLoss()

#   for epoch in range(200):
#     # This should be used to accumulate the total loss on the training data
#     total_loss = 0.0

#     # This should be used to count the number of training examples. (Be careful
#     # because this is not the number of batches)
#     num_instances = 0

#     for X, y in train_data_loader:
#       # TODO
#       # You need to implement computing the loss for this batch
#       # and updating the model's parameters.
#       # raise NotImplementedError()

#       optimizer.zero_grad()
#       output = network(X)
#       loss = cross_entropy_loss(output, y)
#       loss.backward()
#       optimizer.step()
#       size = np.asarray(y.size())
#       total_loss += loss.item() * size
#       num_instances += size

#     train_loss = total_loss / num_instances
#     valid_acc, valid_loss = compute_loss_and_accuracy(network, valid_data_loader)

#     train_losses.append(train_loss)
#     valid_accs.append(valid_acc)
#     valid_losses.append(valid_loss)
#   return train_losses, valid_accs, valid_losses

# # Load the data and create the iterators. You should not need
# # to modify this cell
# train_data, valid_data, test_data = load_datasets(normalize=False)
# train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# valid_data_loader = DataLoader(valid_data, batch_size=64)
# test_data_loader = DataLoader(test_data, batch_size=64)

# # Implements the FeedForward experiment. You can base the Convolutional experiment
# # on this code. You should not need to edit this cell.
# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.0001, 0.00005, 0.00001]:
# # for lr in [0.0001]:
#   network = FeedForward()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

# # TODO
# # You should implement the Convolutional experiment here. It should be
# # very similar to the cell above.

# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.01, 0.001, 0.0001]:
# # for lr in [0.0001]:
#   network = Convolutional()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

# # TODO
# # You should implement the Convolutional experiment here. It should be
# # very similar to the cell above.

# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.01, 0.001, 0.0001]:
# # for lr in [0.0001]:
#   network = Convolutional()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

# """## (Optional) Extra Credit"""

# # TODO
# # If you want to run the extra credit experiment, repeat the above experiments
# # but load the normalized data.
# train_data, valid_data, test_data = load_datasets(normalize=True)
# train_data_loader = DataLoader(train_data, batch_size=64, shuffle=True)
# valid_data_loader = DataLoader(valid_data, batch_size=64)
# test_data_loader = DataLoader(test_data, batch_size=64)

# # Implements the FeedForward experiment. You can base the Convolutional experiment
# # on this code. You should not need to edit this cell.
# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.0001, 0.00005, 0.00001]:
# # for lr in [0.0001]:
#   network = FeedForward()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

# # Implements the FeedForward experiment. You can base the Convolutional experiment
# # on this code. You should not need to edit this cell.
# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.001, 0.01, 0.1]:
# # for lr in [0.0001]:
#   network = FeedForward()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.01, 0.001, 0.0001]:
# # for lr in [0.0001]:
#   network = Convolutional()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

# best_network = None
# best_acc = None

# _, axs = plt.subplots(1,3)
# axs[0].set_title('Training Loss')
# axs[1].set_title('Validation Loss')
# axs[2].set_title('Validation Accuracies')

# for lr in [0.05, 0.1, 0.25]:
# # for lr in [0.0001]:
#   network = Convolutional()
#   network.cuda()
#   sgd = torch.optim.SGD(network.parameters(), lr=lr)

#   train_losses, valid_accs, valid_losses = run_experiment(network, train_data_loader, valid_data_loader, sgd)
#   valid_acc = valid_accs[-1]
#   print(f'LR = {lr}, Valid Acc: {valid_acc}')
#   if best_acc is None or valid_acc > best_acc:
#     best_acc = valid_acc
#     best_network = network
  

#   axs[0].plot(train_losses, label=str(lr))
#   axs[1].plot(valid_losses, label=str(lr))
#   axs[2].plot(valid_accs, label=str(lr))

# plt.legend()

# test_acc, _ = compute_loss_and_accuracy(best_network, test_data_loader)
# print('Test Accuracy: ' + str(test_acc))

"""
# 2 Document Classification [40 Points]

##2.2 Document Representation
"""

import json
from sklearn.metrics import accuracy_score
import numpy as np

def get_vocabulary(D):
    """
    Given a list of documents, where each document is represented as
    a list of tokens, return the resulting vocabulary. The vocabulary
    should be a set of tokens which appear more than once in the entire
    document collection plus the "<unk>" token.
    """
    # TODO
    # raise NotImplementedError

    vocabulary = set()
    count = set()
    for document in D:
      for word in document:
        if word in count:
          if word in vocabulary:
            continue
          vocabulary.add(word)
        else:
          count.add(word)
    vocabulary.add('<unk>')
    return vocabulary

class BBoWFeaturizer(object):
  def convert_document_to_feature_dictionary(self, doc, vocab):
    """
    Given a document represented as a list of tokens and the vocabulary
    as a set of tokens, compute the binary bag-of-words feature representation.
    This function should return a dictionary which maps from the name of the
    feature to the value of that feature.
    """
    feat_dict = {}
    for token in doc:
      if token in vocab:
        feat_dict[token] = 1
      else:
        feat_dict["<unk>"] = 1
    return feat_dict

class CBoWFeaturizer(object):
  def convert_document_to_feature_dictionary(self, doc, vocab):
    """
    Given a document represented as a list of tokens and the vocabulary
    as a set of tokens, compute the count bag-of-words feature representation.
    This function should return a dictionary which maps from the name of the
    feature to the value of that feature.
    """
    feat_dict = {}
    for token in doc:
      if token in vocab:
        if token in feat_dict:
          feat_dict[token] += 1
        else:
          feat_dict[token] = 1
      else:
        if "<unk>" in feat_dict:
          feat_dict["<unk>"] += 1
        else:
          feat_dict["<unk>"] = 1
    return feat_dict

def compute_idf(D, vocab):
    """
    Given a list of documents D and the vocabulary as a set of tokens,
    where each document is represented as a list of tokens, return the IDF scores
    for every token in the vocab. The IDFs should be represented as a dictionary that
    maps from the token to the IDF value. If a token is not present in the
    vocab, it should be mapped to "<unk>".
    """
    # TODO
    # raise NotImplementedError
    IDF = {}
    dictionary = {}
    for document in D:
      document = set(document)
      flag = 0
      for word in document:
        if word in vocab:
          dictionary[word] = dictionary.get(word, 0) + 1
        else:
          flag = 1
      
      if flag == 1:
        dictionary['<unk>'] = dictionary.get('<unk>', 0) + 1

    len_D = len(D)
    for token in dictionary:
      dictionary[token] = np.log(len_D / dictionary[token])

    return dictionary

    
class TFIDFFeaturizer(object):
    def __init__(self, idf):
        """The idf scores computed via `compute_idf`."""
        self.idf = idf
    
    def convert_document_to_feature_dictionary(self, doc, vocab):
        """
        Given a document represented as a list of tokens and
        the vocabulary as a set of tokens, compute
        the TF-IDF feature representation. This function
        should return a dictionary which maps from the name of the
        feature to the value of that feature.
        """
        # TODO
        # raise NotImplementedError
        CBoW = {}
        for token in doc:
          if token in vocab:
            CBoW[token] = CBoW.get(token, 0) + 1
          else:
            CBoW['<unk>'] = CBoW.get('<unk>', 0) + 1
        
        TFIDF = {}
        for token in CBoW:
          TFIDF[token] = CBoW[token] * self.idf[token]

        return TFIDF

# You should not need to edit this cell
# def load_document_dataset(file_path):
#     D = []
#     y = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             instance = json.loads(line)
#             D.append(instance['document'])
#             y.append(instance['label'])
#     return D, y

# def convert_to_features(D, featurizer, vocab):
#     X = []
#     for doc in D:
#         X.append(featurizer.convert_document_to_feature_dictionary(doc, vocab))
#     return X

"""##2.3 Naive Bayes Experiment"""

def train_naive_bayes(X, y, k, vocab):
    """
    Computes the statistics for the Naive Bayes classifier.
    X is a list of feature representations, where each representation
    is a dictionary that maps from the feature name to the value.
    y is a list of integers that represent the labels.
    k is a float which is the smoothing parameters.
    vocab is the set of vocabulary tokens.
    
    Returns two values:
        p_y: A dictionary from the label to the corresponding p(y) score
        p_v_y: A nested dictionary where the outer dictionary's key is
            the label and the innner dictionary maps from a feature
            to the probability p(v|y). For example, `p_v_y[1]["hello"]`
            should be p(v="hello"|y=1).
    """
    # TODO
    # raise NotImplementedError
    p_y = {}
    p_y_1 = y.count(1) / len(y)
    p_y_0 = y.count(0) / len(y)
    p_y[0] = p_y_0
    p_y[1] = p_y_1

    p_v_y = {}
    p_v_y[0] = {}
    p_v_y[1] = {}
    donominator_0 = 0
    donominator_1 = 0
    len_v = len(vocab)
    
    for i in range(len(X)):
        if y[i] == 0:
          donominator_0 += sum(X[i][j] for j in X[i])
        else:
          donominator_1 += sum(X[i][j] for j in X[i])

    for i in range(len(X)):
      if y[i] == 0:
        for feature in X[i]:
          p_v_y[0][feature] = p_v_y[0].get(feature, 0) + X[i][feature]
      else:
        for feature in X[i]:
          p_v_y[1][feature] = p_v_y[1].get(feature, 0) + X[i][feature]

    for key in p_v_y[0]:
      p_v_y[0][key] = (p_v_y[0][key] + k) / (k * len_v + donominator_0)
    
    for key in p_v_y[1]:
      p_v_y[1][key] = (p_v_y[1][key] + k) / (k * len_v + donominator_1)
    
    for token in vocab:
      if token not in p_v_y[0]:
        p_v_y[0][token] = k / (k * len_v + donominator_0)
      if token not in p_v_y[1]:
        p_v_y[1][token] = k / (k * len_v + donominator_1)
    
    return p_y, p_v_y

def predict_naive_bayes(D, p_y, p_v_y):
    """
    Runs the prediction rule for Naive Bayes. D is a list of documents,
    where each document is a list of tokens.
    p_y and p_v_y are output from `train_naive_bayes`.
    
    Note that any token which is not in p_v_y should be mapped to
    "<unk>". Further, the input dictionaries are probabilities. You
    should convert them to log-probabilities while you compute
    the Naive Bayes prediction rule to prevent underflow errors.
    
    Returns two values:
        predictions: A list of integer labels, one for each document,
            that is the predicted label for each instance.
        confidences: A list of floats, one for each document, that is
            p(y|d) for the corresponding label that is returned.
    """
    # TODO
    # raise NotImplementedError
    predictions = []
    confidences = []
    for document in D:
      prob_0, prob_1 = 0, 0
      for token in document:
        if token in p_v_y[0]:
          prob_0 += np.log(p_v_y[0][token])
        else:
          prob_0 += np.log(p_v_y[0]['<unk>'])

        if token in p_v_y[1]:
          prob_1 += np.log(p_v_y[1][token])
        else:
          prob_1 += np.log(p_v_y[1]['<unk>'])

      prob_0 += np.log(p_y[0])
      prob_1 += np.log(p_y[1]) 

      pro_0_e = np.exp(prob_0)
      pro_1_e = np.exp(prob_1)

      if prob_0 > prob_1:
        predictions.append(0)
        confidences.append(pro_0_e / (pro_0_e + pro_1_e))
      else:
        predictions.append(1)
        confidences.append(pro_1_e / (pro_0_e + pro_1_e))
      
    return predictions, confidences

"""## Running experiments for document classification"""

# Variables that are named D_* are lists of documents where each
# document is a list of tokens. y_* is a list of integer class labels.
# X_* is a list of the feature dictionaries for each document.
# TODO you likely need to update these paths for your drive setup.

# D_train, y_train = load_document_dataset('/content/drive/MyDrive/Colab Notebooks/data/train.jsonl')
# D_valid, y_valid = load_document_dataset('/content/drive/MyDrive/Colab Notebooks/data/valid.jsonl')
# D_test, y_test = load_document_dataset('/content/drive/MyDrive/Colab Notebooks/data/test.jsonl')

# vocab = get_vocabulary(D_train)

"""BBoWFeaturizer"""

# Compute the features, for example, using the BBowFeaturizer.
# You actually only need to conver the training instances to their
# feature-based representations.
# 
# This is just starter code for the experiment. You need to fill in
# the rest.
# featurizer = BBoWFeaturizer()
# X_train = convert_to_features(D_train, featurizer, vocab)
# best_accuracy = 0
# best_k = 0
# for k in [0.001, 0.01, 0.1, 1.0, 10.0]:
#   p_y, p_v_y = train_naive_bayes(X_train, y_train, k, vocab)
#   predictions, confidences = predict_naive_bayes(D_valid, p_y, p_v_y)
#   valid_acc = accuracy_score(y_valid, predictions)
#   print(f'k = {k}, Valid Acc: {valid_acc}')
#   if valid_acc > best_accuracy:
#     best_accuracy = valid_acc
#     best_k = k

# # using the best value of k for each representation
# p_y, p_v_y = train_naive_bayes(X_train, y_train, best_k, vocab)
# predictions, confidences = predict_naive_bayes(D_test, p_y, p_v_y)
# test_acc = accuracy_score(y_test, predictions)
# print(f'k = {best_k}, Test Acc: {test_acc}')

# featurizer = CBoWFeaturizer()
# X_train = convert_to_features(D_train, featurizer, vocab)
# best_accuracy = 0
# best_k = 0
# for k in [0.001, 0.01, 0.1, 1.0, 10.0]:
#   p_y, p_v_y = train_naive_bayes(X_train, y_train, k, vocab)
#   predictions, confidences = predict_naive_bayes(D_valid, p_y, p_v_y)
#   valid_acc = accuracy_score(y_valid, predictions)
#   print(f'k = {k}, Valid Acc: {valid_acc}')
#   if valid_acc > best_accuracy:
#     best_accuracy = valid_acc
#     best_k = k

# # using the best value of k for each representation
# p_y, p_v_y = train_naive_bayes(X_train, y_train, best_k, vocab)
# predictions, confidences = predict_naive_bayes(D_test, p_y, p_v_y)
# test_acc = accuracy_score(y_test, predictions)
# print(f'k = {best_k}, Test Acc: {test_acc}')

# featurizer = TFIDFFeaturizer(compute_idf(D_train, vocab))
# X_train = convert_to_features(D_train, featurizer, vocab)
# best_accuracy = 0
# best_k = 0
# for k in [0.001, 0.01, 0.1, 1.0, 10.0]:
#   p_y, p_v_y = train_naive_bayes(X_train, y_train, k, vocab)
#   predictions, confidences = predict_naive_bayes(D_valid, p_y, p_v_y)
#   valid_acc = accuracy_score(y_valid, predictions)
#   print(f'k = {k}, Valid Acc: {valid_acc}')
#   if valid_acc > best_accuracy:
#     best_accuracy = valid_acc
#     best_k = k

# # using the best value of k for each representation
# p_y, p_v_y = train_naive_bayes(X_train, y_train, best_k, vocab)
# predictions, confidences = predict_naive_bayes(D_test, p_y, p_v_y)
# test_acc = accuracy_score(y_test, predictions)
# print(f'k = {best_k}, Test Acc: {test_acc}')