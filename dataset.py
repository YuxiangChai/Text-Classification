import re
import torch
import torch.utils.data as data


class WordDict:

    def __init__(self, train_path):
        train_file = open(train_path, 'r')
        test_file = open(test_path, 'r')
        self.word2idx = {}
        self.idx2word = {}

        i = 1

        for line in train_file:
            sentence = re.findall('__label__\d* (.*)', line)[0]
            words = sentence.split()

            for word in words:
                if word not in self.word2idx:
                    self.word2idx[word] = i
                    self.idx2word[i] = word
                    i += 1

    def get_words(self):
        return self.word2idx, self.idx2word


class DataSet(data.Dataset):

    def __init__(self, path, word_dict, embed_size):
        self.path = path
        self.file = open(path, 'r')
        self.labels = []
        self.sentences = []
        self.word2idx = word_dict[0]
        self.idx2word = word_dict[1]

        for line in self.file:
            sentence_vec = []

            label = int(re.findall('__label__(\d*)', line)[0])
            sentence = re.findall('__label__\d* (.*)', line)[0]
            words = sentence.split()

            for word in words:
                sentence_vec.append(self.word2idx[word])

                if len(sentence_vec) >= embed_size:
                    break

            for j in range(embed_size - len(sentence_vec)):
                sentence_vec.append(0)

            self.sentences.append(sentence_vec)
            self.labels.append(label-1)

        self.sentences = torch.IntTensor(self.sentences).to(torch.int64)
        self.labels = torch.IntTensor(self.labels).to(torch.int64)

    def vocab_size(self):
        return len(self.word2idx)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)




# dataset = Dataset('./test.txt')
# print(dataset.labels)
# print(dataset.sentences)
# print(dataset.vocab_size())
