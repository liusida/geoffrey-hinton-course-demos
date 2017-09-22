class dictionary:
    def __init__(self):
        self.word2index = {}
        self.index2word = {}
        self.freezed = False

    @property
    def num_words(self):
        return len(self.word2index)

    def lookup(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            if self.freezed:
                return -1
            else:
                new_index = len(self.word2index)
                self.word2index[word] = new_index
                self.index2word[new_index] = word
                return new_index

    def lookup_index(self, index):
        if index in self.index2word:
            return self.index2word[index]
        else:
            return '-'

    def freeze(self):
        self.freezed = True
