import json
import collections
import numpy as np
import os


class JsonFile(object):

    def __init__(self, json_fname, num_items=-1):
        with open(json_fname, 'r') as f:
            self.dataset = json.load(f)
            self.dataset_items = self.dataset['items']
            if num_items > 0:  # get the first num_items if needed
                self.dataset_items = self.dataset['items'][0: num_items]

        return

    def get_num_items(self):
        return len(self.dataset_items)

    def get_ids_split(self, target_split='train'):
        ids = []
        for item in self.dataset_items:
            imgid = item['imgid']
            split = item['split']
            if split == target_split:
                ids.append(imgid)
        return ids

    def get_item_from_img_id(self, target_img_id):
        item = None
        for current_item in self.dataset_items:
            imgid = current_item['imgid']
            if imgid == target_img_id:
                item = current_item
                break  # found the right item
        return item

    def get_word_list_of_img_id(self, img_id):
        """
        return a list of unique words that correspond to the img_id
        """
        word_list = []
        item = self.get_item_from_img_id(img_id)
        if item is None:
            return word_list
        word_list = self.get_word_list_from_item(item)
        return word_list

    @staticmethod
    def get_word_list_from_item(item):
        """
        return a list of unique words that correspond to item
        """
        txt = item['text']
        word_list = list(set([w.replace('\n', "") for w in txt.split(" ")]))
        return word_list

    # def get_text_of_img_ids(self, img_ids):
    #     id2text = {}
    #     for img_id in img_ids:
    #         # get item from json
    #         item = self.get_item_from_img_id(img_id)
    #         if item is None:
    #             print img_id, " not found"
    #             continue
    #         txt = item['text']
    #         id2text[img_id] = txt
    #     return id2text

    # def get_vocab_from_json(self):
    #     vocab = set()
    #     for item in self.dataset['items']:
    #         imgid = item['imgid']
    #         word_list = self.get_word_list_of_img_id(imgid)
    #         vocab.update(word_list)
    #     return vocab

    @staticmethod
    def get_vocabulary_words_with_counts(txt, min_word_freq):
        """(str, int) -> list
        Extract the vocabulary from a string that occur more than min_word_freq.
        Return a list of the vocabulary and the frequencies.
        """

        data = txt.split()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        # keep words that occur more than min_word_freq
        top_count_pairs = [pair for pair in count_pairs if pair[1] > min_word_freq]
        return top_count_pairs

    def json2txt_file(self, fout_name):
        """
        Read json file corresponding and write a txt file with
        all the text from the json file one line for each line.
        Assume that the text on json is already clean.
        Here we lose all the info of which product belongs to what,
        but this is useful when you just want to see all
        the text, like when training an LSTM based on text alone
        """

        f = open(fout_name, 'w')

        i = 0
        for item in self.dataset_items:
            text = item['text']
            sentences = text.split('\n ')  # assume that sentences end with "\n "
            for l in sentences:
                if len(l) == 0:
                    continue
                if not l.strip().isspace():
                    f.write(l + '\n')
            i += 1

        return

    def get_all_txt_from_json(self):

        """
        Concatenate all text from json and return it.
        """

        self.json2txt_file("tmp.txt")  # save a temp file with all the text
        with open("tmp.txt", 'r') as f:
            txt = f.read()

        os.remove("tmp.txt")  # remove temp file

        return txt

    def get_vocab_words_from_json(self, min_word_freq=5):
        """
        Get vocab words from json sorted by freq
        """
        all_text = self.get_all_txt_from_json()
        vocab_with_counts = self.get_vocabulary_words_with_counts(all_text, min_word_freq)
        vocab_words = [w[0] for w in vocab_with_counts]
        return vocab_words

    def get_num_vocab_words_from_json(self, min_word_freq=5):
        return len(self.get_vocab_words_from_json(min_word_freq=min_word_freq))

    # TODO: Implement:
    # get_num_tokens_in_json
    # get_avg_num_tokens_per_product


# class JsonData(object):
#
#     def __init__(self, data_config):
#         self.d = data_config
#
#         self.json_train = {}
#         self.json_val = {}
#         self.json_test = {}
#
#         return
#
#     def set_json_split(self, split='test'):
#         if split == 'train':
#             fname = self.d['cnn_regions_path_train']
#             self.json_train = np.loadtxt(fname, delimiter=',')
#         elif split == 'val':
#             fname = self.d['cnn_regions_path_val']
#             self.json_val = np.loadtxt(fname, delimiter=',')
#         elif split == 'test':
#             fname = self.d['cnn_regions_path_test']
#             self.json_test = np.loadtxt(fname, delimiter=',')
#         else:
#             raise ValueError("only train, val and test splits supported")
#         return
#
