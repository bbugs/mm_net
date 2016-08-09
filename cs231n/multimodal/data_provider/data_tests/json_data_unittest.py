
from cs231n.multimodal.data_provider.data_tests import data_config
from cs231n.multimodal.data_provider.json_data import JsonFile

d = data_config.dc

json_file = JsonFile(d['json_path_test'], num_items=10)

print "\nnum of items"
print json_file.get_num_items()
# 10

print "\nids from split"
print json_file.get_img_ids()
# [6, 80, 147, 212, 261, 373, 385, 431, 460, 476]

print "\nindex of img id"
print json_file.get_index_from_img_id(476)  # 9

print "\nitem of img id"
print json_file.get_item_from_img_id(target_img_id=476)
# {u'asin': u'B00EC7KR14', u'url': u'http://ecx.images-amazon.com/images/I/41FQgL4OxAL.jpg', u'text': u'vogue ...

print "\nwords of img id"
words = json_file.get_word_list_of_img_id(476)
print "\n", words
print "\nnum words of img id"
print len(words)

vocab_words = json_file.get_vocab_words_from_json(min_word_freq=0)
print "\nvocab_words"
print vocab_words
print "\nnum vocab words"
print json_file.get_num_vocab_words_from_json(min_word_freq=5)

json_file = JsonFile(d['json_path_train'], num_items=-1)
print "\num vocab words for the split"
print json_file.get_num_vocab_words_from_json(min_word_freq=5)


