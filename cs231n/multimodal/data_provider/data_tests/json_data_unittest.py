
from cs231n.multimodal.data_provider.data_tests import data_config
from cs231n.multimodal.data_provider.json_data import JsonFile

d = data_config.dc

json_file = JsonFile(d['json_path_test'], num_items=10)

print json_file.get_num_items()

print json_file.get_ids_split(target_split='test')

print json_file.get_item_from_img_id(target_img_id=476)

words = json_file.get_word_list_of_img_id(476)
print "\n", words
print "\n", len(words)

vocab_words = json_file.get_vocab_words_from_json(min_word_freq=0)
print vocab_words
print json_file.get_num_vocab_words_from_json(min_word_freq=5)

json_file = JsonFile(d['json_path_train'], num_items=-1)
print json_file.get_num_vocab_words_from_json(min_word_freq=5)


