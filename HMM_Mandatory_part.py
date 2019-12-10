import csv
import operator
import math

with open('tag_logit_per_word.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    words_arr = []
    tag_table = []
   
    for row in reader:
        words_arr.append(row[''])
        tag_table.append(row)
    
    tags_arr = reader.fieldnames

words_arr = [word.lower() for word in words_arr]

def get_tag_prob_per_word(tag, word):
    if word in words_arr:
        ind_word = words_arr.index(word)

    return float(tag_table[ind_word][tag])

def load_dataset(filename):
    tag_state = []
    word_observ = []
    with open(filename) as file:
        contents = file.readlines()
    contents = [content.strip() for content in contents]
    idx_line = 0
    
    while idx_line < len(contents):
        sentence = []
        tag = []
        
        while len(contents[idx_line]) > 2:
            content = contents[idx_line].split(' ')
            sentence.append(content[0].lower())
            tag.append(content[1])
            idx_line += 1

        word_observ.append(sentence)
        tag_state.append(tag)
        idx_line += 1
        
    return len(contents), word_observ, tag_state

train_len, sentences_train, tags_train = load_dataset('train_pos.txt')
test_len, sentences_test, tags_test = load_dataset('test_pos.txt')

def get_tag_count(tags):
    tag_count = {}
    for tag in tags:
        for tag_word in tag:
            if tag_word in tag_count:
                tag_count[tag_word] += 1
            else:
                tag_count[tag_word] = 1
    tag_count['<start>'] = len(tags)
    return tag_count


def get_tag_transition(tags):
    tag_trans = {}
    for tag in tags:
        previous_word = '<start>'
        for tag_word in tag:
            if previous_word in tag_trans:
                if tag_word in tag_trans[previous_word]:
                    tag_trans[previous_word][tag_word] += 1
                else:
                    tag_trans[previous_word][tag_word] = 1
            else:
                tag_trans[previous_word] = {tag_word: 1}
            previous_word = tag_word
    return tag_trans


def get_trans_prob_table(tag_trans, tag_count):
    trans_prob_table = {}
    for tag1 in tag_count.keys():
        trans_prob_table[tag1] = {}
        for tag2 in tag_count.keys():
            if tag2 in tag_trans[tag1].keys():
                trans_prob_table[tag1][tag2] = tag_trans[tag1][tag2]/tag_count[tag1]
            else:
                continue
    return trans_prob_table

def get_prior_tag_prob(tag_count):
    prior = {}
    for tag in tag_count.keys():
        prior[tag] = tag_count[tag]/(train_len - 1)
    return prior

def viterbi_algorithm(sentence, trans_prob_table, prior):
    sentence_tag = []
    viterbi_table  = {}
    max_previous_value = 0
    previous_tag = '<start>'
    for word in sentence.split():
        viterbi_table[word] = {}
        if word not in words_arr:
            tag_max = max(trans_prob_table[previous_tag].items(), key=operator.itemgetter(1))[0]
            viterbi_table[word][tag_max] = max_previous_value + math.log(trans_prob_table[previous_tag][tag_max])
        else:
            for tag in tags_arr:
                if tag in trans_prob_table[previous_tag]:
                    viterbi_table[word][tag] = max_previous_value + math.log(trans_prob_table[previous_tag][tag]) + get_tag_prob_per_word(tag, word) - math.log(prior[tag])
        previous_tag = max(viterbi_table[word].items(), key=operator.itemgetter(1))[0]
        max_previous_value = viterbi_table[word][previous_tag]
        sentence_tag.append(previous_tag)
    return sentence_tag, viterbi_table

def get_viterbi_accuracy(sentences, tags, viterbi_algorithm, trans_prob_table, prior):
    n_true = 0
    n_word = 0
    for sentence, tag in zip(sentences, tags):
        sentence = ' '.join(sentence)
        pred_tag, _ = viterbi_algorithm(sentence, trans_prob_table, prior)
        for pred, truth in zip(pred_tag, tag):
            n_word +=1
            if pred == truth:
                n_true += 1
    return n_true / n_word * 100

tag_count = get_tag_count(tags_train)
tag_trans = get_tag_transition(tags_train)
trans_prob_table = get_trans_prob_table(tag_trans, tag_count)
prior = get_prior_tag_prob(tag_count)

print("Viterbi Algorithm Accuracy:")
print(get_viterbi_accuracy(sentences_test, tags_test, viterbi_algorithm, trans_prob_table, prior))
print('rockwell international corp. \'s tulsa unit said it signed a tentative agreement extending its contract with boeing co. to provide structural parts for boeing \'s 747 jetliners .')
print(viterbi_algorithm('rockwell international corp. \'s tulsa unit said it signed a tentative agreement extending its contract with boeing co. to provide structural parts for boeing \'s 747 jetliners .', trans_prob_table, prior)[0])
print(tags_test[0])
