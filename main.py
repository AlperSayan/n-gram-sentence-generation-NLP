import numpy as np
from Corpus import TurkishSplitter
from NGram import NGram, GoodTuringSmoothing, LaplaceSmoothing
import random


# S015674 Mustafa Alper Sayan Department of Computer Science


def main():

    # construct corpus
    corpus = construct_corpus()


    # construct n-grams
    uni_gram_good_turing = NGram.NGram(1)
    bi_gram_good_turing = NGram.NGram(2)
    tri_gram_good_turing = NGram.NGram(3)

    uni_gram_laplace = NGram.NGram(1)
    bi_gram_laplace = NGram.NGram(2)
    tri_gram_laplace = NGram.NGram(3)

    # add each sentence to the n-grams
    for sentence in corpus:

        splitted = sentence.split()

        uni_gram_good_turing.addNGramSentence(splitted)
        bi_gram_good_turing.addNGramSentence(splitted)
        tri_gram_good_turing.addNGramSentence(splitted)

        uni_gram_laplace.addNGramSentence(splitted)
        bi_gram_laplace.addNGramSentence(splitted)
        tri_gram_laplace.addNGramSentence(splitted)

    # create smoother's
    good_turing_smoothing = GoodTuringSmoothing.GoodTuringSmoothing()
    laplace_smoothing = LaplaceSmoothing.LaplaceSmoothing()

    # calculate n-gram probabilities
    uni_gram_good_turing.calculateNGramProbabilitiesSimple(good_turing_smoothing)
    bi_gram_good_turing.calculateNGramProbabilitiesSimple(good_turing_smoothing)
    tri_gram_good_turing.calculateNGramProbabilitiesSimple(good_turing_smoothing)

    uni_gram_laplace.calculateNGramProbabilitiesSimple(laplace_smoothing)
    bi_gram_laplace.calculateNGramProbabilitiesSimple(laplace_smoothing)
    tri_gram_laplace.calculateNGramProbabilitiesSimple(laplace_smoothing)

    # save n-grams as text files
    uni_gram_good_turing.saveAsText("uni_gram_good_turing.txt")
    bi_gram_good_turing.saveAsText("bi_gram_good_turing.txt")
    tri_gram_good_turing.saveAsText("tri_gram_good_turing.txt")

    uni_gram_laplace.saveAsText("uni_gram_laplace.txt")
    bi_gram_laplace.saveAsText("bi_gram_laplace.txt")
    tri_gram_laplace.saveAsText("tri_gram_laplace.txt")

    txt = ''

    # create n sentences
    uni_gram_good_turing_sentences = create_n_sentences(uni_gram_good_turing, 5)
    bi_gram_good_turing_sentences = create_n_sentences(bi_gram_good_turing, 5)
    tri_gram_good_turing_sentences = create_n_sentences(tri_gram_good_turing, 5)

    uni_gram_laplace_sentences = create_n_sentences(uni_gram_laplace, 5)
    bi_gram_laplace_sentences = create_n_sentences(bi_gram_laplace, 5)
    tri_gram_laplace_sentences = create_n_sentences(tri_gram_laplace, 5)


    txt += ('unigram good turing sentences: \n' + uni_gram_good_turing_sentences)
    txt += '------------------------------------------------------------------------\n\n'
    txt += ('bigram good turing sentence: \n' + bi_gram_good_turing_sentences)
    txt += '\n------------------------------------------------------------------------\n'
    txt += ('trigram good turing sentence: \n' + tri_gram_good_turing_sentences)
    txt += '\n------------------------------------------------------------------------\n'
    txt += ('unigram laplace sentence: \n' + uni_gram_laplace_sentences)
    txt += '\n------------------------------------------------------------------------\n'
    txt += ('bigram laplace  sentence: \n' + bi_gram_laplace_sentences)
    txt += '\n------------------------------------------------------------------------\n'
    txt += ('trigram laplace sentence: \n' + tri_gram_laplace_sentences)

    print(txt)

    # create text file for the generated words
    with open("generated_sentences.txt", "w", encoding='UTF-8') as file:
        file.write(txt)


def create_n_sentences(n_gram, n):

    sentences = ''
    for i in range(n):
        sentences += create_sentence(n_gram) + '\n'

    return sentences


def construct_corpus():

    with open("corpus.txt", "r", encoding='UTF-8') as my_file:
        data = my_file.readlines()

    turkish_splitter = TurkishSplitter.TurkishSplitter()

    corpus = []
    for line in data:
        line.replace('.', '')
        sentences = turkish_splitter.split(line)
        sentences_ = []
        for sentence in sentences:
            sentence = '<s> ' + str(sentence) + ' </s>'
            sentences_.append(sentence)

        corpus.extend(sentences_)

    return corpus


def create_sentence(n_gram):

    n = n_gram.getN()

    # get all the words
    words = list(n_gram.constructDictionaryWithNonRareWords(n, 0))

    # for unigram
    if n == 1:

        generate_starting_n_words(n_gram, words)
        probability_start = []
        for word in words:
            probability_single = n_gram.getProbability(word)
            probability_start.append(probability_single)

        starting_word = '.'

        # because we do not want the first word to be punctuation
        while starting_word in ['</s>', '<s>', '.', ',', '?', '!', '"', '…', "'"]:

            starting_word = random.choices(words, weights=probability_start, k=1)[0]

        sentence = starting_word + ' '
        current_word = starting_word

        # loop until you hit a sentence finisher
        while current_word not in ['.', '?', '…', '!', '</s>']:

            current_word = random.choices(words, weights=probability_start, k=1)[0]

            if current_word in ['<s>', '(', ')', "'", '"']:
                continue

            sentence += current_word + ' '

    # for bigram
    elif n == 2:

        starting_word = generate_starting_n_words(n_gram, words)
        previous_word = starting_word
        current_word = ''
        sentence = starting_word

        while current_word not in ['.', '?', '…', '!', '</s>']:

            pairs = []
            probabilities = []

            for word in words:
                probability_of_the_pair = n_gram.getProbability(previous_word, word)
                pair = [previous_word, word]
                pairs.append(pair)
                probabilities.append(probability_of_the_pair)

            length = len(probabilities)

            indexes = np.arange(0, length, 1)
            index = random.choices(indexes, weights=probabilities, k=1)[0]
            current_word = pairs[index][1]

            if current_word in ['<s>', '(', ')', "'", '"']:
                continue

            sentence += ' ' + current_word
            previous_word = current_word

    # for trigram
    else:

        starting_word, previous_word = generate_starting_n_words(n_gram, words)
        previous_word_2 = starting_word
        current_word = ''
        sentence = starting_word + ' ' + previous_word

        while current_word not in ['.', '?', '…', '!', '</s>']:

            triples = []
            probabilities = []

            for word in words:

                probility_of_the_triple = n_gram.getProbability(previous_word_2, previous_word, word)
                triple = [previous_word_2, previous_word, word]

                triples.append(triple)
                probabilities.append(probility_of_the_triple)

            length = len(triples)
            indexes = np.arange(0, length, 1)

            index = random.choices(indexes, weights=probabilities, k=1)[0]
            current_word = triples[index][2]

            if current_word in ['<s>', '(', ')', "'", '"']:
                continue

            sentence += ' ' + current_word
            previous_word_2 = previous_word
            previous_word = current_word

    return sentence.replace('</s>', '.')


def generate_starting_n_words(n_gram, words):

    n = n_gram.getN()

    if n == 2:

        probabilities = []
        pairs = []
        for word in words:
            probabilities.append(n_gram.getProbability('<s>', word))
            pairs.append(['<s>', word])

        length = len(probabilities)
        indexes = np.arange(0, length, 1)
        index = random.choices(indexes, weights=probabilities, k=1)[0]

        starting_word = pairs[index][1]

        return starting_word

    else:

        probabilities = []
        pairs = []
        for word in words:
            probabilities.append(n_gram.getProbability('<s>', '<s>', word))
            pairs.append(['<s>', word])

        length = len(probabilities)
        indexes = np.arange(0, length, 1)
        index = random.choices(indexes, weights=probabilities, k=1)[0]

        starting_word = pairs[index][1]

        probabilities = []
        pairs = []

        for word in words:
            probabilities.append(n_gram.getProbability('<s>', starting_word, word))
            pairs.append([starting_word, word])

        length = len(probabilities)
        indexes = np.arange(0, length, 1)
        index = random.choices(indexes, weights=probabilities, k=1)[0]

        previous_word = pairs[index][1]

        return starting_word, previous_word


if __name__ == '__main__':
    main()
