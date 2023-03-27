import os

import numpy as np
import tensorflow as tf
from tflearn.data_utils import VocabularyProcessor

log = tf.logging.info


class DatasetVectorizerMB:
    
    def __init__(self, model_dir, char_embeddings, raw_sentence=None, save_vocab=True):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.raw_sentence = raw_sentence
        self.char_embeddings = char_embeddings
        self.save_vocab = save_vocab
        # self.restore=restore
        
        if self.raw_sentence is None:
            #VocabularyProcessor.restore('{}/vocab'.format(self.model_dir))
            self.restore()
        else:
            self.raw_sentence = self.raw_sentence
            #self.raw_sentence_pairs = tf.data.shuffle(self.raw_sentence_pairs)
            
            self.raw_sentence = [str(x) for x in list(self.raw_sentence)]
            if self.char_embeddings:
                log('Chosen char embeddings.')
                self.sentences_lengths = [len(list(str(x))) for x in list(self.raw_sentence)]
            else:
                log('Chosen word embeddings.')
                self.sentences_lengths = [len(str(x).split(' ')) for x in list(self.raw_sentence)]
            max_sentence_length = max(self.sentences_lengths)
            log('Maximum sentence length : {}'.format(max_sentence_length))
            
            if self.char_embeddings:
                log('Processing sentences with char embeddings...')
                self.vocabulary = VocabularyProcessor(
                    max_document_length=max_sentence_length,
                    tokenizer_fn=char_tokenizer,
                )
            else:
                log('Processing sentences with word embeddings...')
                self.vocabulary = VocabularyProcessor(
                    max_document_length=max_sentence_length,
                )
            log('Sentences have been successfully processed.')
            self.vocabulary.fit(self.raw_sentence)
            if self.save_vocab:
                self.vocabulary.save('{}/vocabMB'.format(self.model_dir))
    
    @property
    def max_sentence_len(self):
        return self.vocabulary.max_document_length
    
    @property
    def vocabulary_size(self):
        return len(self.vocabulary.vocabulary_._mapping)
    
    
    def restore(self):
        self.vocabulary = VocabularyProcessor.restore('{}/vocabMB'.format(self.model_dir))
    
    def vectorize(self, sentence):
        return np.array(list(self.vocabulary.transform([sentence])))
    
    def vectorize_2d(self, raw_sentence_pairs):
        self.raw_sentence_pairs=raw_sentence_pairs
        num_instances, num_classes = self.raw_sentence_pairs.shape
        self.raw_sentence_pairs = self.raw_sentence_pairs.ravel()
        
        for i, v in enumerate(self.raw_sentence_pairs):
            if v is np.nan:
                print(i, v)
        
        vectorized_sentence_pairs = np.array(list(self.vocabulary.transform(self.raw_sentence_pairs)))
        
        vectorized_sentence_pairs = vectorized_sentence_pairs.reshape(num_instances, num_classes,
                                                                      self.max_sentence_len)
        
        vectorized_sentence1 = vectorized_sentence_pairs[:, 0, :]
        vectorized_sentence2 = vectorized_sentence_pairs[:, 1, :]
        return vectorized_sentence1, vectorized_sentence2


def char_tokenizer(iterator):
    """Tokenizer generator.
  
    Args:
      iterator: Input iterator with strings.
  
    Yields:
      array of tokens per each value in the input.
    """
    for value in iterator:
        yield list(value)
