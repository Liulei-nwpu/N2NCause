import re
import random
from matplotlib.pyplot import axis
#from torch import instance_norm
import tensorflow as tf
import json
from utils.config_helpers import MainConfig
from layers.similarity import cosine_distance,euclidean_distance,manhattan_similarity
import numpy as np
from string import punctuation
import time
from tqdm import tqdm

class Memorybank:
    def __init__(self,vectorize,sentences = None,Labels = None):
        self.MBfile = 'memorybank/ECMUL/instances.txt'
        self.vectorizer = vectorize
        self.max_sentence_length = vectorize.max_sentence_len
        self.embedding_size = 64
        self.word_embeddings = tf.get_variable('word_embeddings',[self.vectorizer.vocabulary_size, self.embedding_size])
        self.embeddings = None
        self.sentences = sentences
        self.Labels = Labels

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.embeddings = self.word_embeddings.eval()

    # get all entitys in the sentences
    def get_entitys(self,sentence):
        arguements = {}
        tag = []
        for i in range(1,26):
            e = 'e' + str(i)
            e_1 = '<{}>'.format(e)
            e_2e = '</{}>'.format(e)
            pattern = '<{}>(.*)</{}>'.format(e, e)
            e_all = re.findall(pattern,sentence)
            if len(e_all) > 0:
                arguements[e] = e_all[0]
                tag.append(e)
            else:
                continue
            # need to pay attention to
            sentence = sentence.replace(e_1,"").replace(e_2e,"")
        # return self.arguements
        return sentence,arguements,tag

    def get_relation_num(self,labels):
        count = 0
        neg = 0
        pos = 0
        for label in labels:
            if label == 'noncause':
                neg += 2
            else:
                relations = re.findall('cause-effect\((.*)\)', label)[0]
                entitys = relations.split(',')
                # print(len(entitys))
                pos += len(entitys)
        count = pos + neg
        return count

    def get_relation(self,label):
        keys = []
        values = []
        relations = re.findall('cause-effect\((.*)\)', label)[0]
        #print(relations)
        entitys = relations.split(',')
        i = 0
        while i < len(entitys):
            e1 = entitys[i].lstrip('(')
            e2 = entitys[i + 1].rstrip(')')
            keys.append(e1)
            values.append(e2)
            i += 2
        return keys,values

    # embedding MB file
    def MB_vector(self):
        pattern_e1,pattern_e2,pattern_cnt = [],[],[]
        with open(self.MBfile,'r',encoding = 'utf-8') as fr:
            lines = fr.readlines()
            for line in lines:
                #print(line)
                ent_con = line.split('\t')
                #print(ent_con)
                e1 = ent_con[0]
                e2 = ent_con[1]
                connective = ent_con[2].strip('\n')
                e1_ids = self.vectorizer.vectorize(e1)
                e2_ids = self.vectorizer.vectorize(e2)
                #connective_ids = self.vectorizer.vectorize(connective)
                pattern_e1.append(e1_ids)
                pattern_e2.append(e2_ids)
                pattern_cnt.append(connective)
        fr.close()
        return pattern_e1,pattern_e2,pattern_cnt

    def Tvector(self):
        file = open('corpora/ECSIN/train.txt','w+',encoding='utf-8')
        cnt = 0
        mem_e1, mem_e2,mem_cnt = self.MB_vector()
        mem_e1_vec,mem_e2_vec = [],[]
        for i in range(0,len(mem_e1)):
            a1 = np.take(self.embeddings,mem_e1[i],axis=0)
            mem_e1_vec.append(a1)
            a2 = np.take(self.embeddings,mem_e2[i],axis=0)
            mem_e2_vec.append(a2)
        for cnt in tqdm(range(len(self.Labels))):
            #print(cnt)
            sentence = self.sentences[cnt]
            if self.Labels[cnt] != 'noncause':
                arguements_vec = {}
                arguements_id = {}
                sentence, arguements, tag = self.get_entitys(sentence)
                keys,values= self.get_relation(self.Labels[cnt])
                for i in tag:
                    arguements_id[i] = self.vectorizer.vectorize(arguements[i])
                    arguements_vec[i] = np.take(self.embeddings,arguements_id[i],axis=0)
                for i in range(0,len(keys)):
                    dict_sim = {}
                    key = keys[i]
                    value = values[i]
                    dict_, dict_cnt= self.soft_causality(key,value,mem_e1_vec,mem_e2_vec,mem_cnt,arguements_vec)
                    for j in list(dict_.keys()):
                        dict_sim[j] = dict_[j]

                    dist_sort = dict(sorted(dict_sim.items(),key = lambda item:item[1],reverse=True))
                    key_out = list(dist_sort.keys())
                    connective = dict_cnt[key_out[0]]
                    connective = connective.strip()
                    sent2_pos = arguements[key] + ' ' + connective + ' ' + arguements[value]
                    sent2_neg = arguements[value] + ' ' + connective + ' ' + arguements[key]

            else:
                for i in range(1,26):
                    e = 'e' + str(i)
                    e_1 = '<{}>'.format(e)
                    e_2e = '</{}>'.format(e)
                    sentence = sentence.replace(e_1,"").replace(e_2e,"")

                words = sentence.split()
                e1 = words[0]
                e2 = words[len(words)-1]
                sent2_pos = 'There is no causality between {} and {}.'.format(e1,e2)
                sent2_neg = '{} is the reason of {}.'.format(e1,e2)
            # x = re.sub(r"\\n", "", sentence)
            # x = x.replace('\\n','')
            # x = x.replace('\n', '')
            # x = x.replace('\r', '')
            # x = x.replace('.', '')
            # x = x.replace('#', '')
            # x = x.replace('\t', '')
            # x = x.replace('\\t', '')
            # x = x.replace('$', '')
            # y = re.sub(r"\"\"", "", x)
            # z = re.sub\
            #     (r"\"", "", y)
            # sentence = re.sub(r"\\", "", z)
            sentence = sentence.strip()
            positive_instance = '{}\t{}\t{}\n'.format(sentence, sent2_pos, 1)
            negative_instance = '{}\t{}\t{}\n'.format(sentence, sent2_neg, 0)
            file.write(positive_instance)
            file.write(negative_instance)
        
        file.close()

    def soft_causality(self,key,value,mem_e1_vec,mem_e2_vec,mem_cnt,arguements_vec):
        #print("is running")
        dict_cnt = {}
        dict_sim = {}
        for j in range(0,len(mem_e1_vec)):
            dict_cnt[j] = mem_cnt[j]
        vec_1 = arguements_vec[key]
        vec_2 = arguements_vec[value]
        q_vec = np.concatenate((vec_1,vec_2),axis = 1)

        ls = random.sample(range(0,len(mem_e1_vec)),25)
    
        for j in ls:
            e1 = mem_e1_vec[j]
            e2 = mem_e2_vec[j]
            k_vec = np.concatenate((e1,e2),axis = 1)
            
            q_vec_1 = np.mean(q_vec,axis=2)
            k_vec_1 = np.mean(k_vec,axis=2)
            dist = np.dot(q_vec_1,k_vec_1.T)/(np.linalg.norm(q_vec_1)*np.linalg.norm(k_vec_1))

            dict_sim[j] = dist

        return dict_sim,dict_cnt
