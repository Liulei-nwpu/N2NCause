import re
import random
from matplotlib.pyplot import axis
#from torch import instance_norm
import tensorflow as tf
import json
from utils.config_helpers import MainConfig
import numpy as np
from tqdm import tqdm

class Memorybank:
    def __init__(self,vectorize,sentences = None,Labels = None,span1s = None,span2s = None,directions = None):
        self.MBfile = 'memorybank/ECMUL/instances.txt'
        self.vectorizer = vectorize
        self.max_sentence_length = vectorize.max_sentence_len
        self.embedding_size = 64
        self.word_embeddings = tf.get_variable('word_embeddings',[self.vectorizer.vocabulary_size, self.embedding_size])
        self.embeddings = None
        self.sentences = sentences
        self.Labels = Labels
        self.span1s = span1s
        self.span2s = span2s
        self.directions = directions

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
        file = open('corpora/Crest/train.txt','w+',encoding='utf-8')
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
            span1 = self.span1s[cnt]
            span2 = self.span2s[cnt]
            if self.Labels[cnt] != '0':
                span1 = self.span1s[cnt]
                span2 = self.span2s[cnt]
                span1_token = self.vectorizer.vectorize(span1)
                span2_token = self.vectorizer.vectorize(span2)
                span1_vec = np.take(self.embeddings,span1_token,axis=0)
                span2_vec = np.take(self.embeddings,span2_token,axis=0)
                dict_sim = {}
                dict_, dict_cnt= self.soft_causality(span1_vec,span2_vec,mem_e1_vec,mem_e2_vec,mem_cnt)
                for j in list(dict_.keys()):
                    dict_sim[j] = dict_[j]

                dist_sort = dict(sorted(dict_sim.items(),key = lambda item:item[1],reverse=True))
                key_out = list(dist_sort.keys())
                connective = dict_cnt[key_out[0]]
                connective = connective.strip()
                # neg_num = random.sample(set(range(0,len(self.span1s))),2)
                # random_key = neg_num[0]
                # random_value = neg_num[1]
                # spanr1 = self.span1s[random_key]
                # spanr2 = self.span2s[random_value]
                if self.directions[cnt] == '1':
                    sent2_pos = span1 + ' ' + connective + ' ' + span2
                    sent2_neg = span2 + ' ' + connective + ' ' + span1
                elif self.directions[cnt] == '0':
                    sent2_pos = span2 + ' ' + connective + ' ' + span1
                    sent2_neg = span1 + ' ' + connective + ' ' + span2
                    
            else:
                sent2_pos = 'There is no causality between {} and {}.'.format(span1,span2)
                sent2_neg = '{} is the reason of {}.'.format(span1,span2)
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

    def soft_causality(self,span1_vec,span2_vec,mem_e1_vec,mem_e2_vec,mem_cnt):
        #print("is running")
        dict_cnt = {}
        dict_sim = {}
        for j in range(0,len(mem_e1_vec)):
            dict_cnt[j] = mem_cnt[j]
        vec_1 = span1_vec
        vec_2 = span2_vec
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
