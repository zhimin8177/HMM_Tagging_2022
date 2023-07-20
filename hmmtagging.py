'''张芷敏 作业2 HMM词性标注'''
import jieba.posseg as psg
import jieba
import numpy as np
import re
import os

'''
文件夹中共有500个语料文档，来源为http://corpus.bfsu.edu.cn/info/1070/1389.htm
'''
def remove_space(string):
    pattern = re.compile(r'\s+')
    return re.sub(pattern, '', string)

#把文件夹中的所有文件中的每一行转换成list[[[行1词1,词性1][行1词2，词性2]][[行2词1]]]
def text_to_flag(folder_path):
    current_path = os.path.dirname(__file__)  # 文件夹目录
    # folder_path = "/ToRCHtest/ToRCH2019_V1.5"
    path = current_path + folder_path
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    n = 0
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            f = open(path+"/"+file, encoding="utf-8")  # 打开文件
            iter_f = iter(f)  # 创建迭代器
            for line in iter_f:  # 遍历文件，一行行遍历，读取文本
                text = psg.cut(remove_space(line))#由于语料库事先已经分好词了，将他重新合并交给jieba
                str = []
                for x in text:
                    str.append([x.word, x.flag])
                s.append(str)  # 每个文件的文本存到list中
            n += 1
        print(n)
    f.close()
    return(s)


def generate_dict(s):
    worddict = {} #例：{"词1":1，"词2":2}
    flagdict = {} #例：{"词性1":1，"词性2":2}
    word_id_count = 0
    flag_id_count = 0
    for i in s: #读取文档的每一行
        word_count = len(i)
        for j in i: #读取行中每个单词
            if j[0] not in worddict:
                worddict[j[0]] = word_id_count
                word_id_count += 1
            if j[1] not in flagdict:
                flagdict[j[1]] = flag_id_count
                flag_id_count += 1
    return worddict,flagdict

class HMM:
    def __init__(self,folder_path="/ToRCH2019/ToRCH2019_V1.5"):
        self.text_flag_list = text_to_flag(folder_path)
        self.worddict, self.flagdict = generate_dict(self.text_flag_list)
        flag_num = len(self.flagdict)
        self.start_matrix = np.zeros(flag_num) #开始矩阵
        self.trans_matrix = np.zeros((flag_num,flag_num)) #转移矩阵
        self.emis_matrix = np.zeros((flag_num,len(self.worddict))) #发射矩阵


    def generate_matrix(self):
        for i in self.text_flag_list: #读取文档的每一行
            prev_flag = None
            for j in i: #读取行中每个单词
                word_id = self.worddict[j[0]]
                flag_id = self.flagdict[j[1]]
                self.emis_matrix[flag_id][word_id] += 1 #将对应词性和单词记录进发射矩阵
                if prev_flag is None: #如果是句子第一个单词
                    self.start_matrix[flag_id] += 1 #录入初始矩阵
                    prev_flag = flag_id #记录当前词性，用于转移矩阵收录
                else: #如果不是第一个词
                    self.trans_matrix[prev_flag][flag_id] += 1 #录入转移矩阵 
                    prev_flag = flag_id #记录当前词性，用于转移矩阵收录

    def normalize_matrix(self):
        self.start_matrix = self.start_matrix/np.sum(self.start_matrix)
        self.trans_matrix = self.trans_matrix/np.sum(self.trans_matrix,axis = 1,keepdims = True)
        self.emis_matrix = self.emis_matrix/np.sum(self.emis_matrix,axis = 1,keepdims = True)
    
    def train_matrix(self):
        self.generate_matrix()
        self.normalize_matrix()


#viterbi算法
def viterbi(hmm,text):
    worddict = hmm.worddict
    flagdict = hmm.flagdict
    start_p = hmm.start_matrix
    trans_p = hmm.trans_matrix
    emis_p = hmm.emis_matrix
    seperated_text = jieba.cut(text)
    text_to_cal = []
    for word in seperated_text: #把分词结果输入列表
        try:
            text_to_cal.append(worddict[word])
        except: #未登录词处理
            wordcount = len(worddict)
            worddict[word] = wordcount
            temp_matrix = np.ones((len(flagdict),1))
            emis_p = np.concatenate((emis_p,temp_matrix),axis=1) #在发射矩阵增加全为1的末列
            text_to_cal.append(worddict[word])
    
    flag_count = len(flagdict)
    text_count = len(text_to_cal)
    delta = np.zeros((text_count,flag_count)) #第i个字是j词性的最大概率
    phi = np.zeros((text_count,flag_count),dtype=int) #存储路径


    for i in range(flag_count): #对于每个词性，词性的初始概率*第一个单词发射矩阵中对应词性的概率
        delta[0][i] = start_p[i]*emis_p[i][text_to_cal[0]]


    for i in range(1,text_count): #第一位之后的单词 
        for j in range(flag_count):
            temp = [delta[i-1][k]*trans_p[k][j] for k in range(flag_count)] #乘以转移矩阵对应的概率
            delta[i][j] = max(temp)*emis_p[j][text_to_cal[i]] #第i个词是j词性的最大概率
            phi[i][j] = temp.index(max(temp)) #第i个词是j词性的前一个词性

    probability = max(delta[text_count-1,:]) #最后的概率
    path = [int(np.argmax(delta[text_count-1,:]))] #列表用于记录最优路径，填入最后的词性 （概率最大）

    for i in reversed(range(1,text_count)): #从后面开始记录最优路径
        end = path[-1] #记录上一个词性是什么
        path.append(phi[i,end])

    new_dict_f = {v:k for k,v in flagdict.items()}

    #输出答案
    ans_path = []
    for flag in reversed(path):
        ans_path.append(new_dict_f[flag]) #转换最优路径形式与顺序

    i = 0
    ans = []
    seperated_text = jieba.cut(text)
    for word in seperated_text:
        temp = word + " /" + ans_path[i] #合并词和词性
        ans.append(temp)
        i+=1
   
    print("%e"%probability) #科学计数法输出概率
    print("hmm预测结果      ：" + " ".join(ans))

    jieba_ans = psg.cut(text)
    print("jieba词性标注结果：" + " ".join(["%s /%s" % (w,tag) for w,tag in jieba_ans]))

hmm = HMM()
hmm.train_matrix()
print("请输入需要识别词性的语句：")
text = input()
viterbi(hmm,text)

# textlist = []
# textlist.append("隔离就是只能呆在酒店哪里都不能去")
# textlist.append("窗户的外面是喧嚣的城市")
# textlist.append("都市的生活都挺好")
# for text in textlist:  
#     viterbi(hmm,text)

'''
最终概率：2.344162e-36
hmm预测结果      ：隔离 /v 就是 /d 只能 /v 呆 /v 在 /p 酒店 /n 哪里 /r 都 /d 不能 /v 去 /v
jieba词性标注结果：隔离 /v 就是 /d 只能 /v 呆 /v 在 /p 酒店 /n 哪里 /r 都 /d 不能 /v 去 /v

最终概率：1.223969e-22
hmm预测结果      ：窗户 /n 的 /uj 外面 /f 是 /v 喧嚣 /v 的 /uj 城市 /ns
jieba词性标注结果：窗户 /n 的 /uj 外面 /f 是 /v 喧嚣 /v 的 /uj 城市 /ns

最终概率：5.103278e-19
hmm预测结果      ：都市 /ns 的 /uj 生活 /vn 都 /d 挺 /d 好 /a
jieba词性标注结果：都市 /ns 的 /uj 生活 /vn 都 /d 挺 /d 好 /a

由以上例子可见该hmm词性识别在喂了1,008,709词的情况下，拥有较高的准确率，所识别的结果与jieba分出来的无异。
例子1中的“隔离”一词被视为动词，但在该语义下实际应为名词，可见机器词性识别并非完美，仍有进步空间。
'''
