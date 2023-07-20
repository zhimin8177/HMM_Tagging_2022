﻿# HMM_Tagging_2022

这是一个用于实现中文词性标注的隐马尔可夫模型。
总共在500个中文语料文档上进行训练，然后用训练好的HMM模型来预测用户输入的句子的词性。

  
## 函数说明：
**text_to_flag**：该函数从指定文件夹读取文本文件，并将每行转换为一个由词-词性对组成的列表。使用jieba库进行中文分词和词性标注。  
**generate_dict**：该函数为在语料库中遇到的词和词性生成词典。为每个词和词性分配唯一的ID，用于索引HMM矩阵。  
**HMM**：这个类表示隐马尔可夫模型。初始化时加载文本数据，并生成三个矩阵：start_matrix、trans_matrix和emis_matrix，分别表示HMM模型的初始状态概率、状态转移概率和词语发射概率。  
**generate_matrix**：计算三个矩阵的概率值，使用训练数据（text_flag_list）来统计。  
**normalize_matrix**：将三个矩阵中的概率值进行归一化，使其成为有效的概率分布。  
**train_matrix**：调用generate_matrix和normalize_matrix来训练HMM模型。  
**viterbi函数**：这实现维特比算法，这是一种动态规划算法，用于找到给定输入句子中最可能的词性序列，利用已经训练好的HMM模型。  
  

## 代码作用：

接受用户输入的句子，使用训练好的HMM模型预测句子中每个词的词性，并将HMM模型的预测结果与使用jieba库进行词性标注的结果进行比较。最后，输出HMM模型预测的词性和jieba库词性标注的对比结果，并用预测词性序列的概率。演示了三个样本句子的结果，展示HMM在中文词性标注上的有效性。同时也指出HMM在某些情况下仍存在改进的空间，例如在处理“隔离”一词时被错误地标记为动词，实际上它应该是一个名词。
  


## 模型结果：

最终概率：2.344162e-36  
hmm预测结果      ：隔离 /v 就是 /d 只能 /v 呆 /v 在 /p 酒店 /n 哪里 /r 都 /d 不能 /v 去 /v   
jieba词性标注结果：隔离 /v 就是 /d 只能 /v 呆 /v 在 /p 酒店 /n 哪里 /r 都 /d 不能 /v 去 /v  

最终概率：1.223969e-22  
hmm预测结果      ：窗户 /n 的 /uj 外面 /f 是 /v 喧嚣 /v 的 /uj 城市 /ns  
jieba词性标注结果：窗户 /n 的 /uj 外面 /f 是 /v 喧嚣 /v 的 /uj 城市 /ns  

最终概率：5.103278e-19  
hmm预测结果      ：都市 /ns 的 /uj 生活 /vn 都 /d 挺 /d 好 /a  
jieba词性标注结果：都市 /ns 的 /uj 生活 /vn 都 /d 挺 /d 好 /a  
  


## 模型效果分析：

由以上例子可见该hmm词性识别在喂了1,008,709词的情况下，拥有较高的准确率，所识别的结果与jieba分出来的无异。
例子1中的“隔离”一词被视为动词，但在该语义下实际应为名词，可见机器词性识别并非完美，仍有进步空间。
