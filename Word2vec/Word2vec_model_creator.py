from gensim.models.word2vec import LineSentence
from gensim.models import Word2Vec
import warnings
import time
warnings.filterwarnings("ignore")

#size, window and cbow or skipgram defination
dimension=[300,600,1000]
window=[4,10]
objective_function=[0,1]

#creating a table for outputs on screen
#add corpus file for input_file
of_dict = {1: "skipgram",0: "cbow"}
input_file="corpus.txt"
print ('%-25s %-25s' % ("file name","time in seconds"))


#model creation
for i in dimension:
    for ii in window:
        for iii in objective_function:  
            second=time.time()#time stamp
            model = Word2Vec(LineSentence(input_file), vector_size = i, window=ii,sg=iii)
            #writing an output file with the name of size, window and cbow or skipgram
            output_file="model_"+str(i)+"_"+str(ii)+"_"+of_dict[iii]
            model.save(output_file)
            model.wv.save_word2vec_format(output_file, binary=False)
            print ('%-25s %-25s' % (output_file,str(float(time.time()-second))))
