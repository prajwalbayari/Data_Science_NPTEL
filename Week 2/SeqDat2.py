#INDEXING This can be performed on lists,tuples,arrays and dictionary
#Set does not support indexing as it is a container to hold nonlinear data

sample="Hello world"
print(sample[1],sample[10],sample.index('o'),sample.index("wo"))
print(sample[-3],sample[-11])
print(sample.index('o',0,11)) #index(item,start,end)

sam=['Hell',1.0,10,'i']
print(sam[0],sam[-4])

from array import*
arr=array('f',[10,20,30])
print(arr[-3])

tup=('1','py',10,20)
print(tup.index("py"))

#In dictionary we access data using key 

dic=dict([(1,10),(2,20),('sample','Hello')])
print(dic[1],dic['sample'])