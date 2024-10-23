#Slicing slice(start,stop,step)

#Cannot perform slicing on dictionary and set

# strsample="College of engineering"
# print(strsample)
# print(strsample[slice(1,4,2)])
# print(strsample[:3])
# print(strsample[2:])

# sam=[1,2,"Hello","world",100.0]
# print(sam[:1])
# print(sam[1:])

# from array import*

# arr=array('i',[1,2,3,4,5,6,7,8,9])
# print(arr)
# print(arr[4:])

# ran=range(1,100,2)
# for x in ran: print(x)
# print(ran[40:])


#Concatenation (+)


# strsample="Hello"
# print(strsample+'',"World")

# lstsample=[1,2,'sam']
# print(lstsample)
# lstsample+="py"
# print(lstsample)

# from array import*
# arrsam=array('i',[1,2,3,4])
# print(arrsam)
# arrsam+=array('i',[50,60])
# print(arrsam)

# tupe=(1,2,'s')
# print(tupe)
# tupe+=("he","ll")
# print(tupe)


# se=("Exam","Post",100.0989)
# print(se)
# se+=(77,'a')
# print(se)
# se=se,100
# print(se)

#Multiplication (*)
#Cannot be performed on range and dictionary

# strsample="HEllo"
# strsample*=3
# print(strsample)

# lstsample=[1,2,'sam']
# lstsample*=10
# print(lstsample)
# lstsample[2]*=10
# print(lstsample)

# tupe=2,3,4,5
# print(tupe)
# tupe*=2
# print(tupe)

from array import*
arr=array('f',[1,2,3,4,5])
print(arr)
arr*=3
print(arr)