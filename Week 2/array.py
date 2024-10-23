import numpy as np
arr =np.array([1,2,3,4,5,6])
# print(arr)
# print(type(arr))
# print(len(arr))
# print(arr.ndim)
# print(arr.shape)

# arr2=arr.reshape(2,3)
# print(arr2)
# print(arr2.shape)

# arr3=arr.reshape(3,-1)
# print(arr3)
# print(arr3.shape) 


# my_lis1=[1,2,3,4,5]
# my_lis2=[6,7,8,9,10]
# my_lis3=[11,12,13,14,15]

# mul_arr=np.array([my_lis1,my_lis2,my_lis3])
# print(mul_arr)

# mul_arr=mul_arr.reshape(1,15)
# print(mul_arr)

# a=np.arange(20)
# print(a)
# a=a.reshape(4,-1)
# print(a)

# a=[1,2,3,4]
# b=[1,2,3,4]
# print(np.multiply(a,b))
# print(np.add(a,b))

# print(np.sum(a))
# print(np.sum(b))

# arr = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
# print(arr[0][1])

s={1,2,4,5}
l=[]

# for x in range(len(s)):
#     l+=[1+x]
# print(l)

# arr=np.array(np.arange(0,15))
# arr=arr.reshape(3,5)
# print(arr)

t1 = (1, 2, "tuple", 4)
t2 = (5, 6, 7)

# t1.append(5)
# print(t1)
# t3=t2[t1[1]]
# print(t3)
# t3=t1+t2
# print(t2+t1)
# t3=(t1,t2)
# print (t3)
# t3=(list(t1),list(t2))
# print(t3)

# student = {'name': 'Jane', 'age': 25, 'courses': ['Math', 'Statistics']}
# student['phone'] = '123-456'
# student.update({'age' : 26})
# print(student)

name="Mahesh"
l=[]
for i in name:
    l.append(i.capitalize())
print(l)