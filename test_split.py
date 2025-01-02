def split(l):
	if sum(i[1] for i in l)>45:
		return split(l[:len(l)//2]) + split(l[len(l)//2:])
	else:
		return [l]

test = [[i,i*3] for i in range(16)]
print(test)
print(split(test))
