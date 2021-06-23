import glob

files = glob.glob("./not/*")
i=0
for file in files:
    i+=1

print("not=" + str(i))

files_2 = glob.glob("./human/*")
i=0
for file in files_2:
    i+=1

print("human=" + str(i))

files_3 = glob.glob("./demon/*")
i=0
for file in files_3:
    i+=1

print("demon=" + str(i))

#簡潔に
name = ["not","human","demon","demon_padding","human_padding"]
a = 0
for a in range(5):
    b = glob.glob("./img/" + name[a] + "/*")
    i=0
    for file in b:
        i+=1

    print(name[a] + "=" + str(i))
    a+=1


