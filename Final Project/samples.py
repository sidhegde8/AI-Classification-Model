import zipfile
import os
from util import arrayInvert

def readlines(filename):
    if os.path.exists(filename):
        return [l[:-1] for l in open(filename).readlines()]
    else:
        z = zipfile.ZipFile('data.zip')
        return z.read(filename).split('\n')

def loadDataFile(filename, n, width, height):
    fin = readlines(filename)
    fin.reverse()
    items = []
    for i in range(n):
        data = []
        for j in range(height):
            if len(fin) == 0:
                print("End of file reached at %d examples" % i)
                break
            line = fin.pop()
            data.append(list(line))
        if len(data) < height:
            break
        items.append(convertToValues(data, width, height))
    return items

def loadLabelsFile(filename, n):
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        labels.append(int(line))
    return labels

def convertToInteger(data):
    if type(data) != list:
        if data == ' ':
            return 0
        elif data == '+':
            return 1
        elif data == '#':
            return 2
        return 0
    else:
        return list(map(convertToInteger, data))

def convertToValues(data, width, height):
    data = convertToInteger(data)
    values = {}
    for y in range(height):
        for x in range(width):
            values[(x,y)] = data[y][x]
    from util import Counter
    return Counter(values)
