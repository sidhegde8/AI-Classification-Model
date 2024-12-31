
import sys
import inspect
import heapq, random

def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)

class Counter(dict):

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        for key in keys:
            self[key] += count

    def argMax(self):
        if len(self.keys()) == 0: return None
        all_items = list(self.items())
        values = [x[1] for x in all_items]
        maxIndex = values.index(max(values))
        return all_items[maxIndex][0]

    def sortedKeys(self):
        sortedItems = self.items()
        sortedItems.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        return sum(self.values())

    def normalize(self):
        total = float(self.totalCount())
        if total == 0:
            return
        for key in self.keys():
            self[key] = self[key] / total

    def copy(self):
        return Counter(dict.copy(self))

    def __mul__(self, y):
        sum_ = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key in y:
                sum_ += x[key] * y[key]
        return sum_

    def __add__(self, y):
        addend = Counter()
        for key in self:
            addend[key] = self[key]
        for key in y:
            addend[key] += y[key]
        return addend

    def __sub__(self, y):
        addend = Counter()
        for key in self:
            addend[key] = self[key]
        for key in y:
            addend[key] -= y[key]
        return addend

def arrayInvert(array):
    result = [[] for i in array]
    for outer in array:
        for inner in range(len(outer)):
            result[inner].append(outer[inner])
    return result
