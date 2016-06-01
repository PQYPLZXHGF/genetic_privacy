from random import shuffle, randrange
from collections import Counter

class RandomQueue:
    def __init__(self, initial = None):
        if initial is not None:
            self.elements = list(initial)
            shuffle(self.elements)
        else:
            self.elements = []

    def enqueue(self, item):
        self.elements.append(item)

    def dequeue(self):
        if len(self.elements) > 1:
            location = randrange(0, len(self.elements))
            temp = self.elements[-1]
            self.elements[-1] = self.elements[location]
            self.elements[location] = temp
        return self.elements.pop()

    def __repr__(self):
        return "RandomQueue({})".format(repr(self.elements))

def test_random_queue():
    counts = Counter()
    trials = 1000000
    for _ in range(trials):
        queue = RandomQueue()
        for i in range(3):
            queue.enqueue(i)
        counts[queue.dequeue()] += 1
    return {key: count / trials for key, count in counts.items()}
