from collections import Counter
from random import sample
from itertools import combinations
from unidecode import unidecode
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

compare = lambda x, y: Counter(x) == Counter(y)

def kmers(file, k=3):
    for i in range(len(file) - k + 1):
        yield file[i:i + k]

class KMedoidsClustering:
    def __init__(self, folder, files):
        path = folder
        self.files = files
        self.langs = {}
        self.clusters = {}
        self.all_clusters = {}
        self.new_candidates = {}
        self.prev_start = []
        self.part_silhouettes = []
        self.all_silhouettes = []
        self.all_similarities = {}
        self.indexes = {}

        for x in files:
            path += x
            path += ".txt"
            file = unidecode(open(path, "rt", encoding="utf8").read().replace("\n", " ").replace("  ", " ").lower())
            self.langs[x] = Counter(kmers(file, 3))
            path = path[:6]

        tmp = {(x,y): self.cosine_similarity(self.langs[x], self.langs[y]) for x, y in combinations(self.langs.keys(), 2)}

        for (x,y) in tmp.keys():
            self.all_similarities[(y,x)] = tmp[(x,y)]

        self.all_similarities.update(tmp)

    def cosine_similarity(self, v1, v2):
        size_t1 = np.sqrt(np.sum(np.square(list(v1.values()))))
        size_t2 = np.sqrt(np.sum(np.square(list(v2.values()))))

        in_both = v1.keys() & v2.keys()

        scal_product = sum(v1[i] * v2[i] for i in in_both)

        sim = scal_product / (size_t1 * size_t2)

        return sim

    def k_medoids(self, iteration, start, i):
        if not (i == 500 | compare(start, self.prev_start)):
            self.clusters = {x: [x] for x in start}

            for l in self.langs.keys():
                if l not in start:
                    _, p = max((self.all_similarities[(l,p)], p) for p in start)
                    self.clusters[p].append(l)

            self.prev_start = list(start)
            self.new_candidates = {}
            start.clear()

            # new medoids
            for vals in self.clusters.values():
                self.new_candidates.clear()
                for v in vals:
                    self.new_candidates[v] = sum(self.all_similarities[(v,p)] for p in vals if (p != v))
                new_medoid = max(self.new_candidates, key=self.new_candidates.get)
                start.append(new_medoid)


            self.k_medoids(iteration, start, i+1)

        else:
            self.all_clusters[iteration] = dict(self.clusters)
            self.silhouette()


    def silhouette(self):
        dist = []
        for vals in self.clusters.values():
                for v in vals:
                    if len(vals) > 1:
                        a = sum((1-self.all_similarities[(v,p)]) for p in vals if (p != v))/(len(vals)-1)

                        for c in self.clusters.values():
                            if set(c) != set(vals):
                                x = sum((1-self.all_similarities[(v,q)]) for q in c)/(len(c))
                                dist.append(x)

                        b = min(dist)
                        self.part_silhouettes.append(((b-a)/max(a,b)))
                    else:
                        #only one lang in cluster
                        self.part_silhouettes.append(0)

        self.all_silhouettes.append(sum(self.part_silhouettes)/len(self.part_silhouettes))

        self.part_silhouettes.clear()
        dist.clear()

    def draw_silhouettes(self):
        plt.figure()
        axes = plt.gca()
        axes.set_xlim([-1,1])
        axes.set_ylim(0, 100)

        self.all_silhouettes.sort()

        for i in range(100):
            x = self.all_silhouettes[i]
            axes.plot((0, x), (i, i), 'b', linewidth=1.5)

        axes.plot((0,0), (0, 100), 'k', linewidth=0.8)


        plt.show()

    def recognize_language(self, file):
        path = 'sample\\'
        path += file

        f = unidecode(open(path, "rt", encoding="utf8").read().replace("\n", " ").replace("  ", " ").lower())
        t = Counter(kmers(f, 3))

        #cosine similarity with all langs
        tuples = [(x, self.cosine_similarity(t,self.langs[x])) for x in self.langs.keys()]

        t_sorted = sorted(tuples, key=lambda x: x[1], reverse=True)

        s = sum(x[1] for x in t_sorted[:3])

        #top 3 probabilities
        probabilities = [(x[0], x[1]/s) for x in t_sorted[:3]]
        print(probabilities)

    def cosine_distance(self, i1, i2):
        v1 = self.langs[self.indexes[i1[0]]]
        v2 = self.langs[self.indexes[i2[0]]]

        dist = 1 - self.cosine_similarity(v1, v2)

        return dist

    def hierarchical(self):
        c = [[x] for x in range(22)]

        self.indexes = {x: self.files[x] for [x] in c}

        labels = ["bosanščina (latinica)", "bolgarščina", "češčina", "danščina", "angleščina", "španščina",
                  "estonščina", "finščina", "francoščina", "nemščina", "grščina", "madžarščina", "italijanščina",
                  "makedonščina", "norveščina", "poljščina", "portugalščina", "romunščina", "srbščina (latinica)",
                  "srbščina (cirilica)", "slovaščina", "slovenščina"]

        Z= linkage(c, method='average', metric=self.cosine_distance)

        plt.figure()
        dendrogram(Z, orientation="right", labels=labels)
        plt.show()


files = ["src1", "blg", "czc", "dns", "eng", "spn", "est", "fin", "frn", "ger", "grk", "hng", "itn", "mkj", "nrn",
         "pql", "por", "rum", "src3", "src5", "slo", "slv"]

# novice:
# folder = 'other\\'

folder = 'ready\\'
kmc = KMedoidsClustering(folder, files)
all_starts = {}


for i in range(100):
    start = sample(files, 5)
    all_starts[i] = start[:]
    kmc.k_medoids(i, start, 0)


print(min(kmc.all_silhouettes))
print('start')
print(all_starts[kmc.all_silhouettes.index(min(kmc.all_silhouettes))])
print(kmc.all_clusters[kmc.all_silhouettes.index(min(kmc.all_silhouettes))])

print(max(kmc.all_silhouettes))
print('start')
print(all_starts[kmc.all_silhouettes.index(max(kmc.all_silhouettes))])
print(kmc.all_clusters[kmc.all_silhouettes.index(max(kmc.all_silhouettes))])



kmc.draw_silhouettes()

# language recognition

#sample = ["src1", "czc", "eng", "est", "frn", "ger", "itn", "por", "spn", "slv"]
# for i in sample:
#     i += ".txt"
#     print(i)
#
#     kmc.recognize_language(i)


# hierarchical clustering:

#kmc.hierarchical()


