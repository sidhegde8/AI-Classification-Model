import math
import util
import Classification

class NaiveBayesClassifier(Classification.Classification):

    def __init__(self, legalLabels, smoothing=1):
        self.legalLabels = legalLabels
        self.type = "naivebayes"
        self.smoothing = smoothing
        self.prior = util.Counter()
        self.featureProb = util.Counter()
        self.features = []

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        if len(trainingData) == 0:
            raise Exception("No training data provided")

        allFeatures = set()
        for datum in trainingData:
            for f in datum.keys():
                allFeatures.add(f)
        self.features = list(allFeatures)

        labelCounts = util.Counter()
        featureOnCounts = util.Counter()
        featureTotalCounts = util.Counter()

        for i, datum in enumerate(trainingData):
            label = trainingLabels[i]
            labelCounts[label] += 1
            for f in self.features:
                val = datum[f]
                featureTotalCounts[(f, label)] += 1
                if val > 0:
                    featureOnCounts[(f, label)] += 1

        totalSamples = sum(labelCounts.values())
        for lbl in self.legalLabels:
            self.prior[lbl] = float(labelCounts[lbl]) / totalSamples
        k = self.smoothing
        for lbl in self.legalLabels:
            labelTotal = labelCounts[lbl]
            for f in self.features:
                numerator = featureOnCounts[(f, lbl)] + k
                denominator = labelTotal + 2 * k 
                self.featureProb[(f, lbl)] = float(numerator) / denominator

    def classify(self, testData):
        predictions = []
        for datum in testData:
            logProbs = self.computeLabelLogProbs(datum)
            predictions.append(logProbs.argMax())
        return predictions

    def computeLabelLogProbs(self, datum):
        logJoint = util.Counter()
        for lbl in self.legalLabels:
            log_val = math.log(self.prior[lbl])
            for f in self.features:
                p = self.featureProb[(f, lbl)]
                val = datum[f]
                if val > 0:
                    log_val += math.log(p)
                else:
                    log_val += math.log(1 - p)
            logJoint[lbl] = log_val
        return logJoint
