import util
import Classification

class PerceptronClassifier(Classification.Classification):
    def __init__(self, legalLabels, max_iterations, learning_rate):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate 
        self.weights = {label: util.Counter() for label in self.legalLabels} 

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        for i in range(len(trainingData)):
            datum = trainingData[i]  
            true_label = trainingLabels[i]  
            
            scores = util.Counter()
            for label in self.legalLabels:
                scores[label] = self.weights[label] * datum
            
            predicted_label = scores.argMax()

            if predicted_label != true_label:
                for feature, value in datum.items():
                    self.weights[true_label][feature] += self.learning_rate * value
                    self.weights[predicted_label][feature] -= self.learning_rate * value

    def classify(self, data):
        predictions = []
        for datum in data:
            scores = util.Counter()
            for label in self.legalLabels:
                scores[label] = self.weights[label] * datum
            predictions.append(scores.argMax())
        return predictions
