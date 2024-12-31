import numpy as np
import time
import samples
import util
from naivebayes import NaiveBayesClassifier
from perceptron import PerceptronClassifier
import matplotlib.pyplot as plt

def toBinaryFeatures(data, mode):
    binData = []
    for datum in data:
        binFeat = util.Counter()
        for f, val in datum.items():
            if mode == 'digit':
               # For Digits
                binFeat[f] = 1 if val > 0 else 0
            else:
                # For Faces
                binFeat[f] = 1 if val == 2 else 0
        binData.append(binFeat)
    return binData

def run_digit_experiment():
    numTraining = 5000
    numTest = 1000
    increments = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_runs = 5 

    trainingData = samples.loadDataFile("digitdata/trainingimages", numTraining, 28, 28)
    trainingLabels = samples.loadLabelsFile("digitdata/traininglabels", numTraining)
    testData = samples.loadDataFile("digitdata/testimages", numTest, 28, 28)
    testLabels = samples.loadLabelsFile("digitdata/testlabels", numTest)

    trainingData_bin = toBinaryFeatures(trainingData, mode='digit')
    testData_bin = toBinaryFeatures(testData, mode='digit')

    indices = np.arange(numTraining)

    nb_results = {}
    perc_results = {}

    for fraction in increments:
        nb_accuracies = []
        perc_accuracies = []
        nb_times = []
        perc_times = []

        for run in range(num_runs):
            np.random.shuffle(indices)
            subset_size = int(numTraining * fraction)
            subset_indices = indices[:subset_size]
            X_sub = [trainingData_bin[i] for i in subset_indices]
            y_sub = [trainingLabels[i] for i in subset_indices]

            nb = NaiveBayesClassifier(range(10))
            start = time.time()
            nb.train(X_sub, y_sub, [], [])
            end = time.time()
            nb_train_time = end - start
            nbPred = nb.classify(testData_bin)
            nbAcc = sum(nbPred[i] == testLabels[i] for i in range(numTest)) / float(numTest)
            nb_accuracies.append(nbAcc)
            nb_times.append(nb_train_time)

            perc = PerceptronClassifier(range(10), max_iterations=10, learning_rate=0.01)
            start = time.time()
            perc.train(X_sub, y_sub, [], [])
            end = time.time()
            perc_train_time = end - start
            percPred = perc.classify(testData_bin)
            percAcc = sum(percPred[i] == testLabels[i] for i in range(numTest)) / float(numTest)
            perc_accuracies.append(percAcc)
            perc_times.append(perc_train_time)

        nb_mean_acc = np.mean(nb_accuracies)
        nb_std_acc = np.std(nb_accuracies)
        nb_mean_time = np.mean(nb_times)

        perc_mean_acc = np.mean(perc_accuracies)
        perc_std_acc = np.std(perc_accuracies)
        perc_mean_time = np.mean(perc_times)

        nb_results[fraction] = (nb_mean_acc, nb_std_acc, nb_mean_time)
        perc_results[fraction] = (perc_mean_acc, perc_std_acc, perc_mean_time)

        print("Fraction: {:.0f}%".format(fraction*100))
        print("Naive Bayes -> Mean Acc: {:.2f}% ± {:.2f}, Mean Train Time: {:.4f}s".format(nb_mean_acc*100, nb_std_acc*100, nb_mean_time))
        print("Perceptron -> Mean Acc: {:.2f}% ± {:.2f}, Mean Train Time: {:.4f}s".format(perc_mean_acc*100, perc_std_acc*100, perc_mean_time))
        print("----------------------------------------------------")

    return nb_results, perc_results, numTraining

def run_face_experiment():
    numFaceTrain = 451
    numFaceTest = 150
    increments = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    num_runs = 5

    faceTrainingData = samples.loadDataFile("facedata/facedatatrain", numFaceTrain, 60, 70)
    faceTrainingLabels = samples.loadLabelsFile("facedata/facedatatrainlabels", numFaceTrain)
    faceTestData = samples.loadDataFile("facedata/facedatatest", numFaceTest, 60, 70)
    faceTestLabels = samples.loadLabelsFile("facedata/facedatatestlabels", numFaceTest)

    faceTrainingData_bin = toBinaryFeatures(faceTrainingData, mode='face')
    faceTestData_bin = toBinaryFeatures(faceTestData, mode='face')

    indices = np.arange(numFaceTrain)
    nb_results = {}
    perc_results = {}

    for fraction in increments:
        nb_accuracies = []
        perc_accuracies = []
        nb_times = []
        perc_times = []

        for run in range(num_runs):
            np.random.shuffle(indices)
            subset_size = int(numFaceTrain * fraction)
            subset_indices = indices[:subset_size]
            X_sub = [faceTrainingData_bin[i] for i in subset_indices]
            y_sub = [faceTrainingLabels[i] for i in subset_indices]

            nb = NaiveBayesClassifier([0,1])
            start = time.time()
            nb.train(X_sub, y_sub, [], [])
            end = time.time()
            nb_train_time = end - start
            nbPred = nb.classify(faceTestData_bin)
            nbAcc = sum(nbPred[i] == faceTestLabels[i] for i in range(numFaceTest)) / float(numFaceTest)
            nb_accuracies.append(nbAcc)
            nb_times.append(nb_train_time)

            perc = PerceptronClassifier([0,1], max_iterations=10, learning_rate=0.1)
            start = time.time()
            perc.train(X_sub, y_sub, [], [])
            end = time.time()
            perc_train_time = end - start
            percPred = perc.classify(faceTestData_bin)
            percAcc = sum(percPred[i] == faceTestLabels[i] for i in range(numFaceTest)) / float(numFaceTest)
            perc_accuracies.append(percAcc)
            perc_times.append(perc_train_time)

        nb_mean_acc = np.mean(nb_accuracies)
        nb_std_acc = np.std(nb_accuracies)
        nb_mean_time = np.mean(nb_times)

        perc_mean_acc = np.mean(perc_accuracies)
        perc_std_acc = np.std(perc_accuracies)
        perc_mean_time = np.mean(perc_times)

        nb_results[fraction] = (nb_mean_acc, nb_std_acc, nb_mean_time)
        perc_results[fraction] = (perc_mean_acc, perc_std_acc, perc_mean_time)

        print("Fraction: {:.0f}% (Face data)".format(fraction*100))
        print("Naive Bayes -> Mean Acc: {:.2f}% ± {:.2f}, Mean Train Time: {:.4f}s".format(nb_mean_acc*100, nb_std_acc*100, nb_mean_time))
        print("Perceptron -> Mean Acc: {:.2f}% ± {:.2f}, Mean Train Time: {:.4f}s".format(perc_mean_acc*100, perc_std_acc*100, perc_mean_time))
        print("----------------------------------------------------")

    return nb_results, perc_results, numFaceTrain

def plot_results(nb_results, perc_results, total_training, title="Results"):
    fractions = sorted(nb_results.keys())
    nb_mean_acc = [nb_results[f][0] for f in fractions]
    nb_std_acc = [nb_results[f][1] for f in fractions]
    nb_mean_time = [nb_results[f][2] for f in fractions]

    perc_mean_acc = [perc_results[f][0] for f in fractions]
    perc_std_acc = [perc_results[f][1] for f in fractions]
    perc_mean_time = [perc_results[f][2] for f in fractions]

    x_points = [int(f * total_training) for f in fractions]

    plt.figure(figsize=(10,6))
    plt.errorbar([f*100 for f in fractions], [a*100 for a in nb_mean_acc], yerr=[s*100 for s in nb_std_acc], fmt='o-', label='Naive Bayes Accuracy')
    plt.errorbar([f*100 for f in fractions], [a*100 for a in perc_mean_acc], yerr=[s*100 for s in perc_std_acc], fmt='o-', label='Perceptron Accuracy')
    plt.title(title + " Test Accuracy vs Training Data Fraction")
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot([f*100 for f in fractions], nb_mean_time, 'o-', label='Naive Bayes Training Time')
    plt.plot([f*100 for f in fractions], perc_mean_time, 'o-', label='Perceptron Training Time')
    plt.title(title + " Test Training Time vs Training Data Fraction")
    plt.xlabel("Training Data Used (%)")
    plt.ylabel("Training Time (s)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(x_points, nb_mean_time, 'o-', label='Naive Bayes Training Time')
    plt.plot(x_points, perc_mean_time, 'o-', label='Perceptron Training Time')
    plt.title(title + " Test Training Time vs Number of Training Samples")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Training Time (s)")
    plt.grid(True)
    plt.legend()
    plt.show()

    nb_mean_error = [1 - acc for acc in nb_mean_acc]
    perc_mean_error = [1 - acc for acc in perc_mean_acc]

    nb_error_std = nb_std_acc
    perc_error_std = perc_std_acc

    plt.figure(figsize=(10,6))
    plt.errorbar(x_points, [e*100 for e in nb_mean_error], yerr=[s*100 for s in nb_error_std], fmt='o-', label='Naive Bayes Error')
    plt.errorbar(x_points, [e*100 for e in perc_mean_error], yerr=[s*100 for s in perc_error_std], fmt='o-', label='Perceptron Error')
    plt.title(title + " Test Prediction Error vs Number of Training Samples")
    plt.xlabel("Number of Training Samples")
    plt.ylabel("Error (%)")
    plt.grid(True)
    plt.legend()
    plt.show()

def main():
    print("Running face experiment...")
    face_nb_results, face_perc_results, face_total = run_face_experiment()
    print("Running digit experiment...")
    digit_nb_results, digit_perc_results, digit_total = run_digit_experiment()

    plot_results(face_nb_results, face_perc_results, face_total, title="Face Classification")
    plot_results(digit_nb_results, digit_perc_results, digit_total, title="Digit Classification")

if __name__ == "__main__":
    main()
