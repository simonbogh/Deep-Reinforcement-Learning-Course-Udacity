import numpy as np
import pickle
import matplotlib as mpl
 # Set backend for image rendering, else the import will give an error in MacOS
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pylab


# Plot score from Pickle files
# Finally the achived scores are saved to a file using the 'Pickle'
# framework. The scores can be used later on to e.g. plot graphs.
# file_name = "scores"                    # Filename
# file_object = open(file_name, 'wb')     # Open the with writing rights (w)
# pickle.dump(scores, file_object)        # Save scores to file
# file_object.close()                     # Close the file
scores_1 = pickle.load(open("models_trained/model_32x32/scores.p", "rb"))
scores_2 = pickle.load(open("models_trained/model_64x64/scores.p", "rb"))
scores_2b = pickle.load(open("models_trained/model_64x64_lr_0.005/scores.p", "rb"))
scores_2c = pickle.load(open("models_trained/model_64x64_lr_0.0005/scores.p", "rb"))
scores_3 = pickle.load(open("models_trained/model_128x64/scores.p", "rb"))
scores_4 = pickle.load(open("models_trained/model_128x128/scores.p", "rb"))
scores_5 = pickle.load(open("models_trained/model_256x256/scores.p", "rb"))


# Plot the scores
# Simple data to display in various forms
x = np.arange(len(scores_1))
y = np.sin(x ** 2)

# # Plot 2x2 subplots
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# ax1.plot(x, scores_1)
# ax1.set_title('32x32')
# ax2.plot(x, scores_3)
# ax2.set_title('64x64')
# ax3.plot(x, scores_3)
# ax3.set_title('128x64')
# ax4.plot(x, scores_4)
# ax4.set_title('128x128')
# plt.show()

fig = plt.figure()
#plt.plot(np.arange(len(scores_1)), scores_1)

# Moving average using convolve
# smoothed = np.convolve(scores_1, np.ones(25)/25)
# plt.plot(np.arange(len(smoothed)), smoothed)

# Moving average using comsum, should be faster than convolve
window_width = 50
# cumsum_vec = np.cumsum(np.insert(scores_1, 0, 0))
# ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
# pylab.plot(np.arange(len(ma_vec)), ma_vec, label='32x32')

cumsum_vec = np.cumsum(np.insert(scores_2b, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
pylab.plot(np.arange(len(ma_vec)), ma_vec, label='64x64 lr 0.005')

cumsum_vec = np.cumsum(np.insert(scores_2, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
pylab.plot(np.arange(len(ma_vec)), ma_vec, label='64x64 lr 0.001')

cumsum_vec = np.cumsum(np.insert(scores_2c, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
pylab.plot(np.arange(len(ma_vec)), ma_vec, label='64x64 lr 0.0005')

# cumsum_vec = np.cumsum(np.insert(scores_3, 0, 0))
# ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
# pylab.plot(np.arange(len(ma_vec)), ma_vec, label='128x64')

# cumsum_vec = np.cumsum(np.insert(scores_4, 0, 0))
# ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
# pylab.plot(np.arange(len(ma_vec)), ma_vec, label='128x128')

# cumsum_vec = np.cumsum(np.insert(scores_5, 0, 0))
# ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
# pylab.plot(np.arange(len(ma_vec)), ma_vec, label='256x256')

# plt.plot(np.arange(len(scores_1)), scores_2)
# plt.plot(np.arange(len(scores_1)), scores_3)
# plt.plot(np.arange(len(scores_1)), scores_4)
pylab.title('Scores over time for different learning rates')
pylab.ylabel('Score\n(moving average)')
pylab.xlabel('Episode #')
pylab.legend(loc='upper left')
pylab.show()


# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores_1)), scores_1)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
