import matplotlib.pyplot as plt
import pickle
import os

outdir = 'output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

print("Training with SGD")
os.system('python main.py --optimizer sgd --learning_rate 1e-5 --output='+outdir+'/sgd.pkl')
print("Training with MomentumSGD")
os.system('python main.py --optimizer momentumsgd --learning_rate 1e-5 --output='+outdir+'/momentumsgd.pkl')
print("Training with RMSProp")
os.system('python main.py --optimizer rmsprop --learning_rate 1e-5 --output=' + outdir+'/rmsprop.pkl')
print("Training with ADAM")
os.system('python main.py --optimizer adam --learning_rate 1e-5 --output='+outdir+'/adam.pkl')
print("Training with AdaGrad")
os.system('python main.py --optimizer adagrad --learning_rate 1e-5 --output='+outdir+'/adagrad.pkl')
optimizers = ['sgd', 'momentumsgd', 'adagrad', 'rmsprop', 'adam']

# Plots the training losses.
for optimizer in optimizers:
    data = pickle.load(open(outdir+'/'+optimizer+".pkl", "rb"))
    plt.plot(data['train_loss'], label=optimizer)
plt.ylabel('Trainig loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('loss.pdf')
plt.show()

# Plots the training accuracies.
for optimizer in optimizers:
    data = pickle.load(open(outdir+'/'+optimizer+".pkl", "rb"))
    plt.plot(data['train_accuracy'], label=optimizer)
plt.ylabel('Trainig accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('accuracy.pdf')
plt.show()
