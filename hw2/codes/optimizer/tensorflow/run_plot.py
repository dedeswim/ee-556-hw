import matplotlib.pyplot as plt
import pickle
import os

outdir='output'
if not os.path.exists(outdir):
    os.makedirs(outdir)

os.system('python main.py --optimizer sgd --learning_rate 1e-2 --output='+outdir+'/sgd.pkl')
os.system('python main.py --optimizer momentumsgd --learning_rate 1e-2 --output='+outdir+'/momentumsgd.pkl')
os.system('python main.py --optimizer rmsprop --learning_rate 1e-2 --output='+outdir+'/rmsprop.pkl')
os.system('python main.py --optimizer adam --learning_rate 1e-2 --output='+outdir+'/adam.pkl')
os.system('python main.py --optimizer adagrad --learning_rate 1e-2 --output='+outdir+'/adagrad.pkl')
optimizers = ['sgd', 'momentumsgd', 'rmsprop', 'adam', 'adagrad']

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
