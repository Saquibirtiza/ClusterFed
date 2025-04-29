import numpy as np
import argparse
import torch
from torchvision import datasets, transforms
import pickle as pkl
import os, shutil
import utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset

parser = argparse.ArgumentParser(description="Sample data for clients")
parser.add_argument("--dataset", default="ACI_IoT", choices=["CIFAR10", "CIFAR100, ACI_IoT"])
parser.add_argument("--n_clients", type=int, default=20)
parser.add_argument("--alpha", type=float, default=1e-1, choices=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
parser.add_argument("--use_IID", type=str, default='False', choices=['True', 'False'])
parser.add_argument("--use_balance", type=str, default='False', choices=['True', 'False'])
parser.add_argument("--input_dir", default="/home/sxi190002/Saquib/Federated Learning/orchestra/data/ACI-IoT-2023.csv")
parser.add_argument("--init_seed", type=int, default=3, help="Random seed")
parser.add_argument("--data_dir", default="./data")
parser.add_argument("--test_ratio", type=float, default="0.3")

args = parser.parse_args()
args.use_IID = (args.use_IID=='True')
args.use_balance = (args.use_balance=='True')

torch.manual_seed(args.init_seed)
np.random.seed(args.init_seed)

os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}', exist_ok=True)
os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/train', exist_ok=True)
os.makedirs(f'{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/test', exist_ok=True)

##### Print setup to confirm things are correct
print("\nSampling configuration:")
print("\tDataset:", args.dataset)
print("\tNumber of clients:", args.n_clients)
print("\tDistribute IID:", args.use_IID)
print("\tCreate balanced partitions:", args.use_balance)
print("\tWriting data at this location: ", args.data_dir + "/" + str(args.n_clients))
if(not args.use_IID):
    print("\tAlpha for Dirichlet distribution:", args.alpha)
print("\n")

##### Determine number of samples in dataset
class ACI_IoT_Dataset(Dataset):
    def __init__(self, X, y):
        # Convert data to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.int64)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Return the feature tensor and the target tensor for each sample
        return self.X[idx], self.targets[idx]

def load_ACI_IoT_data(inputdir, num_Class, test_ratio):
    df = pd.read_csv(inputdir)

    encoder = LabelEncoder()
    df['Connection Type'] = encoder.fit_transform(df['Connection Type'])  
    df['Flow Packets/s'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df['Flow Bytes/s'].replace([np.inf, -np.inf], np.nan, inplace=True)  
    df = df.drop(columns=["Flow ID", "Timestamp", "Src IP", "Dst IP"])
    y = df.iloc[:, -2]
    X = df.drop(columns='Label')

    # Apply normalization using StandardScaler
    scaler = StandardScaler()
    imputer = SimpleImputer(strategy='mean')  # To handle any NaN values
    X = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(X)), columns=X.columns)
    features = X.columns
    X = X.to_numpy()

    # Encode the Labels 
    encoder.fit(y) 
    y = encoder.transform(y)
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=30000, stratify=y, random_state=42)
    X_trainval, _, y_trainval, _ = train_test_split(X_trainval, y_trainval, train_size=70000, stratify=y_trainval, random_state=42)

    # Create PyTorch datasets
    trainval_dataset = ACI_IoT_Dataset(X_trainval, y_trainval)
    test_dataset = ACI_IoT_Dataset(X_test, y_test)

    return trainval_dataset, test_dataset

if(args.dataset=="CIFAR10"):
    n_classes = 10
    train_data = datasets.CIFAR10(f'{args.data_dir}/dataset/CIFAR10', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.CIFAR10(f'{args.data_dir}/dataset/CIFAR10', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
elif(args.dataset=="CIFAR100"):
    n_classes = 100
    train_data = datasets.CIFAR100(f'{args.data_dir}/dataset/CIFAR100', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    test_data = datasets.CIFAR100(f'{args.data_dir}/dataset/CIFAR100', train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
elif(args.dataset=="ACI_IoT"):
    n_classes = 12
    train_data, test_data = load_ACI_IoT_data(args.input_dir, n_classes, args.test_ratio)
else:
    raise Exception("Dataset not recognized")
n_samples_train = len(train_data)
n_samples_test = len(test_data)

#### Determine locations of different classes
all_ids_train = np.array(train_data.targets)
class_ids_train = {class_num: np.where(all_ids_train-1 == class_num)[0] for class_num in range(n_classes)}
all_ids_test = np.array(test_data.targets)
class_ids_test = {class_num: np.where(all_ids_test-1 == class_num)[0] for class_num in range(n_classes)}





##### Determine distribution over classes to be assigned per client
# Returns n_clients x n_classes matrix
n_clients = args.n_clients
if(args.use_IID):
    args.alpha = 1e5



min_size = 0
min_require_size = 10
n_clients = args.n_clients

N = len(train_data.targets)

while min_size < min_require_size:
    all_proportions = []
    idx_batch = [[] for _ in range(n_clients)]
    for k in range(n_classes):
        idx_k = np.where(all_ids_train-1 == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(args.alpha, n_clients))
        proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_ids_train[k])).astype(int)
    

        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        all_proportions.append(proportions)

dist_of_client = np.array(all_proportions).transpose()
samples_per_class_train = [dist_of_client[0]]
for i in range(1,len(dist_of_client)):
    samples_per_class_train.append(dist_of_client[i] - dist_of_client[i-1])

samples_per_class_train = np.array(samples_per_class_train)



min_size = 0
min_require_size = 10
n_clients = args.n_clients

N = len(test_data.targets)
net_dataidx_map = {}


while min_size < min_require_size:
    all_proportions = []
    idx_batch = [[] for _ in range(n_clients)]
    for k in range(n_classes):
        idx_k = np.where(all_ids_test-1 == k)[0]
        np.random.shuffle(idx_k)
        proportions = np.random.dirichlet(np.repeat(args.alpha, n_clients))
        proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_ids_test[k])).astype(int)
        
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_j) for idx_j in idx_batch])
        all_proportions.append(proportions)

dist_of_client = np.array(all_proportions).transpose()
samples_per_class_test = [dist_of_client[0]]
for i in range(1,len(dist_of_client)):
    samples_per_class_test.append(dist_of_client[i] - dist_of_client[i-1])


samples_per_class_test = np.array(samples_per_class_test)


#### Run OT if using balanced partitioning
if(args.use_balance):
    for i in range(100):
        s0 = dist_of_client.sum(axis=0, keepdims=True)
        s1 = dist_of_client.sum(axis=1, keepdims=True)
        dist_of_client /= s0
        dist_of_client /= s1

start_ids_train = np.zeros((n_clients+1, n_classes), dtype=np.int32)
start_ids_test = np.zeros((n_clients+1, n_classes), dtype=np.int32)
for i in range(0, n_clients):
    start_ids_train[i+1] = start_ids_train[i] + samples_per_class_train[i]
    start_ids_test[i+1] = start_ids_test[i] + samples_per_class_test[i]

# Sanity checks
print("\nSanity checks:")
print("\tSum of dist. of classes over clients: {}".format(dist_of_client.sum(axis=0)))
print("\tSum of dist. of clients over classes: {}".format(dist_of_client.sum(axis=1)))
print("\tTotal trainset size: {}".format(samples_per_class_train.sum()))
print("\tTotal testset size: {}".format(samples_per_class_test.sum()))


##### Save IDs
# Train
client_ids = {client_num: {} for client_num in range(n_clients)}
for client_num in range(n_clients):
    l = np.array([], dtype=np.int32)
    for class_num in range(n_classes):
        start, end = start_ids_train[client_num, class_num], start_ids_train[client_num+1, class_num]
        l = np.concatenate((l, class_ids_train[class_num][start:end].tolist())).astype(np.int32)
    client_ids[client_num] = l
    


print("\nDistribution over classes:")
for client_num in range(n_clients):
    with open(f"{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/train/"+args.dataset+"_"+str(client_num)+'.pkl', 'wb') as f:
        pkl.dump(client_ids[client_num], f)
    print("\tClient {cnum}: \n \t\t Train: {cdisttrain} \n \t\t Total: {traintotal} \n \t\t Test: {cdisttest} \n \t\t Total: {testtotal}".format(
        cnum=client_num, cdisttrain=samples_per_class_train[client_num].astype(int), cdisttest=samples_per_class_test[client_num].astype(int), 
        traintotal=samples_per_class_train[client_num].astype(int).sum(), testtotal=samples_per_class_test[client_num].astype(int).sum()))

# Test
client_ids = {client_num: {} for client_num in range(n_clients)}
for client_num in range(n_clients):
    l = np.array([], dtype=np.int32)
    for class_num in range(n_classes):
        start, end = start_ids_test[client_num, class_num], start_ids_test[client_num+1, class_num]
        l = np.concatenate((l, class_ids_test[class_num][start:end].tolist())).astype(np.int32)
    client_ids[client_num] = l

for client_num in range(n_clients):
    with open(f"{args.data_dir}/{args.n_clients}/{args.alpha}/{args.dataset}/test/"+args.dataset+"_"+str(client_num)+'.pkl', 'wb') as f:
        pkl.dump(client_ids[client_num], f)
