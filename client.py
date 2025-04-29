from collections import OrderedDict
import argparse
import warnings
import traceback
import os
import time
import gc

import flwr as fl
from numpy.core.fromnumeric import trace
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import pickle as pkl
import numpy as np
import utils
from config import get_config_dict
from models import create_backbone, simclr, simsiam, byol, specloss, rotpred, orchestra, mymodel
import copy

warnings.filterwarnings("ignore", category=UserWarning)
cudnn.deterministic = True
cudnn.benchmark = False


##### Train functions
# Supervised training
def sup_train(net, trainloader, epochs, lr, device=None):
    net.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()


# SSL trainer
def ssl_train(net, trainloader, train_mode, epochs, lr, global_net=None, device=None, is_orchestra=False):
    net.train()
    if global_net:
        global_net.eval()

    # Random dataloader for relational loss
    random_loader = copy.deepcopy(trainloader)
    random_dataloader = iter(random_loader)

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    # First round of Orchestra only performs local clustering, no training, to initialize global centroids
    if((net.rounds_done[0]==0) and is_orchestra): 
        # Initializing Memory
        net.reset_memory(trainloader, device=device)
        net.local_clustering(device=device)
        return -1

    epoch_loss_collector = []
    for i in range(epochs):
        for batch_idx, ((data1, data2), labels) in enumerate(trainloader):
            input1 = data1.to(device)
            if(is_orchestra):
                input2, input3, deg_labels = data2[0].to(device), data2[1].to(device), data2[2].to(device)
            else:
                input2, input3, deg_labels = data2.to(device), None, None


            try:
                (random_x, _), _ = next(random_dataloader)
            except:
                random_dataloader = iter(random_loader)
                (random_x, _), _ = next(random_dataloader)
            random_x = random_x.to(device)

            optimizer.zero_grad()
            
            
            if train_mode == "custom":
                loss_l, tZ1_local, tZ2_local = net(x1=input1, x2=input2, x3=input3, deg_labels=deg_labels, random_x=random_x, is_global=False)
                loss_g = 0
                loss_g = global_net(x1=input1, x2=input2, x3=input3, deg_labels=deg_labels, tZ1_local=tZ1_local, tZ2_local=tZ2_local, random_x=random_x, is_global=True)
                loss = loss_l + loss_g
            elif train_mode == "orchestra":
                loss_l = net(input1, input2, input3, deg_labels)
                loss = loss_l
            else:
                loss_l = net(input1, input2, input3, deg_labels)
                loss = loss_l

            epoch_loss_collector.append(loss.item())
            loss.backward()
            optimizer.step()
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        print("Epoch: ", i, "Loss: ", epoch_loss)
    print("end training")

    if(is_orchestra):
        net.local_clustering(device=device)


# Rotation prediction trainer
def rot_train(net, trainloader, epochs, lr, device=None):
    net.train()

    # Optimizer
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for _ in range(epochs):

        for batch_idx, ((input1, angles), labels) in enumerate(trainloader):
            input1, angles = input1.to(device), angles.to(device)
            optimizer.zero_grad()
            loss = net(input1, angles)
            loss.backward()
            optimizer.step()


#### Client definitions
def make_client(cid, device=None, stateless=True, config_dict=None):
    print("making client")
    try:
        gc.collect()
        torch.cuda.empty_cache()    
        client_id = int(cid) # cid is of type str when using simulation

        if device is None:
            print("Client {} CUDA_VISIBLE_DEVICES: {}".format(cid, os.environ["CUDA_VISIBLE_DEVICES"]))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model_save_path = config_dict["save_dir"]+"/saved_models/"+config_dict["dataset"]+"_client_"+str(client_id)+".pth"

        ##### Create model
        if config_dict["dataset"]=="CIFAR10":
            n_classes = 10
        elif  config_dict["dataset"]=="ACI_IoT":
            n_classes = 12
        elif  config_dict["dataset"]=="UNSW_IoT":
            n_classes = 10
        else:
            n_classes = 100

        ##### Load data
        trainloader, memloader, testloader = utils.load_data(config_dict, client_id=client_id, n_clients=config_dict['num_clients'], alpha=config_dict['alpha'],
                                                                bsize=config_dict["local_bsize"], in_simulation=config_dict["virtualize"])
        
        


        # Define model; for SSL, projector/predictor will also be needed
        if(config_dict["train_mode"]=="sup"):
            net = create_backbone(name=config_dict["model_class"], num_classes=n_classes, block=config_dict["block"])
            net = net.to(device)

        else:
            if(config_dict["train_mode"]=="simclr"):
                net = simclr(config_dict=config_dict, bbone_arch=config_dict["model_class"]) 
            elif(config_dict["train_mode"]=="simsiam"):
                net = simsiam(config_dict=config_dict, bbone_arch=config_dict["model_class"])
            elif(config_dict["train_mode"]=="byol"):
                net = byol(config_dict=config_dict, bbone_arch=config_dict["model_class"]) 
            elif(config_dict["train_mode"]=="specloss"):
                net = specloss(config_dict=config_dict, bbone_arch=config_dict["model_class"]) 
            elif(config_dict["train_mode"]=="rotpred"):
                net = rotpred(config_dict=config_dict, bbone_arch=config_dict["model_class"])
            elif(config_dict["train_mode"]=="orchestra"):
                net = orchestra(config_dict=config_dict, bbone_arch=config_dict["model_class"])
            elif(config_dict["train_mode"]=="custom"):
                net = mymodel(config_dict=config_dict, bbone_arch=config_dict["model_class"])
                global_net = mymodel(config_dict=config_dict, bbone_arch=config_dict["model_class"])
                global_net = global_net.to(device)
            net = net.to(device)
            


        ##### Flower client
        class flclient(fl.client.NumPyClient):
            def get_parameters(self):
                return [val.cpu().numpy() for _, val in net.state_dict().items()]

            def set_parameters(self, parameters):
                params_dict = zip(net.state_dict().keys(), parameters)
                if(config_dict['stateful_client']):
                    state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict if ('mem_projections' not in k and 'target_' not in k)})
                else:
                    state_dict = OrderedDict({k: torch.Tensor(np.array([v])) if (v.shape == ()) else torch.Tensor(v) for k, v in params_dict if ('mem_projections' not in k)})

                net.load_state_dict(state_dict, strict=False)
                if(config_dict["train_mode"]=="custom"):
                    global_net.load_state_dict(state_dict, strict=False)

            def fit(self, parameters, config):
                try:
                    self.set_parameters(parameters)

                    # Supervised training
                    if(config_dict["train_mode"]=="sup"):
                        sup_train(net, trainloader, epochs=config_dict["local_epochs"], lr=config_dict["local_lr"], device=device)

                    # SSL training
                    else:
                        if(config_dict['train_mode']=='rotpred'):
                            rot_train(net, trainloader, epochs=config_dict["local_epochs"], lr=config_dict['local_lr'], device=device)
                        elif(config_dict["train_mode"]=="custom"):
                            ssl_train(net, trainloader, config_dict["train_mode"], config_dict["local_epochs"], config_dict['local_lr'], global_net=global_net, device=device, is_orchestra=True)
                        else:
                            ssl_train(net, trainloader, config_dict["train_mode"], config_dict["local_epochs"], config_dict['local_lr'], device=device, is_orchestra=(config_dict["train_mode"] == "orchestra"))

                    return self.get_parameters(), len(trainloader), {}
                except Exception as e:
                    print(f"Client {cid} - Exception in client fit {e}")
                    print(f"Client {cid}", traceback.format_exc())

            def evaluate(self, parameters, config):
                print(">>>>>>>>>>>>>>>>>>>>>>> client")
                self.set_parameters(parameters)
                if(config_dict["train_mode"]=="sup"):
                    loss, accuracy = utils.test(net, testloader, device=device, verbose=False)
                    return float(loss), len(testloader), {"accuracy": float(accuracy)}
                else:
                    accuracy = utils.knn_monitor(net.backbone, memloader, testloader, verbose=False, device=device)
                    return float(0), len(testloader), {"accuracy": float(accuracy)}

            def save_net(self):
                ##### Save local model
                state = {'net': net.state_dict()}
                torch.save(state, model_save_path)
                print(f"Client: {client_id} Saving network to {model_save_path}")
            
        gc.collect()
        torch.cuda.empty_cache()
        return flclient()
    except Exception as e:
        print(f"Client {cid} - Exception in make_client {e}")
        print(f"Client {cid}", traceback.format_exc())

##### Federation of the pipeline with Flower
def main(config_dict):
    """Create model, load data, define Flower client, start Flower client."""
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--client_id", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.client_id%8) if torch.cuda.is_available() else "cpu")
    local_client = make_client(args.client_id, device=device, stateless=True, config_dict=config_dict)

    ##### Start client
    fl.client.start_numpy_client("[::]:9081", client=local_client)

    local_client.save_net()

if __name__ == "__main__":
    config_dict = get_config_dict()
    torch.manual_seed(config_dict['seed'])

    main(config_dict)