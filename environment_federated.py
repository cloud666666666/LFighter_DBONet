from __future__ import print_function
from lib2to3.pgen2.tokenize import tokenize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from torch.cuda.amp import autocast, GradScaler
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from models import *
from utils import *
from sampling import *
from datasets import *
import os
import random
from dbo_cluster import DBOClusterer
from tqdm import tqdm_notebook
import copy
from operator import itemgetter
import time
from random import shuffle
from aggregation import *
from IPython.display import clear_output
import gc
from sklearn.metrics import confusion_matrix, classification_report
import torch
try:
    GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
except:
    GPU_NAME = "Unknown"

USE_AMP = torch.cuda.is_available() and "A100" in GPU_NAME

print(f"[AMP] 当前设备: {GPU_NAME}, 是否启用 AMP: {USE_AMP}")
class Peer():
    # Class variable shared among all the instances
    _performed_attacks = 0
    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self,val):
        type(self)._performed_attacks = val

    def __init__(self, peer_id, peer_pseudonym, local_data, labels, criterion, 
                device, local_epochs, local_bs, local_lr, 
                local_momentum, peer_type = 'honest'):

        self.peer_id = peer_id
        self.peer_pseudonym = peer_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.peer_type = peer_type
        self.USE_AMP = USE_AMP
        self.scaler = GradScaler(enabled=self.USE_AMP)
    #======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type = 'no_attack', malicious_behavior_rate = 0, 
                            source_class = None, target_class = None, dataset_name = None) :
        optimizer = torch.optim.SGD(model.parameters(), lr=self.local_lr, momentum=self.local_momentum)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
        epochs = self.local_epochs
        train_loader = DataLoader(self.local_data, self.local_bs, shuffle = True, drop_last=True)
        attacked = 0
        #Get the poisoned training data of the peer in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.peer_type == 'attacker'):
            r = np.random.random()
            if r <= malicious_behavior_rate:
                if dataset_name != 'IMDB':
                    poisoned_data = label_filp(self.local_data, source_class, target_class)
                    train_loader = DataLoader(
                        poisoned_data,
                        batch_size=self.local_bs,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0,  # ✅ 多线程加载
                        pin_memory=True  # ✅ 固定内存（GPU加速传输）
                    )

                self.performed_attacks+=1
                attacked = 1
                # print('Label flipping attack launched by', self.peer_pseudonym, 'to flip class ', source_class,
                # ' to class ', target_class)
        lr=self.local_lr

        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                        momentum=self.local_momentum, weight_decay=5e-4)
        model.train()
        epoch_loss = []
        peer_grad = []
        t = 0
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                data.requires_grad = True

                if dataset_name == 'IMDB':
                    target = target.view(-1).float()
                else:
                    target = target.view(-1).long()

                first_activation = None

                def get_first_activation_hook(module, input, output):
                    nonlocal first_activation
                    first_activation = output.detach()

                handle = list(model.children())[0].register_forward_hook(get_first_activation_hook)

                optimizer.zero_grad()
                with autocast(enabled=self.USE_AMP):
                    output = model(data)
                    loss = self.criterion(output, target)

                if self.USE_AMP:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                model.zero_grad()

                handle.remove()
                assert first_activation is not None, "⚠️ forward hook 未触发，请检查模型结构"

                # 提取特征
                first_activation = first_activation.cpu().numpy()
                input_grad = data.grad.cpu().numpy()

                if dataset_name == 'IMDB':
                    target = target.view(-1,1) * (1 - attacked)

                data, target = data.to(self.device), target.to(self.device)
                # for CIFAR10 multi-LF attack
                # if attacked:
                #     target = (target + 1)%10
                output = model(data)
                target = target.view(-1).long()  # ✅ 加这句

                loss = self.criterion(output, target)
                self.scaler.scale(loss).backward()
                epoch_loss.append(loss.item())
                # get gradients
                cur_time = time.time()
                for i, (name, params) in enumerate(model.named_parameters()):
                    if params.requires_grad:
                        if epoch == 0 and batch_idx == 0:
                            peer_grad.append(params.grad.clone())
                        else:
                            peer_grad[i]+= params.grad.clone()   
                t+= time.time() - cur_time    
                self.scaler.step(optimizer)
                self.scaler.update()
                model.zero_grad()
                optimizer.zero_grad()
               
            # print('Train epoch: {} \tLoss: {:.6f}'.format((epochs+1), np.mean(epoch_loss)))
    
        if (attack_type == 'gaussian' and self.peer_type == 'attacker'):
            update, flag =  gaussian_attack(model.state_dict(), self.peer_pseudonym,
            malicious_behavior_rate = malicious_behavior_rate, device = self.device)
            if flag == 1:
                self.performed_attacks+=1
                attacked = 1
            model.load_state_dict(update)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print("Number of Attacks:{}".format(self.performed_attacks))
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = model.cpu()
        return model.state_dict(), peer_grad , model, np.mean(epoch_loss), attacked, t, [peer_grad[-1].cpu().numpy(),   #output gradient
            first_activation,   #First level activation value
            input_grad]#Input Gradient


#======================================= End of training function =============================================================#
#========================================= End of Peer class ====================================================================


class FL:
    def __init__(self, dataset_name, model_name, dd_type, num_peers,
                 frac_peers, seed, test_batch_size, criterion, global_rounds,
                 local_epochs, local_bs, local_lr, local_momentum,
                 labels_dict, device, attackers_ratio,
                 class_per_peer, samples_per_class,
                 rate_unbalance, alpha, source_class,target_class,
                 USE_AMP=False):  # ✅ 添加这一行参数（并给默认值）

        FL._history = np.zeros(num_peers)
        self.scaler = GradScaler(enabled=USE_AMP)  # USE_AMP 是 True（A100）或 False（CPU/3060）
        self.USE_AMP = USE_AMP
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.num_peers = num_peers
        self.peers_pseudonyms = ['Peer ' + str(i+1) for i in range(self.num_peers)]
        self.frac_peers = frac_peers
        self.seed = seed
        self.test_batch_size = test_batch_size
        self.criterion = criterion
        self.global_rounds = global_rounds
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.labels_dict = labels_dict
        self.num_classes = len(self.labels_dict)
        self.device = device
        self.attackers_ratio = attackers_ratio
        self.class_per_peer = class_per_peer
        self.samples_per_class = samples_per_class
        self.rate_unbalance = rate_unbalance
        self.source_class = source_class
        self.dd_type = dd_type
        self.alpha = alpha
        self.embedding_dim = 100
        self.peers = []
        self.source_class = source_class
        self.target_class = target_class

        self.trainset, self.testset = None, None
        # Fix the random state of the environment
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
       
        #Loading of data
        self.trainset, self.testset, user_groups_train, tokenizer = distribute_dataset(self.dataset_name, self.num_peers, self.num_classes, 
        self.dd_type, self.class_per_peer, self.samples_per_class, self.alpha)

        self.test_loader = DataLoader(
            self.testset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=0,  # ✅ 提高加载效率
            pin_memory=True
        )

        #Creating model
        self.global_model = setup_model(model_architecture = self.model_name, num_classes = self.num_classes, 
        tokenizer = tokenizer, embedding_dim = self.embedding_dim)
        self.global_model = self.global_model.to(self.device)
        
        # Dividing the training set among peers
        self.local_data = []
        self.have_source_class = []
        self.labels = []
        print('--> Distributing training data among peers')
        for p in user_groups_train:
            self.labels.append(user_groups_train[p]['labels'])
            indices = user_groups_train[p]['data']
            peer_data = CustomDataset(self.trainset, indices=indices)
            self.local_data.append(peer_data)
            if  self.source_class in user_groups_train[p]['labels']:
                 self.have_source_class.append(p)
        print('--> Training data have been distributed among peers')

        # Creating peers instances
        print('--> Creating peets instances')
        m_ = 0
        if self.attackers_ratio > 0:
            #pick m random participants from the workers list
            # k_src = len(self.have_source_class)
            # print('# of peers who have source class examples:', k_src)
            m_ = int(self.attackers_ratio * self.num_peers)
            self.num_attackers = copy.deepcopy(m_)
        peers = list(np.arange(self.num_peers))  
        random.shuffle(peers)
        for i in peers:
            if m_ > 0 and contains_class(self.local_data[i], self.source_class):
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum, peer_type = 'attacker'))
                m_-= 1
            else:
                self.peers.append(Peer(i, self.peers_pseudonyms[i], 
                self.local_data[i], self.labels[i],
                self.criterion, self.device, self.local_epochs, self.local_bs, self.local_lr, 
                self.local_momentum))  

        del self.local_data

#======================================= Start of testning function ===========================================================#
    def test(self, model, device, test_loader, dataset_name = None):
        model.eval()
        test_loss = []
        correct = 0
        n = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(self.device), target.to(self.device)
            with autocast(enabled=USE_AMP):
                output = model(data)
            if dataset_name == 'IMDB':
                test_loss.append(self.criterion(output, target.view(-1,1)).item()) # sum up batch loss
                pred = output > 0.5 # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()
            else:
                target = target.view(-1).long()  # ✅ 保证 target 是 [B] 的 long 类型

                test_loss.append(self.criterion(output, target).item()) # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct+= pred.eq(target.view_as(pred)).sum().item()

            n+= target.shape[0]
        test_loss = np.mean(test_loss)
        print('\nAverage test loss: {:.4f}, Test accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct, n,
           100*correct / n))
        return  100.0*(float(correct) / n), test_loss
    #======================================= End of testning function =============================================================#
#Test label prediction function    
    def test_label_predictions(self, model, device, test_loader, dataset_name=None):
        model.eval()
        actuals = []
        predictions = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                if dataset_name != "IMDB":
                    target = target.view(-1).long()  # ✅ 保证 target 是 1D long

                output = model(data)

                if dataset_name == 'IMDB':
                    prediction = output > 0.5
                else:
                    prediction = output.argmax(dim=1, keepdim=True)

                actuals.extend(target.view(-1, 1))  # ✅ 统一扩展为 [B, 1]
                predictions.extend(prediction)

        return [i.item() for i in actuals], [i.item() for i in predictions]

    #choose random set of peers
    def choose_peers(self):
        #pick m random peers from the available list of peers
        m = max(int(self.frac_peers * self.num_peers), 1)
        selected_peers = np.random.choice(range(self.num_peers), m, replace=False)

        # print('\nSelected Peers\n')
        # for i, p in enumerate(selected_peers):
        #     print(i+1, ': ', self.peers[p].peer_pseudonym, ' is ', self.peers[p].peer_type)
        return selected_peers

        
    def run_experiment(self, attack_type = 'no_attack', malicious_behavior_rate = 0,
        source_class = None, target_class = None, rule = 'fedavg', resume = False):
        simulation_model = copy.deepcopy(self.global_model)
        print('\n===>Simulation started...')
        lfd = LFD(self.num_classes)
        fg = FoolsGold(self.num_peers)
        tolpegin = Tolpegin()
        # copy weights
        global_weights = simulation_model.state_dict()
        last10_updates = []
        test_losses = []
        global_accuracies = []
        source_class_accuracies = []
        cpu_runtimes = []
        noise_scalar = 1.0
        # best_accuracy = 0.0
        mapping = {'honest': 'Good update', 'attacker': 'Bad update'}

        #start training
        start_round = 0
        if resume:
            print('Loading last saved checkpoint..')
            checkpoint = torch.load('./checkpoints/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7')
            simulation_model.load_state_dict(checkpoint['state_dict'])
            start_round = checkpoint['epoch'] + 1
            last10_updates = checkpoint['last10_updates']
            test_losses = checkpoint['test_losses']
            global_accuracies = checkpoint['global_accuracies']
            source_class_accuracies = checkpoint['source_class_accuracies']
            
            print('>>checkpoint loaded!')
        print("\n====>Global model training started...\n")
        for epoch in tqdm_notebook(range(start_round, self.global_rounds)):
            gc.collect()
            torch.cuda.empty_cache()
            if epoch == 0 and rule == "lfighter_dbo" and (self.source_class is None or self.target_class is None):

                # 提取 local_models 的输出层
                output_weights = [model[-2] for model in local_models]  # 线性层 weight, shape=[C,D]
                dw = [self.global_model[-2] - lw for lw in output_weights]  # 差值

                avg_dw = torch.stack(dw).mean(0)  # 平均输出层变化，shape=[C,D]
                class_norms = torch.norm(avg_dw, dim=1)  # 每类改动的范数，shape=[C]

                sorted_classes = torch.argsort(class_norms, descending=True)
                self.source_class, self.target_class = sorted_classes[:2].tolist()

            # if epoch % 20 == 0:
            #     clear_output()  
            print(f'\n | Global training round : {epoch+1}/{self.global_rounds} |\n')
            selected_peers = self.choose_peers()
            local_weights, local_grads, local_models, local_losses, performed_attacks = [], [], [], [], []  
            peers_types = []
            i = 1        
            attacks = 0
            Peer._performed_attacks = 0
            peer_feature_views = []
            for peer in selected_peers:
                peers_types.append(mapping[self.peers[peer].peer_type])
                # print(i)
                # print('\n{}: {} Starts training in global round:{} |'.format(i, (self.peers_pseudonyms[peer]), (epoch + 1)))
                # print(f"[DEBUG] Starting training for peer {peer}...")
                peer_update, peer_grad, peer_local_model, peer_loss, attacked, t,views = self.peers[peer].participant_update(epoch,
                copy.deepcopy(simulation_model),
                attack_type = attack_type, malicious_behavior_rate = malicious_behavior_rate, 
                source_class = source_class, target_class = target_class, dataset_name = self.dataset_name)
                # print(f"[DEBUG] Finished training for peer {peer}")
                peer_feature_views.append(views)
                local_weights.append(peer_update)
                local_grads.append(peer_grad)
                local_losses.append(peer_loss) 
                local_models.append(peer_local_model) 
                attacks+= attacked
                # print('{} ends training in global round:{} |\n'.format((self.peers_pseudonyms[peer]), (epoch + 1))) 
                i+= 1
            # loss_avg = sum(local_losses) / len(local_losses)
            # print('Average of peers\' local losses: {:.6f}'.format(loss_avg))
            #aggregated global weights
            scores = np.zeros(len(local_weights))
            # Expected malicious peers
            f = int(self.num_peers*self.attackers_ratio)
            if rule == 'median':
                    cur_time = time.time()
                    global_weights = simple_median(local_weights)
                    cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'rmedian':
                cur_time = time.time()
                global_weights = Repeated_Median_Shard(local_weights)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'tmean':
                    cur_time = time.time()
                    trim_ratio = self.attackers_ratio*self.num_peers/len(selected_peers)
                    global_weights = trimmed_mean(local_weights, trim_ratio = trim_ratio)
                    cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'mkrum':
                cur_time = time.time()
                goog_updates = Krum(local_models, f = f, multi=True)
                scores[goog_updates] = 1
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)

            elif rule == 'foolsgold':
                cur_time = time.time()
                scores = fg.score_gradients(local_grads, selected_peers)
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time + t)

            elif rule == 'Tolpegin':
                cur_time = time.time()
                scores = tolpegin.score(copy.deepcopy(self.global_model), 
                                            copy.deepcopy(local_models), 
                                            peers_types = peers_types,
                                            selected_peers = selected_peers)
                global_weights = average_weights(local_weights, scores)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)
            
            elif rule == 'FLAME':
                cur_time = time.time()
                global_weights = FLAME(copy.deepcopy(self.global_model).cpu(), copy.deepcopy(local_models), noise_scalar)
                t = time.time() - cur_time
                print('Aggregation took', np.round(t, 4))
                cpu_runtimes.append(t)


            elif rule == 'lfighter':
                cur_time = time.time()
                global_weights = lfd.aggregate(copy.deepcopy(simulation_model), copy.deepcopy(local_models), peers_types)
                cpu_runtimes.append(time.time() - cur_time)
            elif rule == 'lfighter_dbo':
                cur_time = time.time()
                clusterer = DBOClusterer(n_clusters=2, device=self.device)
                cluster_labels = clusterer.cluster(peer_feature_views)
                true_labels = [1 if self.peers[p].peer_type == 'attacker' else 0 for p in selected_peers]
                # print("聚类 vs 实际攻击者标签:")
                # print(confusion_matrix(true_labels, cluster_labels))
                # print(classification_report(true_labels, cluster_labels, digits=3))

                # ✅ 标签方向判断
                cm = confusion_matrix(true_labels, cluster_labels)
                if cm[0][0] + cm[1][1] < cm[0][1] + cm[1][0]:
                    cluster_labels = [1 - l for l in cluster_labels]
                    # print("⚠️ 聚类标签方向颠倒，已自动纠正")

                # ✅ 聚类指标
                from sklearn.metrics import adjusted_rand_score
                # print("ARI (Adjusted Rand Index):", adjusted_rand_score(true_labels, cluster_labels))

                scores = [1.0 if l == 0 else 0.0 for l in cluster_labels]
                global_weights = average_weights(local_weights, scores)
                cpu_runtimes.append(time.time() - cur_time)





            elif rule == 'fedavg':
                cur_time = time.time()
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                cpu_runtimes.append(time.time() - cur_time)
            
            else:
                global_weights = average_weights(local_weights, [1 for i in range(len(local_weights))])
                ##############################################################################################
            #Plot honest vs attackers
            # if attack_type == 'label_flipping' and epoch >= 10 and epoch < 20:
            #     plot_updates_components(local_models, peers_types, epoch=epoch+1)   
            #     plot_layer_components(local_models, peers_types, epoch=epoch+1, layer = 'linear_weight')  
            #     plot_source_target(local_models, peers_types, epoch=epoch+1, classes= [source_class, target_class])
            # update global weights
            g_model = copy.deepcopy(simulation_model)
            simulation_model.load_state_dict(global_weights)           
            if epoch >= self.global_rounds-10:
                last10_updates.append(global_weights) 

            current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            
            if np.isnan(test_loss):
                simulation_model = copy.deepcopy(g_model)
                noise_scalar = noise_scalar*0.5
            
            global_accuracies.append(np.round(current_accuracy, 2))
            test_losses.append(np.round(test_loss, 4))
            performed_attacks.append(attacks) 
            state = {
                'epoch': epoch,
                'state_dict': simulation_model.state_dict(),
                'global_model':g_model,
                'local_models':copy.deepcopy(local_models),
                'last10_updates':last10_updates,
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies
                }
            savepath = './checkpoints/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
            torch.save(state,savepath)
            del local_models
            del local_weights
            del local_grads
            gc.collect()
            torch.cuda.empty_cache()
            # print("***********************************************************************************")
            #print and show confusion matrix after each global round
            actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
            classes = list(self.labels_dict.keys())
            print('{0:10s} - {1}'.format('Class','Accuracy'))
            for i, r in enumerate(confusion_matrix(actuals, predictions)):
                print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                if i == source_class:
                    source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))

            if epoch == self.global_rounds-1:
                print('Last 10 updates results')
                global_weights = average_weights(last10_updates, 
                np.ones([len(last10_updates)]))
                simulation_model.load_state_dict(global_weights) 
                current_accuracy, test_loss = self.test(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                global_accuracies.append(np.round(current_accuracy, 2))
                test_losses.append(np.round(test_loss, 4))
                performed_attacks.append(attacks)
                print("***********************************************************************************")
                #print and show confusion matrix after each global round
                actuals, predictions = self.test_label_predictions(simulation_model, self.device, self.test_loader, dataset_name=self.dataset_name)
                classes = list(self.labels_dict.keys())
                print('{0:10s} - {1}'.format('Class','Accuracy'))
                asr = 0.0
                for i, r in enumerate(confusion_matrix(actuals, predictions)):
                    print('{0:10s} - {1:.1f}'.format(classes[i], r[i]/np.sum(r)*100))
                    if i == source_class:
                        source_class_accuracies.append(np.round(r[i]/np.sum(r)*100, 2))
                        actuals = np.array(actuals)
                        preds = np.array(predictions)

                        source_mask = (actuals == source_class)
                        n_source = source_mask.sum()
                        n_attacked = np.sum(preds[source_mask] == target_class)

                        asr = round(n_attacked / n_source * 100, 2) if n_source > 0 else 0.0



        state = {
                'state_dict': simulation_model.state_dict(),
                'test_losses': test_losses,
                'global_accuracies': global_accuracies,
                'source_class_accuracies': source_class_accuracies,
                'asr':asr,
                'avg_cpu_runtime':np.mean(cpu_runtimes)
                }
        savepath = './results/'+ self.dataset_name + '_' + self.model_name + '_' + self.dd_type + '_'+ rule + '_'+ str(self.attackers_ratio) + '_' + str(self.local_epochs) + '.t7'
        torch.save(state,savepath)            
        print('Global accuracies: ', global_accuracies)
        print('Class {} accuracies: '.format(source_class), source_class_accuracies)
        print('Test loss:', test_losses)
        print('Attack succes rate:', asr)
        print('Average CPU aggregation runtime:', np.mean(cpu_runtimes))
