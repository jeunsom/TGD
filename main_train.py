import os
import copy
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader_custom import get_cifar
from model_factory import create_cnn_model, is_resnet

import time
import datetime


SEED = 12
os.environ["CIDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False
	
	
def parse_arguments():
  parser = argparse.ArgumentParser(description='Topological Guidance based Knowledge Distillation Training Code')
  parser.add_argument('--epochs', default=200, type=int,  help='number of total epochs to run')
  parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100')
  parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
  parser.add_argument('--alpha', default=0.99, type=float, help='alpha')
  parser.add_argument('--learning-rate', default=0.1, type=float, help='initial learning rate')
  parser.add_argument('--momentum', default=0.9, type=float,  help='SGD momentum')
  parser.add_argument('--weight-decay', default=1e-4, type=float, help='SGD weight decay (default: 1e-4)')
  parser.add_argument('--teacher1', default='', type=str, help='teacher1 name ORG')
  parser.add_argument('--teacher2', default='', type=str, help='teacher2 name PI')
  parser.add_argument('--student', '--model', default='resnet8', type=str, help='teacher student name')
  parser.add_argument('--teacher_checkpoint1', default='', type=str, help='optinal pretrained checkpoint for teacher1')
  parser.add_argument('--teacher_checkpoint2', default='', type=str, help='optinal pretrained checkpoint for teacher2')
  parser.add_argument('--student_checkpoint', default='', type=str, help='optinal pretrained checkpoint for student')
  parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
  parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
  parser.add_argument('--trial', default=0, type=str,  help='trial memo number')
  parser.add_argument('--seed', default=1234, type=int,  help='seed number')
  parser.add_argument('--save_weight', default=0, type=int,  help='save_default:0 save_flag:1')
  args = parser.parse_args()
  return args


def load_checkpoint(model, checkpoint_path):
	"""
	Loads weights from checkpoint
	:param model: a pytorch nn student
	:param str checkpoint_path: address/path of a file
	:return: pytorch nn student with weights loaded from checkpoint
	"""
	model_ckp = torch.load(checkpoint_path)
	model.load_state_dict(model_ckp['model_state_dict'])
	return model




class SP2(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''
    def __init__(self):
        super(SP2, self).__init__()

    def forward(self, fm_s):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s  = torch.mm(fm_s, fm_s.t())
        return G_s
    
class TrainManager(object):
  def __init__(self, student, teacher1=None, teacher2=None, train_loader=None, test_loader=None, train_config={}):
    self.student = student
    self.teacher1 = teacher1
    self.teacher2 = teacher2
    self.have_teacher = bool(self.teacher1) and bool(self.teacher2) 
    self.device = train_config['device']
    self.name = train_config['name']
    self.optimizer = optim.SGD(self.student.parameters(),
                               lr=train_config['learning_rate'],
                               momentum=train_config['momentum'],
                               weight_decay=train_config['weight_decay'])
    if self.have_teacher:
      self.teacher1.eval()
      self.teacher1.train(mode=False)
      self.teacher2.eval()
      self.teacher2.train(mode=False)
    
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.config = train_config
  def train(self):
    lambda_ = 0.9 #lambda for KD
    T = 4 #temperature tau for KD
    epochs = self.config['epochs']
    trial_id = self.config['trial_id']
    max_val_acc = 0
    iteration = 0
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    save_flag = args.save_weight
    
    
    SP_make = SP2()

    for epoch in range(epochs):
      start_time = time.time()
      self.student.train()
      lr = self.adjust_learning_rate(self.optimizer, epoch) #loss plan
      loss = 0
      alpha = args.alpha

      count_iter = 0
      CE_loss1 = 0.0
      KD_loss1 = 0.0
      T_loss1 = 0.0
      sp_loss1 = 0.0
      
      for batch_idx, (sixD_data, threeD_data, target) in enumerate(self.train_loader):
      
        iteration += 1
        sixD_data = sixD_data.to(self.device) #data for PIs
        threeD_data = threeD_data.to(self.device) #data for original image
        target = target.to(self.device) #data for target labels

        self.optimizer.zero_grad()
        output, sp1s, sp2s, sp3s = self.student(threeD_data)
        
        loss_SL = criterion(output, target) #softmax entropy
        loss = loss_SL

        if self.have_teacher:
          teacher_outputs1,sp1t,sp2t,sp3t = self.teacher1(threeD_data) #original img
          teacher_outputs2,sp1t2,sp2t2,sp3t2 = self.teacher2(sixD_data) #tda img
          
          #To compute KD loss
          loss_KD1 = nn.KLDivLoss()(F.log_softmax(output / T, dim=1), F.softmax(teacher_outputs1 / T, dim=1))
          loss_KD2 = nn.KLDivLoss()(F.log_softmax(output / T, dim=1), F.softmax(teacher_outputs2 / T, dim=1))
          
          #To utilize simialrity
          tmap1 = SP_make(sp1t)*alpha + SP_make(sp1t2)*(1-alpha)
          tmap2 = SP_make(sp2t)*alpha + SP_make(sp2t2)*(1-alpha)
          tmap3 = SP_make(sp3t)*alpha + SP_make(sp3t2)*(1-alpha)
          tmap1 = F.normalize(tmap1, p=2, dim=1)
          tmap2 = F.normalize(tmap2, p=2, dim=1)
          tmap3 = F.normalize(tmap3, p=2, dim=1)

          sp1sr = SP_make(sp1s)
          sp1sr = F.normalize(sp1sr, p=2, dim=1)
          sp2sr = SP_make(sp2s)
          sp2sr = F.normalize(sp2sr, p=2, dim=1)
          sp3sr = SP_make(sp3s)
          sp3sr = F.normalize(sp3sr, p=2, dim=1)

          sp_losst = F.mse_loss(sp1sr,tmap1).cuda() + F.mse_loss(sp2sr,tmap2).cuda() + F.mse_loss(sp3sr,tmap3).cuda() #compute loss for similarity
          sp_loss1 += sp_losst * target.size(0)
          
          loss_KD = alpha * loss_KD1 + (1-alpha)*loss_KD2
          KD_loss1 += loss_KD * target.size(0)
          loss = (1 - lambda_) * loss_SL + lambda_ * T * T * loss_KD + sp_losst / 3.0 * 3000.0
        
        
        loss.backward()
        self.optimizer.step()
        
        T_loss1 += loss*target.size(0)
        CE_loss1 += loss_SL*target.size(0)
        count_iter += target.size(0)

      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      current_time = datetime.datetime.now()

      print(f'current_time: {current_time}')
      best_buf = "%.4f" % (best_acc)
      if self.have_teacher:
        ls = T_loss1 / count_iter
        l_KD = KD_loss1 / count_iter
        l_CE = CE_loss1 / count_iter
        print(f'"epoch {epoch}/{epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | lr: {lr:.7f} | ' \
          			f'loss {ls:.9f} loss_CE {l_CE:.9f} loss_KD {l_KD:.9f} | best_acc {best_buf}')
      else:
        ls = T_loss1 / count_iter
        print(f'"epoch {epoch}/{epochs} | Epoch Time: {epoch_mins}m {epoch_secs}s | lr: {lr:.7f} | ' \
          			f'loss {ls:.9f} | best_acc {best_buf}')
      val_acc = self.validate(step=epoch)
      if val_acc > best_acc:
        best_acc = val_acc
        buf = "%.4f" % (val_acc)
        if epoch >= 0 and save_flag > 0: # when the model shows the best accuracy
          self.save(epoch, name='./test_model/{}_{}_ep{}_val{}_best.pth.tar'.format(self.name, trial_id, epoch, buf))
      if epoch % 50 == 0 or epoch% 30 == 0:
        buf = "%.4f" % (val_acc)
        if epoch >= 40 and save_flag > 0: #4:
          self.save(epoch, name='./test_model/{}_{}_ep{}_val{}_current.pth.tar'.format(self.name, trial_id, epoch, buf))
      
      if epoch == epochs - 1 and save_flag > 0:
        buf = "%.4f" % (val_acc)
        self.save(epoch, name='./test_model/{}_{}_ep{}_val{}_final.pth.tar'.format(self.name, trial_id, epoch, buf))
    return best_acc
  
  def validate(self, step=0):
    self.student.eval()
    with torch.no_grad():
      correct = 0
      total = 0
      acc = 0
      T = 4
      loss_KD = 0.0
      KD_loss = 0.0
      loss_SL = 0.0
      alpha = args.alpha
      loss_ = 0.0
      for sixD_data, threeD_data, labels in self.test_loader:
        sixD_data = sixD_data.to(self.device)
        threeD_data = threeD_data.to(self.device) # to test original image
        labels = labels.to(self.device)
        outputs, at1s, at2s, at3s = self.student(threeD_data)
        loss_SL = nn.CrossEntropyLoss()(outputs, labels) #softmax entropy
        loss_ += loss_SL* labels.size(0)
        if self.have_teacher:
          teacher_outputs1,at1t1,at2t1,at3t1 = self.teacher1(threeD_data)
          teacher_outputs2,at1t2,at2t2,at3t2 = self.teacher2(sixD_data)
          
          #KD
          loss_KD1 = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs1 / T, dim=1))
          loss_KD2 = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs2 / T, dim=1))
          
          loss_KD = alpha * loss_KD1 + (1-alpha)*loss_KD2
          
          KD_loss += loss_KD * labels.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
      acc = 100 * correct / total
      #print(correct, total)
      buf = "%.4f" % (acc)
      if self.have_teacher:
        KD_L = KD_loss / total #total
        SF_L = loss_ / total
        print(f'( "metric": "{self.name}_val_accuracy", "value": {buf}, "SF_L": {SF_L:.9f}, "KD_L": {KD_L:.9f} )')
      else:
        SF_L = loss_ / total
        print(f'( "metric": "{self.name}_val_accuracy", "value": {buf}, "SF_L": {SF_L:.9f})')
      return acc
  
  def save(self, epoch, name=None):
    trial_id = self.config['trial_id']
    if name is None:
      torch.save({'epoch': epoch,
                  'model_state_dict': self.student.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  }, '{}_{}_epoch{}.pth.tar'.format(self.name, trial_id, epoch))
    else:
      torch.save({'model_state_dict': self.student.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict(),
                  'epoch': epoch,
                  }, name)
  
  def adjust_learning_rate(self, optimizer, epoch):
    epochs = self.config['epochs']
    models_are_plane = self.config['is_plane']
    same_lr = 0
    
    if same_lr:
      lr = 0.01
    else:
      if epoch < int(40):
        lr = 0.1 #initial learning rate
      elif epoch < int(80):
        lr = 0.1 * 0.2
      elif epoch < int(120):
        lr = 0.1 * 0.2*0.2
      elif epoch < int(160):
        lr = 0.1 * 0.2*0.2*0.2
      else:
        lr = 0.1 * 0.2*0.2*0.2*0.2
    # update optimizer's learning rate
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    
    return lr


if __name__ == "__main__":
  # Parsing arguments and prepare settings for training
  args = parse_arguments()
  print(args)

  SEED = args.seed
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  trial_id = args.trial
  dataset = args.dataset
  
  #num_classes = 100 if dataset == 'cifar100' else 'cifar10'
  if dataset == 'cifar100':
    num_classes = 100
  else:
    num_classes = 10
  
  teacher_model = None
  student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda, num_cls=num_classes)
  train_config = {'epochs': args.epochs,
                  'learning_rate': args.learning_rate,
                  'momentum': args.momentum,
                  'weight_decay': args.weight_decay,
                  'device': 'cuda' if args.cuda else 'cpu',
                  'is_plane': not is_resnet(args.student),
                  'trial_id': trial_id,
                  }
  if args.student_checkpoint:
      student_model = load_checkpoint(student_model, args.student_checkpoint)
        

  if args.teacher1: # to create Teacher1
    teacher_model1 = create_cnn_model(args.teacher1, dataset, use_cuda=args.cuda, num_cls=num_classes, in_channel=3)
    if args.teacher_checkpoint1:
      print("---------- Loading Teacher -------")
      teacher_model1 = load_checkpoint(teacher_model1, args.teacher_checkpoint1)
    else:
      print("---------- Training Teacher -------")
      train_loader, test_loader = get_cifar()
      teacher_train_config = copy.deepcopy(train_config)
      teacher_name = 'teacher_{}_{}_best.pth.tar'.format(args.teacher, trial_id)
      teacher_train_config['name'] = args.teacher1
      teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
      teacher_trainer.train()
      teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
    
    
  if args.teacher2: # to create Teacher2
    teacher_model2 = create_cnn_model(args.teacher2, dataset, use_cuda=args.cuda, num_cls=num_classes, in_channel=6)
    if args.teacher_checkpoint2:
      print("---------- Loading Teacher -------")
      teacher_model2 = load_checkpoint(teacher_model2, args.teacher_checkpoint2)
    else:
      print("---------- Training Teacher -------")
      print("Alpha: ", args.alpha)
      train_loader, test_loader = get_cifar()
      teacher_train_config = copy.deepcopy(train_config)
      teacher_name = 'teacher_{}_{}_best.pth.tar'.format(args.teacher, trial_id)
      teacher_train_config['name'] = args.teacher2
      teacher_trainer = TrainManager(teacher_model, teacher=None, train_loader=train_loader, test_loader=test_loader, train_config=teacher_train_config)
      teacher_trainer.train()
      teacher_model = load_checkpoint(teacher_model, os.path.join('./', teacher_name))
  
  # Student training
  print("---------- Training Student -------")
  student_train_config = copy.deepcopy(train_config)
  train_loader, test_loader = get_cifar()
  student_train_config['name'] = args.student
  student_trainer = TrainManager(student_model, teacher1=teacher_model1, teacher2=teacher_model2, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
  best_student_acc = student_trainer.train()
