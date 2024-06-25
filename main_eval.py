import os
import copy
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader_custom_eval import get_cifar
from model_factory import create_cnn_model, is_resnet

import time
import datetime

SEED = 12
os.environ["CIDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

RS = 20150101


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs, elapsed_time

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	else:
		return False


def parse_arguments():
  parser = argparse.ArgumentParser(description='Topological Guidance based Knowledge Distillation Testing Code')
  parser.add_argument('--dataset', default='cifar100', type=str, help='dataset. can be either cifar10 or cifar100')
  parser.add_argument('--batch-size', default=128, type=int, help='batch_size')
  parser.add_argument('--student', '--model', default='resnet8', type=str, help='test student name')
  parser.add_argument('--student_checkpoint', default='', type=str, help='checkpoint for test student')
  parser.add_argument('--cuda', default=False, type=str2bool, help='whether or not use cuda(train on GPU)')
  parser.add_argument('--dataset-dir', default='./data', type=str,  help='dataset directory')
  parser.add_argument('--trial', default=0, type=str,  help='trial memo number')
  parser.add_argument('--seed', default=1234, type=int,  help='seed number')
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

class EvalManager(object):
  '''
  To test a pretrained model
  '''
  def __init__(self, student, train_loader=None, test_loader=None, train_config={}):
    self.student = student
    self.device = train_config['device']
    self.name = train_config['name']

    self.train_loader = train_loader
    self.test_loader = test_loader
    self.config = train_config
    
  def test(self):
  
    lambda_ = 0.9 #lambda of KD
    T = 4 #temperature tau of KD
    trial_id = self.config['trial_id']
    max_val_acc = 0
    iteration = 0
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    
      
    self.student.eval()
    self.student.train(mode=False)
      
    total = 0
    correct = 0

      
    with torch.no_grad():

      loss = 0
      CE_loss = 0.0
      T_loss = 0.0
      T_loss1 = 0.0
      final_size = 0

      num_classes = 10

      start_time = time.time()
               
      for batch_idx, (threeD_data, target) in enumerate(self.test_loader):
        # test model      
        threeD_data = threeD_data.to(self.device) #original image
        target = target.to(self.device) #target label
        
        output, sp1s, sp2s, sp3s = self.student(threeD_data)

        loss_SL = criterion(output, target) #softmax entropy
        loss = loss_SL

        
        T_loss1 += loss*target.size(0)
        final_size += target.size(0)
       
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
              

      acc = 100 * correct / total
      #print(correct, total)
      best_buf = "%.4f" % (acc)
      
      
      end_time = time.time()
      epoch_mins, epoch_secs, total_time = epoch_time(start_time, end_time)
      current_time = datetime.datetime.now()
      print(f'current_time: {current_time}')
      print(f'correct / total: {correct} / {total}')

      ls = T_loss1 / final_size
      print(f'"Time: {total_time} | {epoch_mins}m {epoch_secs}s | ' \
                    f'loss {ls:.9f} | accuracy {best_buf}')

    return acc


if __name__ == "__main__":
  # Parsing arguments and prepare settings for training
  args = parse_arguments()
  print(args)

  SEED = args.seed
  torch.manual_seed(SEED)
  torch.cuda.manual_seed(SEED)
  trial_id = args.trial
  dataset = args.dataset

  if dataset == 'cifar100':
    num_classes = 100
  else:
    num_classes = 10 #cifar 10
  
  
  student_model = create_cnn_model(args.student, dataset, use_cuda=args.cuda, num_cls=num_classes, in_channel=3) #load model structure
  
  train_config = {'device': 'cuda' if args.cuda else 'cpu',
                  'is_plane': not is_resnet(args.student),
                  'trial_id': trial_id,
                  }

  if args.student_checkpoint:
      student_model = load_checkpoint(student_model, args.student_checkpoint) #load trained model weight
        
        
  # Model testing setup
  print("---------- Testing Student -------")
  student_train_config = copy.deepcopy(train_config)
  train_loader, test_loader = get_cifar(dataset_dir='./data')
  student_train_config['name'] = args.student
  
  # test model
  print("---------- Model evaluation -------")
  student_tester = EvalManager(student_model, train_loader=train_loader, test_loader=test_loader, train_config=student_train_config)
  student_acc = student_tester.test()
    
