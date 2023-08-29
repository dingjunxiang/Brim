from argparse import Namespace
import os

import numpy as np
from sksurv.metrics import concordance_index_censored

import torch

from models.model_genomic import SNN
from models.model_porpoise import PorpoiseMMF, PorpoiseAMIL
from models.model_brim import Brim,Transmil
from models.model_brim_incom import Brim_incom
from utils.utils import *
from utils.loss_func import NLLSurvLoss
import torch.nn as nn


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss




def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)


    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print('\nInit loss function...', end=' ')
    if args.task_type == 'survival':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_omic
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion

    if args.model_type == 'porpoise_mmf':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes, 
        'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 
        'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,
        }
        model = PorpoiseMMF(**model_dict)
    elif args.model_type == 'brim':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes, 
        'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 
        'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,
        }
        model = Brim(**model_dict)
    elif args.model_type == 'incom_gene':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes, 
        'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 
        'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,
        }
        model = Brim_incom(**model_dict)
    elif args.model_type == 'transmil':
        model_dict = {'n_classes': args.n_classes}
        model = Transmil(**model_dict)
    elif args.model_type == 'porpoise_amil':
        model_dict = {'n_classes': args.n_classes}
        model = PorpoiseAMIL(**model_dict)
    elif args.model_type =='snn':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, 
        weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split,  testing = args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None

    print('\nSetup Validation C-Index Monitor...', end=' ')
    print('Done!')

    for epoch in range(args.max_epochs):
        if args.task_type == 'survival':
            train_loop_survival(epoch, model, train_loader, optimizer, args.n_classes,  loss_fn, reg_fn, args.lambda_reg, args.gc)
            stop = validate_survival(cur, epoch, model, val_loader, args.n_classes, early_stopping,  loss_fn, reg_fn, args.lambda_reg, args.results_dir)

    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))
    # model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    results_val_dict, val_cindex = summary_survival(model, val_loader, args.n_classes)

    print('Val c-Index: {:.4f}'.format(val_cindex))

    return results_val_dict, val_cindex


def train_loop_survival(epoch, model, loader, optimizer, n_classes, loss_fn=None, reg_fn=None, lambda_reg=0., gc=16):   
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.train()
    train_loss_surv, train_loss = 0., 0.

    print('\n')
    #all_risk_scores = np.zeros((len(loader)))
    #all_censorships = np.zeros((len(loader)))
    #all_event_times = np.zeros((len(loader)))
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    mse_loss = nn.MSELoss()
    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)
        h = model(x_path=data_WSI, x_omic=data_omic, validing="Training") # return hazards, S, Y_hat, A_raw, results_dict
        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            # h_path, h_omic, h_mm = h
            h_mm, h_omic, h_path, ae_omic, ae_path = h
            loss = 0.5*loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.25*mse_loss(ae_omic, h_omic)
            loss += 0.05*mse_loss(ae_path, h_path)
            loss += 0.05*loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            h = h_mm

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        all_risk_scores.append(risk)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())

        train_loss_surv += loss_value
        train_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item()), float(risk), data_WSI.size(0)))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0]), float(risk[0]), data_WSI.size(0)))
        
        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0: 
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_surv /= len(loader)
    train_loss /= len(loader)

    all_risk_scores = np.concatenate(all_risk_scores)
    all_censorships = np.concatenate(all_censorships)
    all_event_times = np.concatenate(all_event_times)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, train_loss_surv, train_loss, c_index))



def validate_survival(cur, epoch, model, loader, n_classes, early_stopping=None, loss_fn=None, reg_fn=None, lambda_reg=0., results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss_surv, val_loss = 0., 0.
    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic, validing="pseudo_gene") # return hazards, S, Y_hat, A_raw, results_dict

        if not isinstance(h, tuple):
            loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
        else:
            h_mm, h_omic, h_path,ae_omic,ae_path = h
            loss = 0.5*loss_fn(h=h_mm, y=y_disc, t=event_time, c=censor)
            loss += 0.15*loss_fn(h=h_path, y=y_disc, t=event_time, c=censor)
            loss += 0.25*loss_fn(h=h_omic, y=y_disc, t=event_time, c=censor)
            loss_value = loss.item()
            h = h_mm

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        if isinstance(loss_fn, NLLSurvLoss):
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy()

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor.detach().cpu().numpy()
        all_event_times[batch_idx] = event_time.detach().cpu().numpy()

        val_loss_surv += loss_value
        val_loss += loss_value + loss_reg

    val_loss_surv /= len(loader)
    val_loss /= len(loader)
    
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]



    if early_stopping:
        assert results_dir
        early_stopping(epoch, c_index, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def summary_survival(model, loader, n_classes):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            h = model(x_path=data_WSI, x_omic=data_omic,validing="pseudo_gene")
        
        if isinstance(h, tuple):
            # h = h[2]
            h = h[0]

        if h.shape[1] > 1:
            hazards = torch.sigmoid(h)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
        else:
            risk = h.detach().cpu().numpy().squeeze()

        event_time = np.asscalar(event_time)
        censor = np.asscalar(censor)
        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = censor
        all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'risk': risk, 'disc_label': y_disc.item(), 'survival': event_time, 'censorship': censor}})

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    return patient_results, c_index