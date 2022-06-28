# pylint: disable=E1101, E0401, E1102, W0621, W0221
import argparse
import glob
import zipfile
import numpy as np
import pandas as pd
from sklearn import model_selection
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.optim as optim

from random import SystemRandom
import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--std', type=float, default=0.01)
parser.add_argument('--latent-dim', type=int, default=32)
parser.add_argument('--rec-hidden', type=int, default=32)
parser.add_argument('--gen-hidden', type=int, default=50)
parser.add_argument('--embed-time', type=int, default=128)
parser.add_argument('--k-iwae', type=int, default=10)
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--enc', type=str, default='mtan_rnn')
parser.add_argument('--dec', type=str, default='mtan_rnn')
parser.add_argument('--fname', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n', type=int, default=8000)
parser.add_argument('--batch-size', type=int, default=50)
parser.add_argument('--quantization', type=float, default=0.016,
                    help="Quantization on the physionet dataset.")
parser.add_argument('--classif', action='store_true',
                    help="Include binary classification loss")
parser.add_argument('--norm', action='store_true')
parser.add_argument('--kl', action='store_true')
parser.add_argument('--learn-emb', action='store_true')
parser.add_argument('--enc-num-heads', type=int, default=1)
parser.add_argument('--dec-num-heads', type=int, default=1)
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--num-ref-points', type=int, default=128)
parser.add_argument('--dataset', type=str, default='fnirs')
parser.add_argument('--enc-rnn', action='store_false')
parser.add_argument('--dec-rnn', action='store_false')
parser.add_argument('--sample-tp', type=float, default=1.0)
parser.add_argument('--only-periodic', type=str, default=None)
parser.add_argument('--dropout', type=float, default=0.0)
args = parser.parse_args()


if __name__ == '__main__':
    experiment_id = int(SystemRandom().random() * 100000)
    print(args, experiment_id)
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')

    total_dataset = []

    if args.dataset == 'fnirs':
        for zip_file in glob.glob("size_30sec_150ts_stride_03ts.zip"):
            zf = zipfile.ZipFile(zip_file)
            for f in zf.namelist():
              df = pd.read_csv(zf.open(f))
              df.chunk = (df.chunk*0.6)/60
        
              df = df.groupby('chunk')[['AB_I_O','AB_PHI_O','AB_I_DO','AB_PHI_DO','CD_I_O','CD_PHI_O','CD_I_DO',
                                        'CD_PHI_DO','label']].mean().reset_index()
              df['label'] = df['label'].astype('int')
        
              tt = torch.tensor(df.chunk.values, device=device)
              vals = torch.tensor(df.iloc[:,1:-1].values, device=device)
              mask = torch.ones(vals.shape[0],vals.shape[1], device=device)
              labels = torch.tensor(df.label.values, device=device)
              
              if args.learn_emb:
                  n = df.shape[0]
                  for i in range(8):
                      np.random.seed(n*i)
                      prob = np.random.uniform(0.5,0.9)
                      missing_pos = np.random.choice(n, size=int(np.ceil(prob*n)), replace=False)
                      vals[missing_pos,i] = 0
                      mask[missing_pos,i] = 0
              
              record_id = str(f.split('.')[0].split('_')[1])
              total_dataset.append((record_id,tt,vals,mask,labels))
    
    else:
        for zip_file in glob.glob("financial_data.zip"):
            zf = zipfile.ZipFile(zip_file)
            j = 1
            for f in zf.namelist():
               df = pd.read_csv(zf.open(f),names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'])
               dys_min = (df.DateTime.str[8:10].astype('int')-2)*24*60
               hrs_min = (df.DateTime.str[11:13].astype('int')-4)*60
               minutes = df.DateTime.str[14:16].astype('int') + 1
               df.DateTime = dys_min + hrs_min + minutes
        
               tt = torch.tensor(df.DateTime.values, device=device)
               vals = torch.tensor(df.iloc[:,1:].values, device=device)
               mask = torch.ones(vals.shape[0],vals.shape[1], device=device)
               labels = torch.zeros(df.shape[0], device=device)
              
               if args.learn_emb:
                  n = df.shape[0]
                  for i in range(5):
                      np.random.seed(n*i) 
                      prob = np.random.uniform(0.5,0.9) 
                      missing_pos = np.random.choice(n, size=int(np.ceil(prob*n)), replace=False)
                      vals[missing_pos,i] = 0
                      mask[missing_pos,i] = 0
              
               record_id = str(j)
               total_dataset.append((record_id,tt,vals,mask,labels))
               j += 1
        
    train_data, test_data = model_selection.train_test_split(total_dataset, train_size=0.8,
                                                             random_state=42, shuffle=True)
    print('data read')
    record_id, tt, vals, mask, labels = train_data[0]
    
    
    dim = vals.shape[1]
    flag = 1.0

    data_min, data_max = utils.get_data_min_max(total_dataset, device)
    data_min, data_max = data_min.to(device), data_max.to(device)

    
    batch_size = min(args.batch_size, args.n)
    if flag:
        test_data_combined = utils.variable_time_collate_fn(test_data, device, classify=args.classif,
                                                      data_min=data_min, data_max=data_max)

        if args.classif:
            train_data, val_data = model_selection.train_test_split(train_data, train_size=0.8,
                                                                    random_state=11, shuffle=True)
            train_data_combined = utils.variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)
            val_data_combined = utils.variable_time_collate_fn(
                val_data, device, classify=args.classif, data_min=data_min, data_max=data_max)


            train_data_combined = TensorDataset(
                train_data_combined[0], train_data_combined[1].long().squeeze())
            val_data_combined = TensorDataset(
                val_data_combined[0], val_data_combined[1].long().squeeze())
            test_data_combined = TensorDataset(
                test_data_combined[0], test_data_combined[1].long().squeeze())
        else:
            train_data_combined = utils.variable_time_collate_fn(
                train_data, device, classify=args.classif, data_min=data_min, data_max=data_max)

        train_loader = DataLoader(
            train_data_combined, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(
            test_data_combined, batch_size=batch_size, shuffle=False)
        
        
    # model
    if args.enc == 'enc_rnn3':
        rec = models.enc_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.rec_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.enc == 'mtan_rnn':
        rec = models.enc_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.rec_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.enc_num_heads).to(device)
   
        
    if args.dec == 'rnn3':
        dec = models.dec_rnn3(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, 
            args.gen_hidden, 128, learn_emb=args.learn_emb).to(device)
    elif args.dec == 'mtan_rnn':
        dec = models.dec_mtan_rnn(
            dim, torch.linspace(0, 1., args.num_ref_points), args.latent_dim, args.gen_hidden, 
            embed_time=128, learn_emb=args.learn_emb, num_heads=args.dec_num_heads).to(device)


    params = (list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    print('parameters:', utils.count_parameters(rec), utils.count_parameters(dec))
    if args.fname is not None:
        checkpoint = torch.load(args.fname)
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('loading saved weights', checkpoint['epoch'])
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 1))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 3))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 10))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 20))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 30))
        print('Test MSE', utils.evaluate(dim, rec, dec, test_loader, args, 50))

    for itr in range(1, args.niters + 1):
        train_loss = 0
        train_n = 0
        avg_reconst, avg_kl, mse, future_mse = 0, 0, 0, 0
        if args.kl:
            wait_until_kl_inc = 10
            if itr < wait_until_kl_inc:
                kl_coef = 0.
            else:
                kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        else:
            kl_coef = 1

        for train_batch in train_loader:
            train_batch = train_batch.to(device)
            batch_len = train_batch.shape[0]
            observed_data = train_batch[:, :, :dim]
            observed_mask = train_batch[:, :, dim:2 * dim]
            observed_tp = train_batch[:, :, -1]
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = utils.subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean = out[:, :, :args.latent_dim]
            qz0_logvar = out[:, :, args.latent_dim:]
            epsilon = torch.randn(
                args.k_iwae, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            pred_x = dec(
                z0,
                observed_tp[None, :, :].repeat(args.k_iwae, 1, 1).view(-1, observed_tp.shape[1])
            )
            
            # nsample, batch, seqlen, dim
            pred_x = pred_x.view(args.k_iwae, batch_len, pred_x.shape[1], pred_x.shape[2])
            
            # compute loss
            logpx, analytic_kl = utils.compute_losses(
                dim, train_batch, qz0_mean, qz0_logvar, pred_x, args, device)
            loss = -(torch.logsumexp(logpx - kl_coef * analytic_kl, dim=0).mean(0) - np.log(args.k_iwae))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_len
            train_n += batch_len
            avg_reconst += torch.mean(logpx) * batch_len
            avg_kl += torch.mean(analytic_kl) * batch_len
            mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), observed_mask) * batch_len
            
            future_mask = torch.zeros((observed_mask.shape[0],observed_mask.shape[1],observed_mask.shape[2])).to(device)
            future_mask[:,int(future_mask.shape[1]*0.9),:] = 1
            future_mse += utils.mean_squared_error(
                observed_data, pred_x.mean(0), future_mask) * batch_len
            

        print('Iter: {}, avg elbo: {:.4f}, avg reconst: {:.4f}, avg kl: {:.4f}, mse: {:.6f}, future mse: {:.6f}'
            .format(itr, train_loss / train_n, -avg_reconst / train_n, avg_kl / train_n, mse / train_n, future_mse / train_n))
        if itr % 10 == 0:
            print('Test Mean Squared Error: {:.4f}, Test Future Mean Squared Error: {:.4f}' 
                  .format(utils.evaluate(dim, rec, dec, test_loader, args, 1)[0], utils.evaluate(dim, rec, dec, test_loader, args, 1)[1]))
        if itr % 10 == 0 and args.save:
            torch.save({
                'args': args,
                'epoch': itr,
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': -loss,
            }, args.dataset + '_' + args.enc + '_' + args.dec + '_' +
                str(experiment_id) + '.h5')