# pylint: disable=E1101
import torch
import torch.nn as nn
import numpy as np
from physionet import get_data_min_max, variable_time_collate_fn2



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_normal_pdf(x, mean, logvar, mask):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar)) * mask


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def mean_squared_error(orig, pred, mask):
    error = (orig - pred) ** 2
    error = error * mask
    return error.sum() / mask.sum()


def normalize_masked_data(data, mask, att_min, att_max):
    # we don't want to divide by zero
    att_max[att_max == 0.] = 1.

    if (att_max != 0.).all():
        data_norm = (data - att_min) / att_max
    else:
        raise Exception("Zero!")

    if torch.isnan(data_norm).any():
        raise Exception("nans!")

    # set masked out elements back to zero
    data_norm[mask == 0] = 0

    return data_norm, att_min, att_max


def evaluate(dim, rec, dec, test_loader, args, num_sample=10, device="cpu"):
    mse, future_mse, test_n = 0.0, 0.0, 0.0
    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            observed_data, observed_mask, observed_tp = (
                test_batch[:, :, :dim],
                test_batch[:, :, dim: 2 * dim],
                test_batch[:, :, -1],
            )
            if args.sample_tp and args.sample_tp < 1:
                subsampled_data, subsampled_tp, subsampled_mask = subsample_timepoints(
                    observed_data.clone(), observed_tp.clone(), observed_mask.clone(), args.sample_tp)
            else:
                subsampled_data, subsampled_tp, subsampled_mask = \
                    observed_data, observed_tp, observed_mask
            out = rec(torch.cat((subsampled_data, subsampled_mask), 2), subsampled_tp)
            qz0_mean, qz0_logvar = (
                out[:, :, : args.latent_dim],
                out[:, :, args.latent_dim:],
            )
            epsilon = torch.randn(
                num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]
            ).to(device)
            z0 = epsilon * torch.exp(0.5 * qz0_logvar) + qz0_mean
            z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
            batch, seqlen = observed_tp.size()
            time_steps = (
                observed_tp[None, :, :].repeat(num_sample, 1, 1).view(-1, seqlen)
            )
            pred_x = dec(z0, time_steps)
            pred_x = pred_x.view(num_sample, -1, pred_x.shape[1], pred_x.shape[2])
            pred_x = pred_x.mean(0)
            
            future_mask = torch.zeros((observed_mask.shape[0],observed_mask.shape[1],observed_mask.shape[2])).to(device)
            future_mask[:,int(future_mask.shape[1]*0.9),:] = 1
            future_mse += mean_squared_error(observed_data, pred_x, future_mask) * batch
            mse += mean_squared_error(observed_data, pred_x, observed_mask) * batch
            test_n += batch
    return mse / test_n, future_mse / test_n


def compute_losses(dim, dec_train_batch, qz0_mean, qz0_logvar, pred_x, args, device):
    observed_data, observed_mask \
        = dec_train_batch[:, :, :dim], dec_train_batch[:, :, dim:2*dim]

    noise_std = args.std  # default 0.1
    noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
    noise_logvar = 2. * torch.log(noise_std_).to(device)
    logpx = log_normal_pdf(observed_data, pred_x, noise_logvar,
                           observed_mask).sum(-1).sum(-1)
    pz0_mean = pz0_logvar = torch.zeros(qz0_mean.size()).to(device)
    analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                            pz0_mean, pz0_logvar).sum(-1).sum(-1)
    if args.norm:
        logpx /= observed_mask.sum(-1).sum(-1)
        analytic_kl /= observed_mask.sum(-1).sum(-1)
    return logpx, analytic_kl


def evaluate_classifier(model, test_loader, dec=None, args=None, classifier=None,
                        dim=41, device='cuda', reconst=False, num_sample=1):
    pred = []
    true = []
    test_loss = 0
    for test_batch, label in test_loader:
        test_batch, label = test_batch.to(device), label.to(device)
        batch_len = test_batch.shape[0]
        observed_data, observed_mask, observed_tp \
            = test_batch[:, :, :dim], test_batch[:, :, dim:2*dim], test_batch[:, :, -1]
        with torch.no_grad():
            out = model(
                torch.cat((observed_data, observed_mask), 2), observed_tp)
            if reconst:
                qz0_mean, qz0_logvar = out[:, :,
                                           :args.latent_dim], out[:, :, args.latent_dim:]
                epsilon = torch.randn(
                    num_sample, qz0_mean.shape[0], qz0_mean.shape[1], qz0_mean.shape[2]).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                z0 = z0.view(-1, qz0_mean.shape[1], qz0_mean.shape[2])
                if args.classify_pertp:
                    pred_x = dec(z0, observed_tp[None, :, :].repeat(
                        num_sample, 1, 1).view(-1, observed_tp.shape[1]))
                    #pred_x = pred_x.view(num_sample, batch_len, pred_x.shape[1], pred_x.shape[2])
                    out = classifier(pred_x)
                else:
                    out = classifier(z0)
            if args.classify_pertp:
                N = label.size(-1)
                out = out.view(-1, N)
                label = label.view(-1, N)
                _, label = label.max(-1)
                test_loss += nn.CrossEntropyLoss()(out, label.long()).item() * batch_len * 50.
            else:
                label = label.unsqueeze(0).repeat_interleave(
                    num_sample, 0).view(-1)
                test_loss += nn.CrossEntropyLoss()(out, label).item() * batch_len * num_sample
        pred.append(out.cpu().numpy())
        true.append(label.cpu().numpy())
    pred = np.concatenate(pred, 0)
    true = np.concatenate(true, 0)
    acc = np.mean(pred.argmax(1) == true)
    # auc = metrics.roc_auc_score(true, pred,multi_class='ovr') if not args.classify_pertp else 0.
    return test_loss/pred.shape[0], acc # , auc


def variable_time_collate_fn(batch, device=torch.device("cpu"), classify=False, activity=False,
                             data_min=None, data_max=None):
    """
    Expects a batch of time series data in the form of (record_id, tt, vals, mask, labels) where
      - record_id is a patient id
      - tt is a 1-dimensional tensor containing T time values of observations.
      - vals is a (T, D) tensor containing observed values for D variables.
      - mask is a (T, D) tensor containing 1 where values were observed and 0 otherwise.
      - labels is a list of labels for the current patient, if labels are available. Otherwise None.
    Returns:
      combined_tt: The union of all time observations.
      combined_vals: (M, T, D) tensor containing the observed values.
      combined_mask: (M, T, D) tensor containing 1 where values were observed and 0 otherwise.
    """

    D = batch[0][2].shape[1]
    # number of labels
    N = 1 # batch[0][-1].shape[0]
    len_tt = [ex[1].size(0) for ex in batch]
    maxlen = np.max(len_tt)
    enc_combined_tt = torch.zeros([len(batch), maxlen]).type('torch.DoubleTensor').to(device)
    enc_combined_vals = torch.zeros([len(batch), maxlen, D]).to(device)
    enc_combined_mask = torch.zeros([len(batch), maxlen, D]).type('torch.DoubleTensor').to(device)
    if classify:
        if activity:
            combined_labels = torch.zeros([len(batch), maxlen, N]).to(device)
        else:
            combined_labels = torch.zeros([len(batch), N]).to(device)
    

    
    for b, (record_id, tt, vals, mask, labels) in enumerate(batch):
        currlen = tt.size(0)
        enc_combined_tt[b, :currlen] = tt.to(device)
        enc_combined_vals[b, :currlen] = vals.to(device)
        enc_combined_mask[b, :currlen] = mask.to(device)
        if classify:
            if activity:
                combined_labels[b, :currlen] = labels.to(device)
            else:
                combined_labels[b] = labels.to(device)

    if not activity:
        enc_combined_vals, _, _ = normalize_masked_data(enc_combined_vals, enc_combined_mask,
                                                        att_min=data_min, att_max=data_max)

    if torch.max(enc_combined_tt) != 0.:
        enc_combined_tt = enc_combined_tt / torch.max(enc_combined_tt)
        

    combined_data = torch.cat(
        (enc_combined_vals, enc_combined_mask, enc_combined_tt.unsqueeze(-1)), 2)
    if classify:
        return combined_data, combined_labels
    else:
        return combined_data



def irregularly_sampled_data_gen(n=10, length=20, seed=0):
    np.random.seed(seed)
    # obs_times = obs_times_gen(n)
    obs_values, ground_truth, obs_times = [], [], []
    for i in range(n):
        t1 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t2 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        t3 = np.sort(np.random.uniform(low=0.0, high=1.0, size=length))
        a = 10 * np.random.randn()
        b = 10 * np.random.rand()
        f1 = .8 * np.sin(20*(t1+a) + np.sin(20*(t1+a))) + \
            0.01 * np.random.randn()
        f2 = -.5 * np.sin(20*(t2+a + 20) + np.sin(20*(t2+a + 20))
                          ) + 0.01 * np.random.randn()
        f3 = np.sin(12*(t3+b)) + 0.01 * np.random.randn()
        obs_times.append(np.stack((t1, t2, t3), axis=0))
        obs_values.append(np.stack((f1, f2, f3), axis=0))
        #obs_values.append([f1.tolist(), f2.tolist(), f3.tolist()])
        t = np.linspace(0, 1, 100)
        fg1 = .8 * np.sin(20*(t+a) + np.sin(20*(t+a)))
        fg2 = -.5 * np.sin(20*(t+a + 20) + np.sin(20*(t+a + 20)))
        fg3 = np.sin(12*(t+b))
        #ground_truth.append([f1.tolist(), f2.tolist(), f3.tolist()])
        ground_truth.append(np.stack((fg1, fg2, fg3), axis=0))
    return obs_values, ground_truth, obs_times



def compute_pertp_loss(label_predictions, true_label, mask):
    criterion = nn.CrossEntropyLoss(reduction='none')
    n_traj, n_tp, n_dims = label_predictions.size()
    label_predictions = label_predictions.reshape(n_traj * n_tp, n_dims)
    true_label = true_label.reshape(n_traj * n_tp, n_dims)
    mask = torch.sum(mask, -1) > 0
    mask = mask.reshape(n_traj * n_tp,  1)
    _, true_label = true_label.max(-1)
    ce_loss = criterion(label_predictions, true_label.long())
    ce_loss = ce_loss * mask
    return torch.sum(ce_loss)/mask.sum()



def subsample_timepoints(data, time_steps, mask, percentage_tp_to_sample=None):
    # Subsample percentage of points from each time series
    for i in range(data.size(0)):
        # take mask for current training sample and sum over all features --
        # figure out which time points don't have any measurements at all in this batch
        current_mask = mask[i].sum(-1).cpu()
        non_missing_tp = np.where(current_mask > 0)[0]
        n_tp_current = len(non_missing_tp)
        n_to_sample = int(n_tp_current * percentage_tp_to_sample)
        subsampled_idx = sorted(np.random.choice(
            non_missing_tp, n_to_sample, replace=False))
        tp_to_set_to_zero = np.setdiff1d(non_missing_tp, subsampled_idx)

        data[i, tp_to_set_to_zero] = 0.
        if mask is not None:
            mask[i, tp_to_set_to_zero] = 0.

    return data, time_steps, mask
