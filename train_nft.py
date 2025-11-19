import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb
import datetime

from reactot.trainer.pl_trainer import SBModule
from reactot.model.leftnet import LEFTNet
from reactot.analyze.rmsd import pymatgen_rmsd, rmsd_str
import reactot.diffusion._utils as utils


from ts_rl.sampler import KRepeatSampler
from ts_rl.energy_scorer import EnergyScorer
from ts_rl.stat_tracking import PerMoleculeStatTracker, PerPromptStatTracker


# torch.serialization.add_safe_globals([LEFTNet])

device = 'cuda:4'

def decode_molecule(pos, idx):
    """pos: (num_atom, 3), idx: (num_atom, 1)"""
    idx2symbol = {
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        3: "Li",
        12: "Mg",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        47: "Ag",
        48: "Cd",
        40: "Zr",
        72: "Hf",
    }
    atom_strings = [
        f"{idx2symbol.get(int(idx[i].item()), 'X')} {pos[i, 0]:.6f} {pos[i, 1]:.6f} {pos[i, 2]:.6f}"
        for i in range(pos.shape[0])
    ]
    
    mol = "; ".join(atom_strings)
    return mol
    

def load_model(checkpoint_path, device='cuda'):
    model = SBModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=device,
    )
    model = model.eval()
    model = model.to(device)

    model.training_config["use_sampler"] = True
    model.training_config["swapping_react_prod"] = False
    model.training_config["datadir"] = "./reactot/data/transition1x"

    model.setup(stage="fit", device=device, swapping_react_prod=False)
    return model



def process_samples(samples, log=False):

    for i, sample in enumerate(samples):
        new_dic = {}
        for subk in sample['representations'][0].keys():
            new_subk_tensor = torch.stack([_[subk] for _ in sample['representations']])
            new_dic[subk] = new_subk_tensor
        samples[i]['representations'] = new_dic

    new_samples = {}
    for k in samples[0].keys():
        if isinstance(samples[0][k], list):
            new_ele = [sub_x for n in samples for sub_x in n[k] ]
        elif isinstance(samples[0][k], dict):
            new_ele = dict()
            for subk in samples[0][k].keys():
                new_ele[subk] = torch.cat([n[k][subk] for n in samples], dim=1)
                # print(k, subk, new_ele[subk].shape)
        elif isinstance(samples[0][k], torch.Tensor):
            tmp = [sample[k] for sample in samples]
            new_ele = torch.cat(tmp, dim=0)
        else:
            new_ele = torch.cat([s[k] for s in samples], dim=0)
        new_samples[k] = new_ele
    
    if log:
        print_samples(new_samples)

    return new_samples


def print_samples(samples):
    for k in samples.keys():
        if isinstance(samples[k], list):
            print('list:', k, len(samples[k]), 'each element:', type(samples[k][0]))
        elif isinstance(samples[k], torch.Tensor):
            print('tensor:', k, samples[k].shape)
        elif isinstance(samples[k], dict):
            print(f'dict: {k}')
            for subk in samples[k].keys():
                print(f'in dict: {subk} - {samples[k][subk].shape}')



def reshuffle(samples, perm, device):
    for k in samples.keys():
        if isinstance(samples[k], torch.Tensor):
            samples[k] = samples[k][perm]
        elif isinstance(samples[k], list):
            perm_cpu_list = perm.detach().cpu().long().tolist()
            samples[k] = [samples[k][i] for i in perm_cpu_list]
        elif isinstance(samples[k], dict):
            if k == 'representations':
                zero_tensor = torch.zeros(1, device=device).long()
                sample_sizes = samples[k]['size'][0]  
                boundaries = torch.cat([zero_tensor, sample_sizes.cumsum(dim=0)], dim=0)  # [N+1]
                
                group_indices = [
                    torch.arange(boundaries[i], boundaries[i+1], device=device, dtype=torch.long) 
                    for i in range(samples[k]['size'].shape[1])  # 遍历所有样本
                ]
                new_atom_perm = torch.cat([group_indices[i] for i in perm])
                
                samples[k]['size'] = samples[k]['size'][:, perm]  # [3, N] 
                samples[k]['pos'] = samples[k]['pos'][:, new_atom_perm, :]  # [3, total_atoms, 3]
                samples[k]['one_hot'] = samples[k]['one_hot'][:, new_atom_perm, :]  # [3, total_atoms, 5]
                samples[k]['charge'] = samples[k]['charge'][:, new_atom_perm, :]  # [3, total_atoms, 1]
                samples[k]['mask'] = samples[k]['mask'][:, new_atom_perm]  # [3, total_atoms]
    return samples

def create_batches(samples, batch_size, device):
    total_size = samples['conditions'].shape[0]
    num_batches = (total_size + batch_size - 1) // batch_size  # 向上取整
    
    batches = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_size)
        actual_batch_size = end_idx - start_idx
        
        batch = {}
        
        for k in samples.keys():
            if isinstance(samples[k], torch.Tensor):
                batch[k] = samples[k][start_idx:end_idx]
            
            elif isinstance(samples[k], list):
                batch[k] = samples[k][start_idx:end_idx]
            
            elif isinstance(samples[k], dict) and k == 'representations':
                rep = samples[k]
                batch[k] = []  # list of len 3

                # 每个“分子样本”对应一行
                for i in range(3):
                    rep_i = {}

                    # size: [3, total_batch] → [batch]
                    rep_i['size'] = rep['size'][i, start_idx:end_idx]

                    # 当前batch每个样本的原子数
                    batch_sizes = rep_i['size']
                    if start_idx == 0:
                        global_start = 0
                    else:
                        global_start = rep['size'][i, :start_idx].sum().item()
                    global_end = global_start + batch_sizes.sum().item()

                    # pos, one_hot, charge, mask 去掉第一维
                    rep_i['pos'] = rep['pos'][i, global_start:global_end, :]
                    rep_i['one_hot'] = rep['one_hot'][i, global_start:global_end, :]
                    rep_i['charge'] = rep['charge'][i, global_start:global_end, :]
                    rep_i['mask'] = rep['mask'][i, global_start:global_end]

                    batch[k].append(rep_i)
        
        batches.append(batch)
    
    return batches


def update_model_ema(old_model, current_model, eta):
    with torch.no_grad():
        for old_param, curr_param in zip(old_model.parameters(), current_model.parameters()):
            old_param.data.mul_(eta).add_(curr_param.data, alpha=1 - eta)



def main(args):
    
    # logger
    project_name = "ts_rl"
    group_name = "nft"

    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{group_name}_{timestamp_str}"

    wandb.init(
        project=project_name, 
        group=group_name, 
        name=run_name
    )
    
    # reference model
    old_model = load_model(args.checkpoint_path, device)
    old_model.nfe = args.sample_time_step
    
    # data
    train_dataset = old_model.train_dataset
    train_sampler = KRepeatSampler(train_dataset, args.train_max_num, args.repeat_k, args.sample_batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn)

    val_loader = old_model.val_dataloader(bz=args.train_batch_size, shuffle=False)
    
    sample_batch_num = len(train_loader)
    total_batch_size = sample_batch_num*args.sample_batch_size  # total sample number e.g. 2048

    model = load_model(args.checkpoint_path, device)
    model.nfe = args.sample_time_step

    # scorer
    scorer_1 = EnergyScorer(method='xtb')

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, betas=[0.9, 0.99], weight_decay=args.weight_decay)

    global_epoch=0
    while True:
        samples = []
        log = {}
        
        # sample
        old_model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc=f"Epoch {global_epoch}  : sampling"):
                representations, conditions = batch
                # result = model.eval_sample_batch(batch)
                
                traj_bath, log_prob_batch, target_batch, idx_batch = old_model.sample_batch_traj(batch)

                # compute rewards
                reward_list = []
                target_mol_list = []
                predict_mol_list = []
                predict_list = []

                for traj, target, idx in zip(traj_bath, target_batch, idx_batch):
                    
                    # x1
                    # predict = traj[0]
                    # target_mol = decode_molecule(target, idx)
                    # predict_mol = decode_molecule(predict, idx) 
                    # reward = scorer_1(predict_mol, target_mol)
                    # print("========")
                    # print("x1")
                    # print(reward, rmsd_str(target_mol, predict_mol))
                    # rmsd_list_1.append(rmsd_str(target_mol, predict_mol))
                    
                    # x0
                    predict = traj[-1]
                    target_mol = decode_molecule(target, idx)
                    predict_mol = decode_molecule(predict, idx) 
                    reward = scorer_1(predict_mol, target_mol)
                    # print(reward)
                    
                    # calculate rmsd
                    # print("x0")
                    # print(reward, rmsd_str(target_mol, predict_mol))
                    # rmsd_list_2.append(rmsd_str(target_mol, predict_mol))
                                    
                    reward_list.append(reward)
                    target_mol_list.append(target_mol)
                    predict_mol_list.append(predict_mol)
                    predict_list.append(predict)
            
                # print(sum(rmsd_list_1) / len(rmsd_list_1))
                # print(sum(rmsd_list_2) / len(rmsd_list_2))
                samples.append(
                    {
                        "representations": representations,
                        "conditions":conditions,
                        "predict_mols": predict_mol_list,
                        'predicts': predict_list,  # unequal size for each element
                        # "trajs": traj_bath,
                        "target_mols": target_mol_list,
                        # "log_probs": torch.stack(log_prob_batch).squeeze(-1),
                        "rewards": torch.tensor(reward_list, device=model.device),
                    }
                )

        '''
            representations[dict]:
                size: [3,N]
                pos: [3,sum(size), 3]
                one_hot: [3,sum(size), 5]
                charge: [3,sum(size), 1]
                mask: [3,sum(size)]
                3 indictes the p1, p2 and (p1+p2)/2
            conditions[tensor]: [N,1]
            trajs[list]: [N]*tensor of vary size
            log_probs[tensor]: [N,T]
            rewards[tensor]: [N]
            advantages[tensor]: [N]
        '''
        samples = process_samples(samples, log=False)
        
        log['mean_reward'] = torch.mean(samples['rewards']).item()  

        stat_tracker = PerPromptStatTracker(global_std=True)
        advantages = stat_tracker.update(samples['target_mols'], samples['rewards'].cpu())
        advantages = torch.tensor(advantages, device=model.device)
        samples['advantages'] = advantages

        # train
        model.train()
        for inner_epoch_id in range(args.train_epoch):
            # shuffle
            perm = torch.randperm(total_batch_size, device=model.device)

            samples = reshuffle(samples, perm, model.device)

            # rebatch
            batch_samples = create_batches(samples, args.train_batch_size, model.device)

            total_loss = 0

            for j, sub_sample in tqdm(
                list(enumerate(batch_samples)),
                desc=f"Epoch {global_epoch}.{inner_epoch_id}: training",
                position=0,
            ):
                train_timesteps = utils.space_indices(model.ddpm.T, args.sample_time_step + 1)
                train_timesteps = train_timesteps[::-1]
                
                for k in range(args.sample_time_step):
                    # print_samples(sub_sample)
                    # print(sub_sample['representations']['size'])

                    advantages_clip = torch.clamp(
                        sub_sample["advantages"][:],
                        -args.adv_clip_max,
                        args.adv_clip_max,
                    )
                    normalized_advantages_clip = (advantages_clip / args.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)

                    # forward, add noise
                    t_int = torch.tensor([train_timesteps[k]]*args.train_batch_size).to(model.device).unsqueeze(-1)
                    xt, x0, timestep = model.ddpm.compute_xt(sub_sample['representations'], sub_sample['conditions'], t_int)
                    # gt velocity
                    v = model.ddpm.compute_label(timestep, x0, xt)

                    # prediction by reference model
                    with torch.no_grad():
                        old_prediction_v = old_model.ddpm.forward_once(xt, t_int, sub_sample['representations'], sub_sample['conditions'])

                    # prediction by model
                    prediction_v = model.ddpm.forward_once(xt, t_int, sub_sample['representations'], sub_sample['conditions'])

                    # implicit positive prediction v
                    positive_prediction_v = args.nft_beta * prediction_v + (1-args.nft_beta) * old_prediction_v.detach()
                    negative_prediction_v = (1+args.nft_beta)*old_prediction_v.detach() - args.nft_beta * prediction_v

                    # convert v to x0
                    positive_prediction_x0 = xt - timestep * positive_prediction_v
                    negative_prediction_x0 = xt - timestep * negative_prediction_v

                    # print(positive_prediction_x0.shape, negative_prediction_x0.shape)

                    # adaptive loss weight in diffusionNFT
                    with torch.no_grad():
                        positive_weight_factor = (
                            torch.abs(positive_prediction_x0.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                        negative_weight_factor = (
                            torch.abs(negative_prediction_x0.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    

                    # policy loss
                    positive_loss = ((positive_prediction_x0 - x0) ** 2 / positive_weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    negative_loss = ((negative_prediction_x0 - x0) ** 2 / negative_weight_factor).mean(dim=tuple(range(1, x0.ndim)))
                    

                    # reward expand from atom-level to molecule-level
                    r = r.repeat_interleave(sub_sample['representations'][0]['size'])

                    ori_policy_loss = r * positive_loss / args.nft_beta + (1.0 - r) * negative_loss / args.nft_beta
                    loss = (ori_policy_loss * args.adv_clip_max).mean()
                    # ori_policy_loss = r * positive_loss + (1.0 - r) * negative_loss
                    # loss = ori_policy_loss.mean()
                    # print(loss)
                    # print(loss.grad_fn)

                    # kl loss
                    if args.beta>0:
                        kl_loss = ((prediction_v - old_prediction_v) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        )
                        loss += args.beta * torch.mean(kl_loss)
                    
                    # print(f'loss:{loss}, reward:{torch.mean(sub_sample["rewards"])}')
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

        # eval
        model.eval()
        model.ddpm.opt = args
        res, rmsds = model.eval_rmsd(
            val_loader,
            write_xyz=False,
            bz=args.sample_batch_size,
            refpath="ref_ts",
            max_num_batch=5,
            global_epoch=global_epoch
        )
        
        log['mean_rmsd'] = np.mean(rmsds)
        log['median_rmsd'] = np.median(rmsds)   
        
        wandb.log(log)

        # update reference model
        update_model_ema(old_model, model, args.ema_update_eta)
        
        global_epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint_path", type=str, default='./reactot-pretrained.ckpt')

    # sample
    parser.add_argument("--repeat_k", type=int, default=8)
    parser.add_argument("--sample_time_step", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=32)
    
    # train
    parser.add_argument("--train_max_num", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_epoch", type=int, default=1)
    parser.add_argument("--rl_type", type=str, default='grpo')

    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.1)
    
    # eval
    parser.add_argument("--solver", type=str, choices=["ddpm", "ei", "ode"], default="ode")
    parser.add_argument("--nfe", type=int, default=50)
    
    # ei
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--diz", type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize", action="store_true")

    # ode
    parser.add_argument("--method", type=str, default="midpoint") 
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_update_eta", type=float, default=0.3)

    # nft related
    parser.add_argument("--nft_beta", type=float, default=1)


    args = parser.parse_args()
    main(args)