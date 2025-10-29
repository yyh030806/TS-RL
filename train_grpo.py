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
from ts_rl.stat_tracking import PerMoleculeStatTracker


torch.serialization.add_safe_globals([LEFTNet])

device = 'cuda'

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
            samples[k] = [samples[k][i] for i in perm]
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


def main(args):
    
    # logger
    project_name = "ts_rl"
    group_name = "grpo"

    current_time = datetime.datetime.now()
    timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{group_name}_{timestamp_str}"

    wandb.init(
        project=project_name, 
        group=group_name, 
        name=run_name
    )
    
    wandb.init(project=project_name, group=group_name, name=run_name)
    
    # reference model
    reference_model = load_model(args.checkpoint_path)
    reference_model.nfe = args.sample_time_step
    
    # data
    train_dataset = reference_model.train_dataset
    train_sampler = KRepeatSampler(train_dataset, args.train_max_num, args.repeat_k, args.sample_batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn)

    val_loader = reference_model.val_dataloader(bz=args.train_batch_size, shuffle=False)
    
    sample_batch_num = len(train_loader)
    total_batch_size = sample_batch_num*args.sample_batch_size  # total sample number e.g. 2048

    model = load_model(args.checkpoint_path)
    model.nfe = args.sample_time_step

    # scorer
    scorer_1 = EnergyScorer(method='xtb')

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, betas=[0.9, 0.99], weight_decay=args.weight_decay)

    global_epoch=0
    while True:
        samples = []
        log = {}
        
        # sample
        old_model = model
        old_model.eval()
        for batch in tqdm(train_loader, desc=f"Epoch {global_epoch}  : sampling"):
            representations, conditions = batch
            # result = model.eval_sample_batch(batch)
            
            traj_bath, log_prob_batch, target_batch, idx_batch = old_model.sample_batch_traj(batch)

            # compute rewards
            reward_list = []
            target_mol_list = []
            predict_mol_list = []

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
                
                # calculate rmsd
                # print("x0")
                # print(reward, rmsd_str(target_mol, predict_mol))
                # rmsd_list_2.append(rmsd_str(target_mol, predict_mol))
                                
                reward_list.append(reward)
                target_mol_list.append(target_mol)
                predict_mol_list.append(predict_mol)
            
            # print(sum(rmsd_list_1) / len(rmsd_list_1))
            # print(sum(rmsd_list_2) / len(rmsd_list_2))
            samples.append(
                {
                    "representations": representations,
                    "conditions":conditions,
                    "trajs": traj_bath,
                    "target_mols": target_mol_list,
                    "log_probs": torch.stack(log_prob_batch).squeeze(-1),
                    "rewards": torch.tensor(reward_list, device=model.device),
                    # "advantages": advantages
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

        tracker = PerMoleculeStatTracker()
        advantages = tracker.update(samples['target_mols'], samples['rewards'].tolist(), args.rl_type)
        advantages = torch.tensor(advantages, device=model.device)  # [N]

        samples['advantages'] = advantages
        
        log['mean_reward'] = torch.mean(samples['rewards']).item()  

        # train
        model.train()
        for inner_epoch_id in range(args.train_epoch):
            # shuffle
            perm = torch.randperm(total_batch_size, device=model.device)

            samples = reshuffle(samples, perm, model.device)

            # rebatch
            batch_samples = create_batches(samples, args.train_batch_size, model.device)


            for j, sub_sample in tqdm(
                list(enumerate(batch_samples)),
                desc=f"Epoch {global_epoch}.{inner_epoch_id}: training",
                position=0,
            ):
                train_timesteps = utils.space_indices(model.ddpm.T, args.sample_time_step + 1)
                train_timesteps = train_timesteps[::-1]
                for k in range(args.sample_time_step):

                    sample = torch.cat([ traj[k] for traj in sub_sample['trajs']], dim=0)
                    prev_sample = torch.cat([ traj[k+1] for traj in sub_sample['trajs']], dim=0)
                    # log_prob: list [float], len=batch_size
                    # prev_sample_mean: list [(num_atom, 3)], len=batch_size
                    # std_dev_t: float
                    log_prob, prev_sample_mean, std_dev_t = model.sample_log_prob(sub_sample['representations'], sub_sample['conditions'], sample, prev_sample,
                                                                    time_step=train_timesteps[k], prev_time_step=train_timesteps[k+1])

                    with torch.no_grad():
                        _, prev_sample_mean_ref, _ = reference_model.sample_log_prob(sub_sample['representations'], sub_sample['conditions'], sample, prev_sample,
                                                                    time_step=train_timesteps[k], prev_time_step=train_timesteps[k+1])
                    
                    advantages = torch.clamp(
                            sub_sample["advantages"],
                            -args.adv_clip_max,
                            args.adv_clip_max,
                        )
                    
                    log_prob = torch.stack(log_prob)
                    ratio = torch.exp(log_prob - sub_sample["log_probs"][:,k])
                    unclipped_loss = -advantages * ratio
                    clipped_loss = -advantages * torch.clamp(
                        ratio,
                        1.0 - args.clip_range,
                        1.0 + args.clip_range,
                    )
                    policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                    
                    prev_means = torch.stack([t.mean() for t in prev_sample_mean])
                    prev_means_ref = torch.stack([t.mean() for t in prev_sample_mean_ref])

                    if args.beta > 0:
                        kl_loss = ((prev_means - prev_means_ref) ** 2) / (2 * std_dev_t ** 2)
                        kl_loss = torch.mean(kl_loss)
                        loss = policy_loss + args.beta * kl_loss
                    else:
                        loss = policy_loss
                    
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
        )
        
        log['mean_rmsd'] = np.mean(rmsds)
        log['median_rmsd'] = np.median(rmsds)   
        
        wandb.log(log)
        
        global_epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--checkpoint_path", type=str, default='./reactot-pretrained.ckpt')

    # sample
    parser.add_argument("--repeat_k", type=int, default=4)
    parser.add_argument("--sample_time_step", type=int, default=10)
    parser.add_argument("--sample_batch_size", type=int, default=16)
    
    # train
    parser.add_argument("--train_max_num", type=int, default=16)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_epoch", type=int, default=4)
    parser.add_argument("--rl_type", type=str, default='grpo')

    parser.add_argument("--adv_clip_max", type=float, default=5.0)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=1)
    
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    


    args = parser.parse_args()
    main(args)