import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from reactot.trainer.pl_trainer import SBModule
from reactot.model.leftnet import LEFTNet

from ts_rl.sampler import KRepeatSampler
from ts_rl.energy_scorer import EnergyScorer
from ts_rl.stat_tracking import PerMoleculeStatTracker


# torch.serialization.add_safe_globals([LEFTNet])

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



import torch

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
    
    reference_model = load_model(args.checkpoint_path)
    reference_model.nfe = args.sample_time_step
    
    train_dataset = reference_model.val_dataset
    train_sampler = KRepeatSampler(train_dataset, args.repeat_k, args.sample_batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn)

    sample_batch_num = len(train_loader)
    total_batch_size = sample_batch_num*args.sample_batch_size  # total sample number e.g. 2048

    model = load_model(args.checkpoint_path)
    model.nfe = args.sample_time_step

    samples = []

    scorer = EnergyScorer()
    

    global_epoch=0
    while True:
        # sample
        model.eval()
        for batch in tqdm(train_loader):
            representations, conditions = batch
            traj_bath, log_prob_batch, target_batch, idx_batch = model.sample_batch_traj(batch)

            # compute rewards
            reward_list = []
            target_mol_list = []
            predict_mol_list = []

            for traj, target, idx in zip(traj_bath, target_batch, idx_batch):
                predict = traj[-1]
                target_mol = decode_molecule(target, idx)
                predict_mol = decode_molecule(predict, idx) 
                # reward = scorer(predict_mol, target_mol)
                
                reward_list.append(0)
                target_mol_list.append(target_mol)
                predict_mol_list.append(predict_mol)
            

            tracker = PerMoleculeStatTracker()
            advantages = tracker.update(target_mol_list, reward_list, args.rl_type)
            advantages = torch.tensor(advantages, device=model.device)  # [N]


            samples.append(
                {
                    "representations": representations,
                    "conditions":conditions,
                    "trajs": traj_bath,
                    "target_mols": target_mol_list,
                    "log_probs": torch.stack(log_prob_batch).squeeze(-1),
                    "rewards": torch.tensor(reward_list, device=model.device),
                    "advantages": advantages
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
            log_probs[tensor]: [N,2]
            rewards[tensor]: [N]
            advantages[tensor]: [N]
        '''
        samples = process_samples(samples, log=True)

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
                train_timesteps = [step_index  for step_index in range(args.train_timestep_num)]
                for k in train_timesteps:
                    advantages = torch.clamp(
                            sub_sample["advantages"],
                            -args.adv_clip_max,
                            args.adv_clip_max,
                        )
                    

                    assert None


        global_epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--repeat_k",     type=int,   default=2)
    
    parser.add_argument("--sample_time_step",     type=int,   default=2)
    parser.add_argument("--sample_batch_size",    type=int,   default=128)
    parser.add_argument("--train_batch_size",     type=int,   default=16)
    parser.add_argument("--train_epoch",     type=int,   default=16)
    parser.add_argument("--train_batch_num",     type=int,   default=16)
    parser.add_argument("--rl_type",     type=str,   default='grpo')

    parser.add_argument("--train_timestep_num",     type=int,   default=2)
    parser.add_argument("--adv_clip_max", type=float, default=5.0)


    
    parser.add_argument("--checkpoint_path",     type=str, default='./reactot-pretrained.ckpt')

    # ei
    parser.add_argument("--order",          type=int, default=1)
    parser.add_argument("--diz",            type=str, default="linear", choices=["linear", "quad"])
    parser.add_argument("--normalize",      action="store_true")

    # ode
    parser.add_argument("--method",         type=str,   default="midpoint") 
    parser.add_argument("--atol",           type=float, default=1e-2)
    parser.add_argument("--rtol",           type=float, default=1e-2)

    args = parser.parse_args()
    main(args)