import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from reactot.trainer.pl_trainer import SBModule
from reactot.model.leftnet import LEFTNet

from ts_rl.sampler import KRepeatSampler
from ts_rl.energy_scorer import EnergyScorer


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

def main(args):
    
    reference_model = load_model(args.checkpoint_path)
    
    train_dataset = reference_model.val_dataset
    train_sampler = KRepeatSampler(train_dataset, args.repeat_k, args.sample_batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn)
    

    while True:
        old_model = load_model(args.checkpoint_path)
        old_model.nfe = args.sample_time_step
        
        # sample
        idx_list = []
        traj_list = []
        log_prob_list = []
        target_list = []
        
        for batch in tqdm(train_loader):
            traj_bath, log_prob_batch, target_batch, idx_batch = old_model.sample_batch_traj(batch)
            traj_list.extend(traj_bath)
            log_prob_list.extend(log_prob_batch)
            target_list.extend(target_batch)
            idx_list.extend(idx_batch)
        
        # compute rewards
        scorer = EnergyScorer()
        reward_list = []
        for traj, target, idx in zip(traj_list, target_list, idx_list):
            predict = traj[-1]
            target_mol = decode_molecule(target, idx)
            predict_mol = decode_molecule(predict, idx) 
            reward = scorer(predict_mol, target_mol)
            reward_list.append(reward)
            
        # compute advantages
        
        # train
        for i in range(args.train_epoch):
            # rebatch data in buffer
            for j in range(args.train_batch_num):
                for k in range(args.train_timestep_num):
                    # calculate old_logp
                    # update
                    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--repeat_k",     type=int,   default=4)
    
    parser.add_argument("--sample_time_step",     type=int,   default=5)
    parser.add_argument("--sample_batch_size",    type=int,   default=128)
    parser.add_argument("--train_batch_size",     type=int,   default=16)
    parser.add_argument("--train_epoch",     type=int,   default=16)
    
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