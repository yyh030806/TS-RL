import numpy as np
from collections import deque
import torch

class PerMoleculeStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_molecules = set()

    def update(self, molecules, rewards, type='grpo'):
        molecules = np.array(molecules)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(molecules)
        advantages = np.empty_like(rewards)*0.0
        for molecule in unique:
            molecule_rewards = rewards[molecules == molecule]
            if molecule not in self.stats:
                self.stats[molecule] = []
            self.stats[molecule].extend(molecule_rewards)
            self.history_molecules.add(hash(molecule))  # Add hash of molecule to history_molecules
        for molecule in unique:
            self.stats[molecule] = np.stack(self.stats[molecule])
            molecule_rewards = rewards[molecules == molecule]  # Fix: Recalculate molecule_rewards for each molecule
            mean = np.mean(self.stats[molecule], axis=0, keepdims=True)
            if self.global_std:
                std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
            else:
                std = np.std(self.stats[molecule], axis=0, keepdims=True) + 1e-4
            if type=='grpo':
                advantages[molecules == molecule] = (molecule_rewards - mean) / std
            elif type=='rwr':
                # advantages[molecules == molecule] = (molecule_rewards - mean) / std
                advantages[molecules == molecule] = molecule_rewards
                # advantages[molecules == molecule] = torch.softmax(torch.tensor(molecule_rewards), dim=0).numpy()
            elif type=='sft':
                advantages[molecules == molecule] = (torch.tensor(molecule_rewards) == torch.max(torch.tensor(molecule_rewards))).float().numpy()
            elif type=='dpo':
                # Get the advantages of the current molecule
                molecule_advantages = torch.tensor(molecule_rewards)
                # Find the indices of the maximum and minimum values
                max_idx = torch.argmax(molecule_advantages)
                min_idx = torch.argmin(molecule_advantages)
                # If all rewards in a group are the same
                if max_idx == min_idx:
                    min_idx = 0
                    max_idx = 1
                result = torch.zeros_like(molecule_advantages).float()
                # Set the maximum index to 1, minimum index to -1
                result[max_idx] = 1.0
                result[min_idx] = -1.0
                advantages[molecules == molecule] = result.numpy()
                # print("reward difference one group", molecule_advantages[max_idx]-molecule_advantages[min_idx])
            
        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_molecules = len(self.history_molecules)
        return avg_group_size, history_molecules
    
    def clear(self):
        self.stats = {}

def main():
    tracker = PerMoleculeStatTracker()
    
    molecules = ['a', 'b', 'a', 'c', 'b', 'a']
    rewards = [1, 2, 3, 4, 5, 6]
    advantages = tracker.update(molecules, rewards, 'dpo')
    print("Advantages:", advantages)
    avg_group_size, history_molecules = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Molecules:", history_molecules)
    tracker.clear() # important!!!
    print("Stats after clear:", tracker.stats)
    
if __name__ == "__main__":
    main()