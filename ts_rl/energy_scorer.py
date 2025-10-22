from reactot.analyze.geomopt import calc_deltaE


class EnergyScorer:
    def __init__(self):
        pass
    
    def __call__(self, st_tar, st_pred):
        # deltaE = calc_deltaE(st_tar, st_pred)
        # return abs(deltaE)
        
        return 0
        
if __name__ == '__main__':
    st_tar = 'H 0 0 0; Cl 0 0 1'
    st_pred = 'H 0 0 0; F 0 0 1'
    
    scorer = EnergyScorer()
    score = scorer(st_tar, st_pred)
    
    print(score)