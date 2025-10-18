
def main(args):
    
    # outer iteration
    while True:
    
        # sample
        for i in range(args.sample_batch_num):
            # (image, next_image, rewards, logp)
            pass
        
        # merge sampled data batch into one buffer
        # calculate advantages
        
        # train
        for i in range(args.train_epoch):
            # rebatch data in buffer
            for j in range(args.train_batch_num):
                for k in range(args.train_timestep_num):
                    # calculate old_logp
                    # update
                    pass

if __name__ == "__main__":
    pass