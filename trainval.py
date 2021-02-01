import torch
import numpy as np
import argparse
import pandas as pd
import sys
import os
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from src import utils as ut
import torchvision
from torchvision import transforms      #helen added this
from haven import haven_utils as hu
from haven import haven_chk as hc

from src import datasets, models
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler
from src import wrappers
from haven import haven_wizard as hw
from PIL import Image                   #helen added this

def trainval(exp_dict, savedir, args):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """


    # set seed
    # ==================
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    #helen commented out the following lines to hard code in that the device was 'cpu' to resolve errors
    #if args.use_cuda:
        #device = 'cuda'
        #torch.cuda.manual_seed_all(seed)
        #assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    #else:
    device = 'cpu'

    print('Running on device: %s' % device)
    
    # Dataset
    # Load val set and train set
    val_set = datasets.get_dataset(dataset_name=exp_dict["dataset"], split="val",
                                   transform=exp_dict.get("transform"),
                                   datadir=args.datadir)
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train", 
                                     transform=exp_dict.get("transform"),
                                     datadir=args.datadir)
    
    # Load train loader, val loader, and vis loader
    train_loader = DataLoader(train_set, 
                            sampler=RandomSampler(train_set,
                            replacement=True, num_samples=max(min(500, 
                                                            len(train_set)), 
                                                            len(val_set))),
                            batch_size=exp_dict["batch_size"])

    val_loader = DataLoader(val_set, shuffle=False, batch_size=exp_dict["batch_size"])
    vis_loader = DataLoader(val_set, sampler=ut.SubsetSampler(train_set,
                                                     indices=[0, 1, 2]),
                            batch_size=1)

    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).cuda()
    opt = torch.optim.Adam(model_original.parameters(), 
                        lr=1e-5, weight_decay=0.0005)

    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).cuda()

    score_list = []

    # Checkpointing
    # =============
    #score_list_path = os.path.join(savedir, "score_list.pkl")      #helen commented out these three lines and hard coded the model and opt paths to resolve errors
    #model_path = os.path.join(savedir, "model_state_dict.pth")
    #opt_path = os.path.join(savedir, "opt_state_dict.pth")
    score_list_path = '/Users/helenpropson/Documents/git/marepesca/results/testresults/score_list.pkl'      #helen added this
    model_path = '/Users/helenpropson/Documents/git/marepesca/results/testresults/model_state_dict.pth'     #helen added this
    opt_path = '/Users/helenpropson/Documents/git/marepesca/results/testresults/opt_state_dict.pth'         #helen added this

    #helen hard coded that the experiment would resume instead of restarting from epoch 0
    #if os.path.exists(score_list_path):
        # resume experiment
    score_list = hu.load_pkl(score_list_path)   #helen changed this from ut.load_pkl to hu.load_pkl to resolve error
    model.load_state_dict(torch.load(model_path))
    opt.load_state_dict(torch.load(opt_path))
    s_epoch = score_list[-1]["epoch"] + 1

    #else:
        # restart experiment
        #score_list = []
        #s_epoch = 0

    # ***************            helen added this code
    im = Image.open("/Users/helenpropson/Documents/git/marepesca/tank.jpg")
    # im.show()  #this line will display the image you are running the model on if uncommented

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = transforms.Normalize(mean=mean, std=std)

    data_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])   #transformations we will use on our image
    im_new = data_transform(im)                                                         #transforms the image into a tensor and normalizes it
    im_final = im_new.unsqueeze(0)                                                      #adds another dimension so image is the correct shape for the model
    print("now trying helen's code")                                                    #print statement for debugging
    #model.vis_on_batch_helen(im_final, f'im_new')    #uncomment this line to run model on image

    # ***************            this is the end of helen's code

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}

        # visualize
        model.vis_on_loader(vis_loader, savedir=os.path.join(savedir, "images"))

        print("after vis_on_loader")    #helen add this print statement as an update while iterating

        # validate
        score_dict.update(model.val_on_loader(val_loader))

        print("after validate")

        # train
        score_dict.update(model.train_on_loader(train_loader))

        print("after train")

        # Add score_dict to score_list
        score_list += [score_dict]

        # Report and save
        print(pd.DataFrame(score_list).tail())
        hu.save_pkl(score_list_path, score_list)
        hu.torch_save(model_path, model.state_dict())
        hu.torch_save(opt_path, opt.state_dict())
        print("Saved in %s" % savedir)

if __name__ == '__main__':
    # 8. define a list of experiments
    import exp_configs

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+",
                        help='Define which exp groups to run.')
    parser.add_argument('-sb', '--savedir_base', default=None,
                        help='Define the base directory where the experiments will be saved.')
    parser.add_argument('-d', '--datadir', default=None,
                        help='Define the dataset directory.')
    parser.add_argument("-r", "--reset",  default=0, type=int,
                        help='Reset or resume the experiment.')
    parser.add_argument("--debug",  default=False, type=int,
                        help='Debug mode.')
    parser.add_argument("-ei", "--exp_id", default=None,
                        help='Run a specific experiment based on its id.')
    parser.add_argument("-j", "--run_jobs", default=0, type=int,
                        help='Run the experiments as jobs in the cluster.')
    parser.add_argument("-nw", "--num_workers", type=int, default=0,
                        help='Specify the number of workers in the dataloader.')
    parser.add_argument("-v", "--visualize_notebook", type=str, default='',
                        help='Create a jupyter file to visualize the results.')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)

    args, others = parser.parse_known_args()

    # 9. Launch experiments using magic command
    hw.run_wizard(func=trainval, exp_groups=exp_configs.EXP_GROUPS, args=args)

#command helen uses for trainval: python /Users/helenpropson/Documents/git/marepesca/DeepFish-master/trainval.py -e loc -sb /Users/helenpropson/Documents/git/marepesca/results -d /Users/helenpropson/Documents/git/marepesca/DeepFish -r 1
