import sys, os, pprint

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)
from haven import haven_utils as hu
import numpy as np
from src import datasets, models, wrappers
import argparse
from tqdm.auto import tqdm

from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms      #helen added this
import torch
from torch.backends import cudnn
from torch import nn
import torchvision.transforms as T
import exp_configs


cudnn.benchmark = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--datadir',
                        type=str, default='/mnt/public/datasets/DeepFish')
    parser.add_argument("-e", "--exp_config", default='loc')
    parser.add_argument("-uc", "--use_cuda", type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if args.use_cuda else 'cpu')

    exp_dict = exp_configs.EXP_GROUPS[args.exp_config][0]
    train_set = datasets.get_dataset(dataset_name=exp_dict["dataset"],
                                     split="train", 
                                     transform=exp_dict.get("transform"),
                                     datadir=args.datadir)
    
  
    # Create model, opt, wrapper
    model_original = models.get_model(exp_dict["model"], exp_dict=exp_dict).to('cpu') #.cuda()
    opt = torch.optim.Adam(model_original.parameters(), 
                        lr=1e-5, weight_decay=0.0005)

    model = wrappers.get_wrapper(exp_dict["wrapper"], model=model_original, opt=opt).to('cpu') #.cuda()

    if args.exp_config == 'loc':
        batch = torch.utils.data.dataloader.default_collate([train_set[3]])
    else:
        batch = torch.utils.data.dataloader.default_collate([train_set[0]])

    #***************            helen added this code
    im = Image.open("/Users/helenpropson/Documents/git/marepesca/tank.jpg")
    #im.show()  #this line will display the image you are running the model on if uncommented

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize_transform = transforms.Normalize(mean=mean, std=std)

    #print("creating tensor") #print line statement used to debug
    data_transform = transforms.Compose([transforms.ToTensor(), normalize_transform])   #transformations we will use on our image
    #these are the transformations listed under DeepFish/src/datasets/__init__.py / for the function get_transformer: data_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(), normalize_transform])

    im_new = data_transform(im)     #transforms the image into a tensor and normalizes it
    im_final = im_new.unsqueeze(0)  #adds another dimension so image is the correct shape for the model

    #print(im_final)    #this should now print a tensor if uncommented
    #print(im_final.shape)  #this line will help to check the shape of im_final is correct if uncommented

    #plt.imshow(im_new.numpy()[0], cmap='gray')   #these two lines should display the image as grayscale if uncommented
    #plt.show()

    #print("we have passed this line")  #print statement to see if there are errors with loading/transforming data

    #***************            this is the end of helen's code

    for e in range(50):
        score_dict = model.train_on_batch(batch)
        print(e, score_dict)

        model.vis_on_batch(batch, f'single_image_{args.exp_config}.png')
        #hu.save_image("pics", model.vis_on_batch(batch, view_support=True)[0])


        # ***************            helen added this code
        #if e==0:       #change e==0 to the epoch number you want to test your image at
            #model.vis_on_batch_helen(im_final, f'im_new')      #this line will run your image through the model and display the prediction if uncommented
        # ***************            this is the end of helen's code


        # validate on batch
        val_dict = model.val_on_batch(batch)
        pprint.pprint(val_dict)

#command helen uses for train_single_image: python /Users/helenpropson/Documents/git/marepesca/DeepFish-master/scripts/train_single_image.py -e loc -d /Users/helenpropson/Documents/git/marepesca/DeepFish
