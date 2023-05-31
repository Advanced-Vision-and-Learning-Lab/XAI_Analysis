import numpy as np
import torch
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from captum.attr import *
from captum._utils.models.linear_model import SkLearnLasso, SkLearnLinearModel
from pytorch_grad_cam import HiResCAM
from barbar import Bar
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu
import pdb
import tqdm

def transform_list(saved_list):
    'Transform list to numpy array'
    saved_list = np.concatenate(saved_list,axis=0)
    return saved_list

def get_feature_mask(x_batch):
    """Compute Slic segmentation for multiple images"""
    batch_num = x_batch.shape[0]
    feature_mask = np.zeros((batch_num,x_batch.shape[-2],x_batch.shape[-1]))
    
    for img in range(0, batch_num):
        feature_mask[img] = slic(x_batch[img].permute(1,2,0).to(torch.double).cpu().numpy(), 
                            n_segments=100, channel_axis=-1, slic_zero=True,
                            start_label=0)
    
    return feature_mask

# Function to reshape the input tensor for ViT model
def vit_reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :, :].reshape(tensor.size(0),
                                        height, width, tensor.size(2))

    # Similar to CNN case, bring channels to first dimension
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def inverse_normalize(tensor, mean=(0,0,0), std=(1,1,1)):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def eurosat_to_rgb(x):
    x = x[[3, 2, 1], :, :].float()
    x.sub_(x.min())
    return x.div_(x.max())

def get_magnitude(x):
    x = (x[0, :, :]).float()
    x = x.expand(2, -1, -1)
    return x


def threshold_otsu_pytorch(x, Params) -> float:
    #Convert tensor from 0 - 1 to 0 - 255 (TBD)
    # x = torch.tensor(x) * 255.0
    x = x.clone()
    x = inverse_normalize(x, Params["mean"], Params["std"])
    x *= 255
    x = x.long()
    mask = torch.zeros((x.shape))
    for img in range(0,x.shape[0]):
        for channel in range(0,x.shape[1]):
            temp_img = (x[img][channel].cpu().numpy()*255).astype(np.uint8)
            thresh = threshold_otsu(temp_img)
            binary = temp_img > thresh
            mask[img][channel] = torch.as_tensor(binary)
    x_agg = mask.prod(axis=1).unsqueeze(1)    
    #Convert to binary if multichannel
    x_thres = (x_agg > 0).float()
    return x_thres

def plot_feature_mask(x_batch, s_batch, feature_mask, Params):
    x_batch[0] = inverse_normalize(x_batch[0], Params["mean"], Params["std"])
    x_plot = x_batch[0].permute(1,2,0)
    fig, ax = plt.subplots(1,3)
    ax[0].imshow(x_plot[:,:,0].cpu().numpy())
    ax[1].imshow(s_batch[0][0].cpu().numpy())
    ax[2].imshow(mark_boundaries(x_plot[:,:,0].cpu().numpy(), feature_mask[0].cpu().numpy().astype(int)))
    ax[0].set_title("Input Image", fontsize=8)
    ax[1].set_title("Binary Threshold mask", fontsize=8)
    ax[2].set_title("Feature mask", fontsize=8)
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    plt.show()


def get_attributions(dataloaders, dataset, model, device, Params, parallel=False):
    x_batch_all = []
    y_batch_all = []
    s_batch_all = []
    a_batch_lime_all = []
    a_batch_gradcam_all = []
    a_batch_gradshap_all = []
    a_batch_saliency_all = []
    a_batch_occ_all = []
    GT_val = np.array(0)

    #Subsample data from dataset["test"]
    images, labels, index = iter(dataloaders).__next__()
    for sample in tqdm.tqdm(range(0, len(index))):
        if dataset == "MSTAR":
            image = get_magnitude(images[sample])
        else:
            image = images[sample] 
        inputs, classes = image, labels[sample]
        GT_val = np.concatenate((GT_val, classes.cpu().numpy()),axis = None)
        x_batch, y_batch = inputs.unsqueeze(0).to(device), classes.unsqueeze(0).to(device)
        feature_mask = get_feature_mask(x_batch)
        feature_mask = torch.Tensor(feature_mask).to(torch.long).to(device)
        s_batch = threshold_otsu_pytorch(x_batch, Params)

        #plot_feature_mask(x_batch, s_batch, feature_mask, Params)
        
        #LIME
        if dataset == "Eurosat_MSI":
            x_batch, y_batch = x_batch.to(torch.long), y_batch.to(torch.long)
            x_batch = x_batch.to(torch.float32) # for eurosat dataset
            
        a_batch_lime = Lime(model, interpretable_model=SkLearnLinearModel("linear_model.Ridge")).attribute(inputs=x_batch, target=y_batch, 
                    feature_mask=feature_mask, n_samples=16, perturbations_per_eval=32, return_input_shape=True, 
                    show_progress=False).sum(axis=1).cpu().numpy()
        a_batch_lime_all.append(a_batch_lime)
        
        #GRADCAM
        #Extract the model name
        if (parallel):
            model_name = model.module.__class__.__name__ 
        else:
            model_name = model.__class__.__name__

        if(model_name == "ConvNeXt"):
            if (parallel):
                gradcam = HiResCAM(model, target_layers=[model.module.features[-1][-1]])                
            else:
                gradcam = HiResCAM(model, target_layers=[model.features[-1][-1]]) 
            x_batch = torch.autograd.Variable(x_batch, requires_grad=True)
            a_batch_gradcam = gradcam(input_tensor=x_batch)
        
        elif(model_name == "VisionTransformer"):
            if(parallel):
                gradcam = HiResCAM(model, target_layers=[model.module.encoder.layers[-2].ln_1], reshape_transform=vit_reshape_transform)
            else:
                gradcam = HiResCAM(model, target_layers=[model.encoder.layers[-2].ln_1], reshape_transform=vit_reshape_transform)
            x_batch = torch.autograd.Variable(x_batch, requires_grad=True)
            a_batch_gradcam = gradcam(input_tensor=x_batch)

        if(model_name == "FocalNet"):
            if (parallel):
                gradcam = HiResCAM(model, target_layers=[model.module.layers[-2].downsample.proj])
            else:
                gradcam = HiResCAM(model, target_layers=[model.layers[-2].downsample.proj])
            x_batch = torch.autograd.Variable(x_batch, requires_grad=True)
            a_batch_gradcam = gradcam(input_tensor=x_batch)
            

        a_batch_gradcam_all.append(a_batch_gradcam)

        #GRADSHAP
        rand_img_dist = torch.randn(x_batch.shape).to(device)
        a_batch_gradshap = GradientShap(model).attribute(inputs=x_batch, target=y_batch, 
                            baselines=rand_img_dist).sum(axis=1).cpu().detach().numpy()
        a_batch_gradshap_all.append(a_batch_gradshap)

        #SALIENCY
        a_batch_saliency = Saliency(model).attribute(inputs=x_batch, target=y_batch, 
                            abs=True).sum(axis=1).cpu().numpy()
        a_batch_saliency_all.append(a_batch_saliency)

        #OCCLUSION
        a_batch_occ = Occlusion(model).attribute(inputs=x_batch, target=y_batch, strides=(x_batch.shape[1], 8, 8),
                            sliding_window_shapes=(x_batch.shape[1], 15, 15)).sum(axis=1).cpu().numpy()
        a_batch_occ_all.append(a_batch_occ)
        s_batch_all.append(s_batch.cpu().numpy())
        x_batch_all.append(x_batch.cpu().detach().numpy())

    y_batch_all = GT_val[1:]

    #Transform list into numpy arrays for metric calculation
    a_batch_lime_all = transform_list(a_batch_lime_all)
    a_batch_gradcam_all = transform_list(a_batch_gradcam_all)
    a_batch_gradshap_all = transform_list(a_batch_gradshap_all)
    a_batch_saliency_all = transform_list(a_batch_saliency_all)
    a_batch_occ_all = transform_list(a_batch_occ_all)
    s_batch_all = transform_list(s_batch_all)
    x_batch_all = transform_list(x_batch_all)

    return {"Lime": a_batch_lime_all, "HiResCAM": a_batch_gradcam_all, "GradientShap": a_batch_gradshap_all, 
            "Saliency": a_batch_saliency_all, "Occlusion": a_batch_occ_all}, x_batch_all, y_batch_all, s_batch_all