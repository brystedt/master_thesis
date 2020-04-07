from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
from os import listdir


def load_data(rgb = False, dim = (192, 240)):
    prior_path = 'priors_all_copy'
    tumor_path = 'tumors'
    input_images = []
    target_masks = []
    passed_img = 0
    
    #center crop coordinates
    left = 0
    bottom = dim[1] - ((dim[1]-dim[0])/2)
    right = dim[0]
    top = (dim[1]-dim[0])/2
    
    if rgb:
        color_setting = 'RGB'
    else:
        color_setting = 'L'
        
    for filename in sorted(listdir(prior_path)):
        if "DS_Store" in filename:
            pass
        elif "mask" not in filename:
            try:
                #open image
                img_data = Image.open(prior_path + '/' + filename)
                width, height = img_data.size
                
                r = 143 #1 pixel = 0.07mm, 10mm recommended radius
                
                if rgb == True:
                    img_data = img_data.convert('RGB')
                else: 
                    img_data = img_data
                
                #open mask
                maskname = filename.split('.')[0] + "_mask.png"
                pix_data = Image.open(prior_path + '/' + maskname)
                
                #resize input image
                img_resized = img_data.resize(dim)
                img_cropped = img_resized.crop((left, top, right, bottom)) 
                data = np.array(img_cropped)
                data = np.transpose(data)
                input_images.append([data])

                #find pixel
                data = np.array(pix_data)
                pixel_coord = np.where(data == True)                
                xmin = pixel_coord[1]-r
                ymin = pixel_coord[0]-r
                xmax = pixel_coord[1]+r
                ymax = pixel_coord[0]+r
                
                #create mask from pixel coord and recommended radius
                mask_im = Image.new(color_setting, (width,height), (0))
                draw = ImageDraw.Draw(mask_im)
                draw.ellipse((xmin, ymin, xmax, ymax), fill=(255)) #kolla ordningen på coord               
                
                #apply gaussian blur
                mask_im_blurred = mask_im.filter(ImageFilter.GaussianBlur(radius = 25))
                
                #resize mask to dim
                mask_resized = mask_im_blurred.resize(dim)
                mask_cropped = mask_resized.crop((left, top, right, bottom)) 
                mask_data = np.array(mask_cropped)
                mask_data = np.transpose(mask_data)
                
                #add mask to mask array
                target_masks.append([mask_data])

                  
                    
            except (UnidentifiedImageError, NameError):
                passed_img += 1

    input_images = np.asarray(input_images)
    target_masks = np.asarray(target_masks)
        
    #remove filenames (only keep input and target)
    X_img = np.squeeze(input_images)#[:,1])
    Y_mask = np.squeeze(target_masks)#[:,[1,2]]
    
    #shuffle same way  
    X_img, Y_mask = shuffle_data(X_img, Y_mask)
    
    X_img = np.stack(X_img, axis=0)
    X_img = X_img/255
    
    Y_mask = np.stack(Y_mask, axis=0)
    Y_mask = Y_mask/255
    
    return X_img, Y_mask


def shuffle_data(X_img, Y_mask):
    rng_state = np.random.get_state()
    np.random.shuffle(X_img)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_mask)
    return X_img, Y_mask







def divide_train_test(X_img, Y_mask, trainsize, validationsize=0, rgb = False, 
                      augment_flip = False, augment_noise = 0):
    
    print("Dividing data into train and test set.")
    # divide into train and test:
    split1 = int(len(X_img)*trainsize)
    split2 = int(len(X_img)*trainsize) + int(len(X_img)*validationsize)
    X_train = X_img[0:split1]
    X_val = X_img[split1+1:split2]
    X_test = X_img[split2+1:-1]
    Y_train = Y_mask[0:split1]
    Y_val = Y_mask[split1+1:split2]
    Y_test = Y_mask[split2+1:-1]
    
    # format input arrays
    if rgb == False:
        X_train = X_train[:,:,:,np.newaxis]
        X_val = X_val[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
        Y_train = Y_train[:,:,:,np.newaxis]
        Y_val = Y_val[:,:,:,np.newaxis]
        Y_test = Y_test[:,:,:,np.newaxis]
    
    if augment_flip:
        print("Adding flipped images to trainset.")
        X_train, Y_train = data_augment_flip(X_train, Y_train)
        
    if augment_noise != 0:
        print("Adding images with noise to trainset.")
        X_train, Y_train = data_augment_noise(X_train, Y_train, augment_noise)
    
    if augment_flip or augment_noise:
        X_train, Y_train = shuffle_data(X_train, Y_train)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def data_augment_flip(X_img, Y_mask):  #obsobs transpose ????

    for index in range(X_img.shape[0]):
        #flip image
        img = X_img[index]
        ver_img = np.flip(img, 0)
        hor_img = np.flip(img, 1)
        vh_img = np.flip(hor_img, 0)
        ver_img = ver_img[np.newaxis]
        hor_img = hor_img[np.newaxis]
        vh_img = vh_img[np.newaxis]

        #flip target pixel
        mask = Y_mask[index]
        ver_mask = np.flip(mask, 0)
        hor_mask = np.flip(mask, 1)
        vh_mask = np.flip(hor_mask, 0)
        ver_mask = ver_mask[np.newaxis]
        hor_mask = hor_mask[np.newaxis]
        vh_mask = vh_mask[np.newaxis]
        
        #add to arrays
        X_img = np.append(X_img, ver_img, axis = 0)
        X_img = np.append(X_img, hor_img, axis = 0)
        X_img = np.append(X_img, vh_img, axis = 0)
        
        Y_mask = np.append(Y_mask, ver_mask, axis = 0)                    
        Y_mask = np.append(Y_mask, hor_mask, axis = 0)                    
        Y_mask = np.append(Y_mask, vh_mask, axis = 0) 
        
    #return new array with added data
    return X_img, Y_mask
        

def data_augment_noise(X_img, Y_mask, adding_noise): 
    
    nr,row,col,ch = X_img.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    
    inputs = X_img
    targets = Y_mask
    
    print("Making Gauss1")
    gauss1 = np.random.normal(mean,sigma,(nr,row,col,ch))
    gauss1 = gauss1.reshape(nr,row,col,ch)
    noisy1 = inputs + gauss1
    
    print("Adding Gauss1")
    X_img = np.append(X_img, gauss1, axis = 0)
    Y_mask = np.append(Y_mask, targets, axis = 0)
    
    if adding_noise > 1:
        print("Making Gauss2")
        gauss2 = np.random.normal(mean,sigma,(nr,row,col,ch))
        gauss2 = gauss2.reshape(nr,row,col,ch)
        noisy2 = inputs + gauss2
        print("Adding Gauss2")
        X_img = np.append(X_img, gauss2, axis = 0)
        Y_mask = np.append(Y_mask, targets, axis = 0)
     
    return X_img, Y_mask   
    