from PIL import Image, ImageOps
import numpy as np
from os import listdir


def construct_data(rgb = False, dim = (192, 240), coord = True):
    folder_path = 'priors_all_copy'
    input_images = []
    target_pixels = []
    passed_img = 0

    for filename in sorted(listdir(folder_path)):
        if "DS_Store" in filename:
            pass
        elif "mask" not in filename:
            try:
                #open image
                img_data = Image.open(folder_path + '/' + filename)
                if rgb == True:
                    img_data = img_data.convert('RGB')
                else: 
                    img_data = img_data
                #open mask
                maskname = filename.split('.')[0] + "_mask.png"
                pix_data = Image.open(folder_path + '/' + maskname)
                img_resized = img_data.resize(dim)
                data = np.array(img_resized)
                input_images.append([filename,data])

                data = np.array(pix_data)
                pixel_coord = np.where(data == True)
                img_size = pix_data.size
                if coord == True:
                    hori_pix = (pixel_coord[1][0]/img_size[0])*dim[0]
                    vert_pix = (pixel_coord[0][0]/img_size[1])*dim[1]
                else:
                    hori_pix = (pixel_coord[1][0]/img_size[0])
                    vert_pix = (pixel_coord[0][0]/img_size[1])                    
                target_pixels.append([maskname, hori_pix, vert_pix])
                  
                    
            except (UnidentifiedImageError, NameError):
                passed_img += 1

    input_images = np.asarray(input_images)
    target_pixels = np.asarray(target_pixels)
        
    #remove filenames (only keep input and target)
    X_img = np.squeeze(input_images[:,1])
    Y_pix = target_pixels[:,[1,2]]
    
    #shuffle same way  
    X_img, Y_pix = shuffle_data(X_img, Y_pix)
    
    X_img = np.stack(X_img, axis=0)
    X_img = X_img/255
    
    return X_img, Y_pix


def shuffle_data(X_img, Y_pix):
    rng_state = np.random.get_state()
    np.random.shuffle(X_img)
    np.random.set_state(rng_state)
    np.random.shuffle(Y_pix)
    return X_img, Y_pix


def divide_train_test(X_img, Y_pix, trainsize, validationsize=0, rgb = False, 
                      augment_flip = False, augment_noise = 0, coord = True):
    
    print("Dividing data into train and test set.")
    # divide into train and test:
    split1 = int(len(X_img)*trainsize)
    split2 = int(len(X_img)*trainsize) + int(len(X_img)*validationsize)
    X_train = X_img[0:split1]
    X_val = X_img[split1+1:split2]
    X_test = X_img[split2+1:-1]
    Y_train = Y_pix[0:split1]
    Y_val = Y_pix[split1+1:split2]
    Y_test = Y_pix[split2+1:-1]
    
    # format input arrays
    if rgb == False:
        X_train = X_train[:,:,:,np.newaxis]
        X_val = X_val[:,:,:,np.newaxis]
        X_test = X_test[:,:,:,np.newaxis]
    
    if augment_flip:
        print("Adding flipped images to trainset.")
        X_train, Y_train = data_augment_flip(X_train, Y_train, coord)
        
    if augment_noise != 0:
        print("Adding images with noise to trainset.")
        X_train, Y_train = data_augment_noise(X_train, Y_train, augment_noise)
    
    if augment_flip or augment_noise:
        X_train, Y_train = shuffle_data(X_train, Y_train)
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


def data_augment_flip(X_img, Y_pix, coord):  
    #print(coord)
    vert = float(X_img.shape[1])
    hort = float(X_img.shape[2])
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
        pix = Y_pix[index].astype(float)
        #print(pix)
        #input()
        
        if coord:
            ver_pix = np.array([pix[0], vert - pix[1]])[np.newaxis]
            hor_pix = np.array([hort - pix[0], pix[1]])[np.newaxis]
            vh_pix = np.array([hort - pix[0], vert - pix[1]])[np.newaxis]
        else:
            ver_pix = np.array([pix[0], 1 - pix[1]])[np.newaxis]
            hor_pix = np.array([1 - pix[0], pix[1]])[np.newaxis]
            vh_pix = np.array([1 - pix[0], 1 - pix[1]])[np.newaxis]
        
        #add to arrays
        X_img = np.append(X_img, ver_img, axis = 0)
        X_img = np.append(X_img, hor_img, axis = 0)
        X_img = np.append(X_img, vh_img, axis = 0)
        
        Y_pix = np.append(Y_pix, ver_pix, axis = 0)                    
        Y_pix = np.append(Y_pix, hor_pix, axis = 0)                    
        Y_pix = np.append(Y_pix, vh_pix, axis = 0) 
        
        #X_img, Y_pix = shuffle_data(X_img, Y_pix)

    #return new array with added data
    return X_img, Y_pix
        

def data_augment_noise(X_img, Y_pix, adding_noise): 
    
    nr,row,col,ch = X_img.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    
    inputs = X_img
    targets = Y_pix
    
    print("Making Gauss1")
    gauss1 = np.random.normal(mean,sigma,(nr,row,col,ch))
    gauss1 = gauss1.reshape(nr,row,col,ch)
    noisy1 = inputs + gauss1
    
    print("Adding Gauss1")
    X_img = np.append(X_img, gauss1, axis = 0)
    Y_pix = np.append(Y_pix, targets, axis = 0)
    
    if adding_noise > 1:
        print("Making Gauss2")
        gauss2 = np.random.normal(mean,sigma,(nr,row,col,ch))
        gauss2 = gauss2.reshape(nr,row,col,ch)
        noisy2 = inputs + gauss2
        print("Adding Gauss2")
        X_img = np.append(X_img, gauss2, axis = 0)
        Y_pix = np.append(Y_pix, targets, axis = 0)
     
    return X_img, Y_pix    
    


def get_filename(dim, rgb, coord, flip, noise, settype, setsize):
    
    name = str(dim[0]) + "x" + str(dim[1]) 
    
    if rgb:
        name += "_3channel"
    else:
        name += "_1channel"
        
    if coord:
        name += "_coord"
    else:
        name += "_normc"
    if flip:
        name += "_flip"
    if noise != 0:
        name += "_noise" + str(noise)
    
    name += "_" + settype + "_" + str(setsize)
    
    return name
        

def save_data(dim, X_train, Y_train, X_val, Y_val, X_test, Y_test,
             trainsize, validationsize, rgb, augment_flip, 
             augment_noise, coord):
    
    testsize = 1 - validationsize - trainsize
    
    X_train_name = get_filename(dim, rgb, coord, flip=augment_flip, 
                                noise=augment_noise, settype='Xtrain', setsize=trainsize)
    
    Y_train_name = get_filename(dim, rgb, coord, flip=augment_flip, 
                                noise=augment_noise, settype='Ytrain', setsize=trainsize)
    
    X_val_name = get_filename(dim, rgb, coord, flip=augment_flip, 
                                noise=augment_noise, settype='Xval', setsize=validationsize)
    
    Y_val_name = get_filename(dim, rgb, coord, flip=augment_flip, 
                                noise=augment_noise, settype='Yval', setsize=validationsize)
    
    X_test_name = get_filename(dim, rgb, coord, flip=augment_flip, 
                                noise=augment_noise, settype='Xtest', setsize=testsize)
    
    Y_test_name = get_filename(dim, rgb, coord, flip=augment_flip, 
                                noise=augment_noise, settype='Ytest', setsize=testsize)
    
    np.save(X_train_name, X_train)
    np.save(Y_train_name, Y_train)
    np.save(X_val_name, X_val)
    np.save(Y_val_name, Y_val)
    np.save(X_test_name, X_test)
    np.save(Y_test_name, Y_test)
    
    return 

def load_data(dim, rgb, coord, flip, noise, trainsize, validationsize):
    testsize = 1-validationsize-trainsize
    try:
        X_train = np.load(get_filename(dim, rgb, coord, flip, noise, settype='Xtrain', setsize=trainsize))
        Y_train = np.load(get_filename(dim, rgb, coord, flip, noise, settype='Ytrain', setsize=trainsize))
        X_val = np.load(get_filename(dim, rgb, coord, flip, noise, settype='Xval', setsize=validationsize))
        Y_val = np.load(get_filename(dim, rgb, coord, flip, noise, settype='Yval', setsize=validationsize))
        X_test = np.load(get_filename(dim, rgb, coord, flip, noise, settype='Xtest', setsize=testsize))
        Y_test = np.load(get_filename(dim, rgb, coord, flip, noise, settype='Ytest', setsize=testsize))
        print("Data set exists. Import complete.")
    except (IOError, FileNotFoundError):
        print("Data set does not exist. Constructing data set.") 
        print("Importing data...")
        X_img, Y_pix = construct_data(rgb = rgb, dim = dim, coord = coord)
        print("Import complete.")
        print("Dividing and augmenting data...")
        X_train, Y_train, X_val, Y_val, X_test, Y_test = divide_train_test(X_img, Y_pix, 
                                                         trainsize = trainsize, validationsize = validationsize, 
                                                         rgb = rgb, augment_flip = flip, 
                                                         augment_noise = noise, coord = coord)
        save_data(dim, X_train, Y_train, X_val, Y_val, X_test, Y_test, 
                  trainsize, validationsize, rgb, augment_flip=flip, augment_noise=noise, coord=coord)
        print("Saving new data set.")
        print("Data set construction complete")
    return X_train, Y_train, X_val, Y_val, X_test, Y_test
        
        
