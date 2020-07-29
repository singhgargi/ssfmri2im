"""
Description:
General image functions
"""
import numpy as np
from scipy.misc import imresize, imsave
from scipy.ndimage import shift
from sklearn.preprocessing import normalize
import os
import scipy
import pickle

def image_prepare(img,size,interpolation = 'cubic'):
    """
    Select central crop, resize and convert gray to 3 channel image
    :param img: image
    :param size: image output size
    :param interpolation: interpolation used in resize
    :return resized and croped image
    """
    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]

    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]

    img = imresize(img, size=[size, size], interp=interpolation)
    if (np.ndim(img) == 3):
        out_img = img
    else:
        k = img/255.0
#         nx,ny = k.shape
#         k = k.reshape(1,nx*ny)
#         k = normalize(k)
#         x = k.shape
#         k = k.reshape(nx,ny)
#         k = (k-np.min(k))/(np.max(k)-np.min(k))
        img = k        
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img

    return out_img



def rand_shift(img,max_shift = 0 ):
    """
    randomly shifted image
    :param img: image
    :param max_shift: image output size
    :return randomly shifted image
    """
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted



def image_collage(img_arrays, rows =10, border =5,save_file = None):
    """
    create image collage for arrays of images
    :param img_arrays: list of image arrays
    :param rows: number of rows in resulting iamge collage
    :param border: border between images
    :param save_file: location for resulting image
    :return image collage
    """

    img_len =img_arrays[0].shape[2]
    array_len  = img_arrays[0].shape[0]
    num_arrays =len(img_arrays)

    cols = int(np.ceil(array_len/rows))
    img_collage = np.ones([rows * (img_len + border) + border,num_arrays *cols * (img_len + border) , 3])
    for ind in range(array_len):
        x = (ind % cols) * num_arrays
        y = int(ind / cols)

        img_collage[border * (y + 1) + y * img_len:border * (y + 1) + (y + 1) * img_len, cols * (x + 1) + x * img_len:cols * (x + 1) +(x + num_arrays) * img_len]\
            = np.concatenate([img_arrays[i][ind] for i in range(num_arrays) ],axis=1)        
                      
    if(save_file is not None):
        imsave(save_file,img_collage)

    return img_collage



def save_images(images,images_orig = None ,folder=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(images.shape[0]):
        if(images_orig is None):
            scipy.misc.imsave(folder+'img_'+str(i)+'.jpg',images[i])
        else:
            img_concat = np.concatenate([images_orig[i],images[i]],axis=1)
            img_concat = np.squeeze(img_concat)
            scipy.misc.imsave(folder + 'img_' + str(i) + '.jpg', img_concat)
            
def calc_accuracy(original_imgs,reconstructed_img,gt_idx):
	"""
	Inputs:
		original_imgs : numpy array of shape N,112,112,3 where N is the number of images. 
						one of these images is the ground truth image.
		gt_idx : index of the ground truth image. can take values from 0 to N-1.
		reconstructed_img : numpy array of shape 112,112,3.
	Output:
		returns true if the ground truth image has the highest correlation with the reconstructed image  
	"""
	max_idx = 0
	max_corr = -50
	N = original_imgs.shape[0]
	for i in range(N):
		corr = scipy.stats.pearsonr(original_imgs[i].reshape(-1),reconstructed_img.reshape(-1))
		corr = corr[0]
		print("correlation ",corr)
		if corr > max_corr :
			max_corr = corr
			max_idx = i
	if(max_idx==gt_idx):
		print("n",N)
		print("max correlation",max_corr)
		return True
	else:
		print("n",N)
		print("false")
		return False
            
def save_results(images,images_orig = None ,folder=''):
	if not os.path.exists(folder):
		os.makedirs(folder)
	Res = {2:[],5:[],10:[]}
	for it in range(10):
		res = {2:[],5:[],10:[]}
		for i in range(images.shape[0]):
			for j in [2,5,10]:
				gt_idx = j-1
				new = np.delete(np.arange(images.shape[0]),i)
				index = np.random.choice(new, j-1, replace=False)
				index = np.append(index,i)
				print("indices array",index)
				print("image index",i)
				res[j].append(calc_accuracy(images_orig[index],images[i],gt_idx))
				
		Res[2].append(res[2].count(True)/len(res[2]))	
		Res[5].append(res[5].count(True)/len(res[5]))
		Res[10].append(res[10].count(True)/len(res[10]))
	Res2 = sum(Res[2])/len(Res[2])
	Res5 = sum(Res[5])/len(Res[5])
	Res10 = sum(Res[10])/len(Res[10])
	print("2-way accuracy",Res2)
	print("5-way accuracy",Res5)
	print("10-way accuracy",Res10)
	with open(folder+'results.p', 'wb') as fp:
		pickle.dump(Res, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
        
