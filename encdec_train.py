
import os
import config_file
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = config_file.GPU_ID
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from keras.callbacks import TensorBoard, LearningRateScheduler

from KamitaniData.kamitani_data_handler import kamitani_data_handler as data_handler
from Utils.misc import calc_snr, log_image_collage_callback
from Utils.image_functions import *

from Utils.image_loss import *
from Utils.batch_generator import *
from Models.models import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


#################################################### data load #########################################################
with open('../gdrive/My Drive/dataPCA95.p', 'rb') as fp:
    data = pickle.load(fp)
Y = data['responses'][:1200,:]
Y_test_avg = data['responses_test'][:50,:]
Y_test = data['responses_test'][:50,:]
ysam = Y_test
for i in range(34):
    Y_test = np.concatenate((Y_test,ysam),axis=0)
labels = np.arange(ysam.shape[0])
labels = np.tile(labels,35)
NUM_VOXELS = Y.shape[1]
X = data['stimuli'][:1200,:]
X_test = data['stimuli_test'][:50,:]
X_test_avg = X_test
xsam = X_test
for i in range(34):
    X_test = np.concatenate((X_test,xsam),axis=0)
    
# handler = data_handler(matlab_file = config_file.kamitani_data_mat)
# Y,Y_test,Y_test_avg = handler.get_data(roi = 'ROI_VC',imag_data = 0)
# labels_train, labels = handler.get_labels(imag_data = 0)

print("For Y")
print(np.shape(Y))
print("For Y_test")
print(np.shape(Y_test))
print("For Y_test_avg")
print(np.shape(Y_test_avg))
# print("labels_train")
# print(np.shape(labels_train))
print("labels")
print(np.shape(labels))
print(labels[:10])
print(np.min(labels))
print(np.max(labels))


# file= np.load(config_file.images_npz) #_56
# X = file['train_images']
# X_test_avg = file['test_images']

print("X")
print(np.shape(X))
print("X_test_avg")
print(np.shape(X_test_avg))

# X= X[labels_train]
# X_test = X_test_avg[labels]

print("X")
print(np.shape(X))
print("X_test")
print(np.shape(X_test))

NUM_VOXELS = Y.shape[1]
#################################################### losses ##########################################################

snr  = calc_snr(Y_test,Y_test_avg,labels)
snr = snr/snr.mean()
# snr = np.ones((Y.shape[1],), dtype=int)

print("snr")
print(np.shape(snr))
SNR  = tf.constant(snr,shape = [1,len(snr)],dtype = tf.float32)
print("SNR")
print(type(SNR))

def mse_vox(y_true, y_pred):
    print("mse_vox")
    return K.mean(SNR*K.square(y_true-y_pred),axis=-1)

def mae_vox(y_true, y_pred):
    print("mae_vox")
    return K.mean(SNR*K.abs(y_true-y_pred),axis=-1)

def combined_voxel_loss(y_true, y_pred):
    print("combined_voxel_loss")
    return mae_vox(y_true, y_pred) +  0.1 *cosine_proximity(y_true, y_pred)

def maelog_vox(y_true, y_pred):
    print("maelog_vox")
    return K.mean(SNR*K.log(K.abs(y_true-y_pred)+1),axis=-1)


Tv_reg =1
image_loss_ = image_loss()

def feature_loss(y_true, y_pred ):
    print("feature_loss")
    return 0.15*image_loss_.vgg_loss(y_true, y_pred,'block2_conv2')+0.7*image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')+0.15*image_loss_.pixel_loss(y_true, y_pred)
    #return image_loss_.pixel_loss(y_true, y_pred)#image_loss_.vgg_loss(y_true, y_pred,'block1_conv2')

def combined_loss(y_true, y_pred):
    print("combined_loss")
    return feature_loss(y_true, y_pred)+  Tv_reg *total_variation_loss(y_pred)

#################################################### learning param & schedule #########################################
print("beginning learning param & schedule")

initial_lrate = 0.001
epochs_drop =  30.0
RESOLUTION = config_file.image_size
epochs = int(epochs_drop*5)
examples = 16
include_decenc= 1
frac = 3

def step_decay(epoch):

   drop = 0.2
   lrate = initial_lrate * math.pow(drop,
           math.floor((1+epoch)/epochs_drop))
   return lrate
print("ending")
##################################################### model ############################################################
print("beginning model")
image_loss_.calc_norm_factors(X)
print(type(X))
print(X.shape)
print("image_loss")
encoder_weights = config_file.encoder_weights
print("encoded weights")
dec_param = decoder_param(NUM_VOXELS)
enc_param = encoder_param(NUM_VOXELS)
enc_param.drop_out = 0.25
encoder_model = encoder(enc_param)
decoder_model = decoder(dec_param)

encoder_model.trainable = False
print(encoder_weights)
encoder_model.load_weights(encoder_weights )

model = encdec(NUM_VOXELS,RESOLUTION,encoder_model,decoder_model)
model.compile(loss= {'out_rec_img':combined_loss,'out_pred_voxel':combined_voxel_loss},loss_weights=[1.0,1.0],optimizer=Adam(lr=5e-4,amsgrad=True),metrics={'out_rec_img':['mse','mae']})
##################################################### callbacks ########################################################
callback_list = []

if(config_file.decoder_tenosrboard_logs is not None):
    callback = TensorBoard(config_file.decoder_tenosrboard_logs)
    callback.set_model(model)
    callback_list.append(callback)


reduce_lr = LearningRateScheduler(step_decay)
callback_list.append(reduce_lr)
if not os.path.exists(config_file.results):
    os.makedirs(config_file.results)

callback_list.append( log_image_collage_callback(Y_test_avg, X_test_avg, decoder_model, dir = config_file.results+'/test_collge_ep/'))
callback_list.append( log_image_collage_callback(Y[0:50], X[0:50], decoder_model, dir = config_file.results+'/train_collge_ep/'))
##################################################### generators #######################################################

loader_train = batch_generator_encdec(X, Y, Y_test, labels, batch_paired = 48, batch_unpaired = 16)
loader_test = batch_generator_encdec(X_test_avg, Y_test_avg, Y_test, labels, batch_paired = 50, batch_unpaired = 0)
##################################################### fit & save #######################################################
print("reached generator")
model.fit_generator(loader_train, epochs=epochs, verbose=2,callbacks=callback_list,workers=5,use_multiprocessing=True) #epochs
image_collage([X_test_avg,decoder_model.predict(Y_test_avg)], rows =10, border =5,save_file = config_file.results+'/collage.jpeg')
save_images(decoder_model.predict(Y_test_avg),images_orig = X_test_avg ,folder=config_file.results+'/test/')
save_images(decoder_model.predict(Y[0:50]),images_orig = X[0:50] ,folder=config_file.results+'/train/')
save_results(decoder_model.predict(Y_test_avg),images_orig = X_test_avg ,folder=config_file.results+'/accuracyplots/')

if(config_file.decoder_weights is not None):
    decoder_model.save_weights(config_file.decoder_weights)
