
import os
GPU_ID = "0"

#####################  PATHS  ######################################
imagenet_dir = "../"
imagenet_wind_dir = os.path.join(imagenet_dir,"image_net/")
external_images_dir =  os.path.join(imagenet_dir,"datacoco/")

project_dir = "../"
image_size = 112
images_npz = os.path.join(project_dir,"data/images_112.npz")
kamitani_data_format = True
kamitani_data_mat = os.path.join(project_dir,"data/Subject3.mat")
caffenet_models_weights = os.path.join(project_dir,"data/imagenet-caffe-ref.mat")
results  = os.path.join(project_dir,"gdrive/My Drive/resultscocosnrtrain/")


encoder_weights = os.path.join(project_dir,"encoder.hdf5")
retrain_encoder = False
decoder_weights = None

encoder_tenosrboard_logs = None
decoder_tenosrboard_logs = None
#####################  pretrained mat conv net weights (alexnet)  ######################################

DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat'
FILENAME = 'imagenet-caffe-ref.mat'
EXPECTED_BYTES = 228031200


