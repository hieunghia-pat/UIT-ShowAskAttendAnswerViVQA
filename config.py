# paths
qa_path = '../ViVQA'  # directory containing the question and annotation jsons
train_path = '../ViVQA/train'  # directory of training images
# val_path = '../ViVQA/val'  # directory of validation images
val_path = ""
test_path = '../ViVQA/test'  # directory of test images
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
json_train_path = "../ViVQA/vivqa_train_2017.json"
json_test_path = "../ViVQA/vivqa_test_2017.json"

task = 'OpenEnded'
dataset = 'vivqa'

# preprocess config
preprocess_batch_size = 64
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
epochs = 50
batch_size = 128
initial_lr = 1e-3  # default Adam lr
lr_halflife = 50000  # in iterations
data_workers = 8