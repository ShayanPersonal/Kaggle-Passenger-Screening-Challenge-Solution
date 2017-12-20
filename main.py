import torch
import torchvision
import torchsample.transforms
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F
from util import read_data, get_x_views
from skimage import io
from mvcnn import mvcnn
from util import show_memusage, plot_image
from torch.optim import lr_scheduler
from sgdr import CosineLR, WaveLR

TEST_MODEL = False          # Creates predictions file
DEBUG = False               # Loads small dataset and plots augmented images for debugging
VIEW_COUNT_TOTAL = 16       # Total number of views in our scans. APS files have 16.
VIEW_COUNT_SAMPLE = 16      # Total number of views sampled from the scan. I now use all 16 and this line isn't needed.
epochs = 401
state_dict = None           # Loads a previous state of the model for picking back up training or making predictions.
opt_dict = None             # Loads a previous state of the optimizer for picking back up training if it was cut short.

body_zones_flipped = dict([(1,3), (2,4), (3,1), (4,2), (5,5), (6,7), (7,6), (8,10), (9,9), (10,8), (11,12), (12,11), (13,14), (14,13), (15,16), (16,15), (17,17)])

class TransformDataset(torch.utils.data.Dataset):
    # Same as a normal Dataset but randomly augments the data. 
    # Augmentations are done asynchronously on CPU while the previous batch goes through GPU.
    def __init__(self, data_tensor, target_tensor, names, train):
        assert data_tensor.size(0) == target_tensor.size(0) and type(names[0]) is str
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.names = names
        self.train = train

    def __getitem__(self, index):
        np.random.seed()
        name = self.names[index]
        data = self.data_tensor[index]
        target = self.target_tensor[index]
        if self.train:
            data, target = transform_sample(data, target)
        return data, target

    def __len__(self):
        return self.data_tensor.size(0)

class AverageMeter(object):
    # Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def create_one_hot(x):
    one_hot = np.zeros(17)
    one_hot[x-1] = 1
    return one_hot

def invert_target(target):
    inverted_target = torch.Tensor(target.shape)
    for k, v in body_zones_flipped.items():
        inverted_target[k-1] = target[v-1]
    return inverted_target

def sample_views(input):
    # Not used anymore.
    batch_dim = [dim for dim in input.size()]
    batch_dim[1] = VIEW_COUNT_SAMPLE
    input_sampled = torch.zeros(*batch_dim)
    for row, sample in enumerate(input):
        input_sampled[row] = sample[np.random.randint(VIEW_COUNT_TOTAL // VIEW_COUNT_SAMPLE)::VIEW_COUNT_TOTAL // VIEW_COUNT_SAMPLE].contiguous()
    return input_sampled

def transform_sample(im, target=None):
    if target is None:
        invert = False
    else:
        # Invert target if we're inverting the image
        invert = np.random.randint(2)
        if invert:
            target = invert_target(target)

    im = im.numpy()

    if invert:
        im = np.flip(im, 3).copy()
        im = np.flip(im, 0).copy()
        im = np.roll(im, 1, 0)

    rand_int = np.random.randint(VIEW_COUNT_SAMPLE)

    im = np.roll(im, rand_int, 0)
    im = torch.from_numpy(im)

    im = im.view(-1, im.size(2), im.size(3))
    
    rand_affine = torchsample.transforms.RandomAffine(
            translation_range=[-0.01, 0.01], rotation_range=15, zoom_range=[0.95, 1.05], interp='nearest'
    )

    im = rand_affine(im)

    im = im.view(16, 1, im.size(1), im.size(2))


    if target is None:
        return im
    else:
        return im, target


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)

        input = input.repeat(1, 1, 3, 1, 1)

        if DEBUG and False:
            print(list(zip(target[0].cpu().tolist(), [x+1 for x in range(17)])))
            print(list(zip(target[1].cpu().tolist(), [x+1 for x in range(17)])))
            for j in range(VIEW_COUNT_SAMPLE):
                plt.imshow(input.cpu().numpy()[0, j, 0])
                plt.show()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

        del input, target, output, loss

    scheduler.step()
    loss_tracker_train.append(losses.avg)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(async=True)
        target = target.cuda(async=True)

        input = input.repeat(1, 1, 3, 1, 1)

        if DEBUG and False:
            print(list(zip(target[0].cpu().tolist(), [x+1 for x in range(17)])))
            print(list(zip(target[1].cpu().tolist(), [x+1 for x in range(17)])))
            for j in range(VIEW_COUNT_SAMPLE):
                plt.imshow(input.cpu().numpy()[0, j, 0])
                plt.show()

        input = torch.autograd.Variable(input)
        target = torch.autograd.Variable(target)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.data[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('VALIDATION:')
        print('Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_time=batch_time,
                data_time=data_time, loss=losses))
        
        del input, target, output, loss

    loss_tracker_val.append(losses.avg)

def predict(model, name):
    input = name_to_array(name, "test")

    # Imitate a batch
    input = np.expand_dims(input, 0)
    input = torch.Tensor(input)

    if DEBUG and False:
        for j in range(VIEW_COUNT_SAMPLE):
            plt.imshow(input.numpy()[0, j, 0, :, :])
            plt.show()

    input = input.repeat(1,1,3,1,1)
    input_var = torch.autograd.Variable(input.cuda(), requires_grad=False)

    accum = None

    # Run through network
    if type(model) != list:
        model = [model]
    for m in model:
        output = torch.nn.Sigmoid()(m(input_var))
        output = output.data.cpu().numpy()[0]
        if accum is None:
            accum = output
        else:
            accum += output

    return accum / len(model)

def test_model(model, base_dir=None, epoch=0):
    time_str = str(int(time.time()))[2:]
    if base_dir == None:
        base_dir = "predictions/{}".format(time_str)
        os.mkdir(base_dir)
    outfile = open('{}/predictions_{}_{}.csv'.format(base_dir, time_str, epoch), 'w')
    print('Id,Probability', file=outfile)
    test_names = set([filename.split('.')[0] for filename in os.listdir('test/')])
    for name in test_names:
        print(name)
        for bodypart, prob in enumerate(predict(model, name)):
            print("{}_Zone{},{}".format(name, bodypart + 1, prob), file=outfile)

def name_to_array(name, directory):
    # Convert aps file to numpy array.
    ext = directory
    if directory == "test":
        ext = "aps"
    array = np.array(get_x_views("{}/{}.{}".format(directory, name, ext), x=VIEW_COUNT_TOTAL))
    array = np.expand_dims(array, 1)
    array = np.pad(array, ((0,0), (0,0), (0, 1), (0, 0)), mode="constant", constant_values=0)
    
    return array







if TEST_MODEL:
    print("Testing model only")
    models = []
    for name in state_dict:
        model = mvcnn(17, pretrained=True).cuda()
        model.load_state_dict(torch.load(name))
        model.eval()
        models.append(model)
        print("Added {}".format(name))
    test_model(models)
    exit()

print("Initializing model")
model = mvcnn(17, pretrained=True).cuda()

# Create dictionary matching name to vector
if DEBUG:
    train_file = open('stage1_labels_debug.csv')
else:
    train_file = open('stage1_labels.csv')
train_file.readline()
name_to_vector = {}
for line in train_file:
    name_zone, label = line.strip().split(',')
    name, zone = name_zone.split('_')
    zone_int = int(zone[4:])
    one_hot = create_one_hot(zone_int)
    if name not in name_to_vector:
        name_to_vector[name] = np.zeros(17)
    if int(label) == 1:
        name_to_vector[name] += one_hot

print("Loaded training file, now loading in images")

# Convert to raw training input and output
# Images are (660, 512)
sample_count = len(name_to_vector)
print(sample_count)
show_memusage()

names = [None] * sample_count
training_input = np.empty((sample_count, VIEW_COUNT_TOTAL, 1, 660  + 1, 512  + 0), dtype=np.float32)
training_output = np.empty((sample_count, 17))
for i, (name, one_hot) in enumerate(name_to_vector.items()):
    input_tensor = name_to_array(name, "aps")
    training_input[i] = input_tensor
    training_output[i] = one_hot
    names[i] = name
    if i % 100 == 0:
        print(i)

print("Splitting into train/validation sets")
training_split = int(len(training_input)) - 5
training_input, valid_input = training_input[0:training_split], training_input[training_split:]
training_output, valid_output = training_output[0:training_split], training_output[training_split:]

print(training_input.shape)
print(training_output.shape)
print(valid_input.shape)
print(valid_output.shape)

print("Loaded in images, creating DataLoaders")
training_input = torch.Tensor(training_input)
training_output = torch.Tensor(training_output)
valid_input = torch.Tensor(valid_input)
valid_output = torch.Tensor(valid_output)

dataset = TransformDataset(training_input, training_output, names, train=True)
valid_dataset = TransformDataset(valid_input, valid_output, names, train=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, sampler=None, batch_sampler=None, num_workers=8, pin_memory=True, drop_last=True)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, pin_memory=True, drop_last=False)

criterion = torch.nn.BCEWithLogitsLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)
scheduler = CosineLR(optimizer, step_size_min=1e-4, t0=200, tmult=1)

if state_dict:
    model.load_state_dict(torch.load(state_dict))
    if opt_dict:
        optimizer.load_state_dict(torch.load(opt_dict))
    print("Loaded old weights")

torch.backends.cudnn.benchmark = False

print("Beginning training...")
time_str = str(int(time.time()))[2::]
base_dir = "predictions/{}".format(time_str)
loss_tracker_train = []
loss_tracker_val = []
best_loss = 0.010
this_loss = 1.0

for epoch in range(epochs):
    train(data_loader, model, criterion, optimizer, scheduler, epoch)
    validate(valid_data_loader, model, criterion)

    if epoch and epoch % 25 == 0:
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        torch.save(model.state_dict(), "{}/model_{}.torch".format(base_dir, epoch))
        torch.save(optimizer.state_dict(), "{}/opt_{}.torch".format(base_dir, epoch))

        # Save a plot of the average loss over time
        plt.clf()
        plt.plot(loss_tracker_train[1:], label="Training loss")
        plt.plot(loss_tracker_val[1:], label="Validation loss")
        plt.legend(loc="upper left")
        plt.savefig("{}/predictions_{}.png".format(base_dir, epoch))

        print("Predicting...")
        test_model(model, base_dir, epoch)

    this_loss = loss_tracker_val[-1]
    print("This loss: {}".format(this_loss))
    print("Best loss: {}".format(best_loss))

    if this_loss < best_loss + 0.0025:
        print("Found better model with {} loss (old loss was {})".format(this_loss, best_loss))
        best_loss = min(this_loss, best_loss)
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
        torch.save(model.state_dict(), "{}/best_model_{}_{:.4f}.torch".format(base_dir, epoch, this_loss))