from PIL import Image
from torchvision import transforms
import load_data
from model import resnet34
from utils import AverageMeter, ProgressMeter, accuracy, save_ckpt, load_ckpt, train, validate

load_saved_data = False

if load_saved_data:
  outfile = 'input_batch_step1_560.npy'
  input_batch = np.load(outfile)
  outfile = 'output_batch_step1_560.npy'
  output_batch = np.load(outfile)
  outfile = 'input_batch_val_step1_560.npy'
  input_batch_val = np.load(outfile)
  outfile = 'output_batch_val_step1_560.npy'
  output_batch_val = np.load(outfile)
else:
  df_trains, df_tests, df_vals, df_val_targets = load_data.load_data()
  
  input_tensor = preprocess(df_trains)
  input_batch = np.transpose(input_tensor, (1,2,0)).float()#input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
  output_tensor = preprocess(df_tests)
  output_batch = np.transpose(output_tensor, (1,2,0)).float()
  output_batch = output_batch[:,0,:]
  output_batch = torch.unsqueeze(output_batch, 1)

  input_tensor_val = preprocess(df_vals)
  input_batch_val = np.transpose(input_tensor_val, (1,2,0)).float()
  output_tensor_val = preprocess(df_val_targets)
  output_batch_val = np.transpose(output_tensor_val, (1,2,0)).float()
  output_batch_val = output_batch_val[:,0,:]
  output_batch_val = torch.unsqueeze(output_batch_val, 1)
  print(output_batch_val.shape)

  preprocess = transforms.Compose([
      #transforms.Resize(256),
      #transforms.CenterCrop(224),
      transforms.ToTensor(),
      #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])
  
  outfile = 'input_batch_step1_560'
  np.save(outfile, input_batch)
  outfile = 'output_batch_step1_560'
  np.save(outfile, output_batch)
  outfile = 'input_batch_val_step1_560'
  np.save(outfile, input_batch_val)
  outfile = 'output_batch_val_step1_560'
  np.save(outfile, output_batch_val)
  
 
preprocess = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_batch = preprocess(input_batch)
input_batch = np.transpose(input_batch, (1,2,0)).float()
print(input_batch.shape)

output_batch = preprocess(output_batch)
output_batch = np.transpose(output_batch, (1,2,0)).float()

input_batch_val = preprocess(input_batch_val)
input_batch_val = np.transpose(input_batch_val, (1,2,0)).float()

output_batch_val = preprocess(output_batch_val)
output_batch_val = np.transpose(output_batch_val, (1,2,0)).float()

model = resnet34() #mpl()

##### optimizer / learning rate scheduler / criterion #####
optimizer = torch.optim.SGD(model.parameters(), lr=LR,
                            momentum=MOMENTUM, weight_decay=WEIGHTDECAY,
                            nesterov=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lr_decay_steps,
                                                 gamma=0.1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
criterion = torch.nn.BCELoss()
###########################################################

model = model
criterion = criterion
start_epoch = 0

# resume
if RESUME:
  model, optimizer, start_epoch = load_ckp(model, optimizer)


print('==> Load data..')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

transform_noise=transforms.Compose([
    #transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,)),
    AddGaussianNoise(0., 0.004)
])

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        a = np.random.rand(1)

        if self.transform:
            if a > 0.5: 
                x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

#train_dataset = torchvision.datasets.ImageFolder(
#    './train', transform=train_transform)
train_dataset = CustomTensorDataset(tensors=(input_batch, output_batch), transform=transform_noise)
val_dataset = torch.utils.data.TensorDataset(input_batch_val, output_batch_val)


#concat trainset and valset
#concat_dataset = ConcatDataset([train_dataset, val_dataset])

#split trainset and valset randomly (9:1)


new_train_loader = DataLoader(train_dataset, 
                          batch_size=BATCHSIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
new_val_loader = DataLoader(val_dataset,
                        batch_size=BATCHSIZE, shuffle=True,
                        num_workers=2, pin_memory=True)

train_loader = new_train_loader
val_loader = new_val_loader
"""

train_loader = DataLoader(train_dataset,
                          batch_size=BATCHSIZE, shuffle=True,
                          num_workers=4, pin_memory=True)

#val_dataset = torchvision.datasets.ImageFolder('./valid', transform=valid_transform)
val_loader = DataLoader(val_dataset,
                        batch_size=BATCHSIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

"""

last_top1_acc = 0
acc1_valid = 0
best_acc1 = 0
best_train_acc1 = 0
is_train_best = False
is_best = False

train_accs = []
train_pres = []
test_accs = []
test_pres = []
for epoch in range(start_epoch, EPOCHS):
    print("\n----- epoch: {}, lr: {} -----".format(
        epoch, optimizer.param_groups[0]["lr"]))

    # train for one epoch
    start_time = time.time()
    last_top1_acc, last_top1_pre = train(train_loader, epoch, model, optimizer, criterion)
    train_accs.append(last_top1_acc)
    train_pres.append(last_top1_pre)
    elapsed_time = time.time() - start_time
    print('==> {:.2f} seconds to train this epoch\n'.format(
        elapsed_time))

    # validate for one epoch
    start_time = time.time()
    acc1_valid, pre1_valid = validate(val_loader, model, criterion)
    test_accs.append(acc1_valid)
    test_pres.append(pre1_valid)
    elapsed_time = time.time() - start_time
    print('==> {:.2f} seconds to validate this epoch\n'.format(
        elapsed_time))


    # learning rate scheduling
    scheduler.step()

    #is_best = acc1_valid > best_acc1
    #best_acc1 = max(acc1_valid, best_acc1)

    is_train_best = last_top1_acc > best_train_acc1
    best_train_acc1 = max(last_top1_acc, best_train_acc1)

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_ckpt(checkpoint, is_train_best)

    #if is_best:
    #  torch.save(model.state_dict(), SAVEPATH+'model_weight_best.pth')

    if is_train_best:
      torch.save(model.state_dict(), SAVEPATH+'model_weight_best.pth')

    # Save model each epoch
    #torch.save(model.state_dict(), SAVEPATH+'model_weight_epoch{}.pth'.format(epoch))

print(f"Last Top-1 Accuracy: {acc1_valid}")
print(f"Last Top-1 Train Accuracy: {last_top1_acc}")
print(f"Number of parameters: {pytorch_total_params}")

print("***Threshold 0.5***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.5)
print("***Threshold 0.6***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.6)
print("***Threshold 0.7***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.7)
print("***Threshold 0.8***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.8)
print("***Threshold 0.9***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.9)
print("***Threshold 0.95***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.95)
print("***Threshold 0.99***")
acc1_valid = validate(val_loader, model, criterion, threshold=0.99)

print(train_accs)
print(test_accs)
print(train_pres)
print(test_pres)

#mean_result = np.mean(np.array(results[j]),axis=0)

plt.figure(figsize = (15, 5))
plt.plot(train_accs, label = 'train', c='black')
plt.plot(test_accs, label = 'validation', c = 'red')
plt.plot(train_pres, label = 'train precision', c='pink')
plt.plot(test_pres, label = 'val precision', c = 'blue')
#plt.plot(df_ratios[j].iloc[-test_size:,0].values, label = 'true trend', c = 'black')
#plt.plot(mean_result, label = 'forcast average', c = 'pink')
plt.legend()
plt.title('training/validation accuracy generation')
plt.show()
