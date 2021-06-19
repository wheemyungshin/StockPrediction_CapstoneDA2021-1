from PIL import Image
from torchvision import transforms
import load_data

load_saved_data = False

if load_saved_data:
  outfile = '/content/drive/MyDrive/stock_predict_models/input_batch_step1_560.npy'
  input_batch = np.load(outfile)
  outfile = '/content/drive/MyDrive/stock_predict_models/output_batch_step1_560.npy'
  output_batch = np.load(outfile)
  outfile = '/content/drive/MyDrive/stock_predict_models/input_batch_val_step1_560.npy'
  input_batch_val = np.load(outfile)
  outfile = '/content/drive/MyDrive/stock_predict_models/output_batch_val_step1_560.npy'
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
  
  outfile = '/content/drive/MyDrive/stock_predict_models/input_batch_step1_560'
  np.save(outfile, input_batch)
  outfile = '/content/drive/MyDrive/stock_predict_models/output_batch_step1_560'
  np.save(outfile, output_batch)
  outfile = '/content/drive/MyDrive/stock_predict_models/input_batch_val_step1_560'
  np.save(outfile, input_batch_val)
  outfile = '/content/drive/MyDrive/stock_predict_models/output_batch_val_step1_560'
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


