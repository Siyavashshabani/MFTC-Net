




## datasets 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_loader, train_loader = data_loaders(data_dir, num_samples = 3, device = device)



