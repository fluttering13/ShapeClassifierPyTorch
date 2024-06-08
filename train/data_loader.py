from torch.utils.data import DataLoader,random_split
from torchvision.transforms import transforms
from torchvision import transforms, datasets

def exp_data_loader(image_height, image_width, batch_size):
    data_transform = transforms.Compose([
        transforms.Resize((image_height, image_width)),  # resize the image
        transforms.ToTensor(),         # to tensor datatype
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # normalization
    ])

    dataset = datasets.ImageFolder(root='./pic', transform=data_transform)

    dataset_size = len(dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create datasets
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create DataLoaders for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# if __name__ == "__main__":
#     image_height, image_width= 200,200
#     batch_size=5
#     train_loader, val_loader, test_loader=exp_data_loader(image_height, image_width, batch_size)