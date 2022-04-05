import os
from random import randint
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()
# Augmentation
testtransform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Global Definitions

model_path = 'unet_model.pth'
data_path = 'Kaggle/'


def sample(dataset, num_samples):
    samples = []
    for i in range(num_samples):
        n = randint(0, len(dataset))
        img, mask = dataset[n]
        samples.append((img.squeeze(), mask.squeeze()))
    return samples


def showDataset(dataset, num_imgs, model=None):
    imgs = sample(dataset, num_imgs)
    if model is not None:
        number = 3
    else:
        number = 2

    fig, axs = plt.subplots(num_imgs, number, figsize=(10, 5 * num_imgs))
    for i in range(num_imgs):
        # Image
        axs[i, 0].imshow(imgs[i][0].permute(1, 2, 0))
        # Original Mask
        axs[i, 1].imshow(imgs[i][1])
        # Predict from NN
        if model is not None:
            nnMask = model.predict(imgs[i][0])
            axs[i, 2].imshow(nnMask.cpu())
    plt.show()


def transformImage(image, transformation=None, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if transformation:
        transformed = transformation(image=image, mask=mask)
        img = transformed['image']
        mask = transformed['mask'].long()
    return img, mask


class LaneDetectionDataset(Dataset):
    def __init__(self, path, val=False, transforms=None):
        self.transforms = transforms
        if not val:
            self.img_path = path + 'train/'
            self.mask_path = path + 'train_label/'
        else:
            self.img_path = path + 'val/'
            self.mask_path = path + 'val_label/'
        self.img_names = [name for name in os.listdir(self.img_path)]

    def __getitem__(self, idx, imShow=False):
        img_name = self.img_names[idx]
        mask_name = img_name[:-4] + '_label' + img_name[-4:]
        rawImg = cv2.imread(self.img_path + img_name)
        rawMask = cv2.imread(self.mask_path + mask_name, cv2.IMREAD_UNCHANGED)
        if imShow:
            cv2.imshow("Image", rawImg)
            cv2.imshow("Mask", rawMask)
            cv2.waitKey(0)
        img, mask = transformImage(rawImg, self.transforms, rawMask)
        return img, mask

    def __len__(self):
        return len(self.img_names)


class CNNLineDetector:
    def __init__(self, from_scratch=True, path='torchUnet/unet_model.pth', dataPath=data_path):
        self.val_epoch = None
        self.train_epoch = None
        self.optimizer = None
        self.loss = None

        self.train_dataset = LaneDetectionDataset(dataPath, val=False, transforms=testtransform)
        self.val_dataset = LaneDetectionDataset(dataPath, val=True, transforms=testtransform)
        self.trainloader = DataLoader(self.train_dataset, batch_size=2, shuffle=True)
        self.valloader = DataLoader(self.val_dataset, batch_size=2, shuffle=True)
        self.path = path

        self.model = smp.UnetPlusPlus(encoder_name='resnet34',
                                      encoder_weights='imagenet',
                                      in_channels=3,
                                      classes=3).to(device)
        if from_scratch:
            print('Model initialised from Scratch.')
        else:
            self.model.load_state_dict(torch.load(path))
            print('Loaded saved model at: ', path)

    def setTrainingParams(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss = smp.losses.DiceLoss('multiclass')
        self.loss.__name__ = 'DiceLoss'

        self.train_epoch = smp.utils.train.TrainEpoch(
            self.model,
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=[],
            device=device,
            verbose=True
        )

        self.val_epoch = smp.utils.train.ValidEpoch(
            self.model,
            loss=self.loss,
            metrics=[],
            device=device,
            verbose=True
        )

    def train(self, num_epochs, save=True):
        for i in range(1, num_epochs+1):
            print("Running epoch {now}/{max}".format(now=i, max=num_epochs))
            logTraining = self.train_epoch.run(self.trainloader)
            logValidation = self.val_epoch.run(self.valloader)
            print("TRAINING STATUS")
            print(logTraining, logValidation)
        if save:
            torch.save(self.model.cpu().state_dict(), self.path)
            print("Model saved to " + self.path)

    def predict(self, image):
        predictedMask = self.model(image.unsqueeze(0).to(device))
        predictedMask = torch.argmax(predictedMask.squeeze(), axis=0)
        return predictedMask


if __name__ == '__main__':

    # Define datasets
    train_dataset = LaneDetectionDataset(data_path, val=False, transforms=testtransform)
    val_dataset = LaneDetectionDataset(data_path, val=True, transforms=testtransform)

    best = False
    if best:
        model = CNNLineDetector(from_scratch=not best)
    else:
        model = CNNLineDetector(from_scratch=not best)
        model.setTrainingParams()
        model.train(num_epochs=2, save=True)

    showDataset(num_imgs=4, dataset=val_dataset, model=model)


