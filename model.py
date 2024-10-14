import os
import numpy as np
from PIL import Image, ImageOps
import cv2
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR
import UNet as net

class Model():
    def __init__(self):
        self.num_epochs = 7
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = net.UNet().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size = 5, gamma = 0.01)

    @staticmethod
    def getOriginalImages(folder_path):
        image_files = [f for f in os.listdir(folder_path)]
        original_images = []

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            with Image.open(image_path).convert("L") as img:
                original_images.append(img)
        return original_images

    @staticmethod
    def getMaskImages(folder_path):
        mask_files = [f for f in os.listdir(folder_path)]
        mask_images = []

        for mask_file in mask_files:
            mask_path = os.path.join(folder_path, mask_file)
            with Image.open(mask_path).convert("L") as img:
                mask_images.append(img)
        return mask_images

    def getOriginalData(self, folder_path):
        original_data = []
        image_files = self.getOriginalImages(folder_path)

        for image_file in image_files:
            original_data.append(np.array(image_file))
        return original_data

    def getMaskData(self, folder_path):
        mask_data = []
        mask_files = self.getMaskImages(folder_path)
        for mask_file in mask_files:
            mask_data.append(np.array(mask_file))
        return mask_data

    def transformImages(self, original_images, mask_images, rotation_angle, scale_factor, flip_h, flip_v):
        transformed_originals = []
        transformed_masks = []

        for original_img, mask_img in zip(original_images, mask_images):
            rotated_original = original_img.rotate(rotation_angle)
            rotated_mask = mask_img.rotate(rotation_angle)
            new_size = (int(rotated_original.size[0] * scale_factor), int(rotated_original.size[1] * scale_factor))
            scaled_original = rotated_original.resize(new_size, Image.Resampling.BICUBIC)
            scaled_mask = rotated_mask.resize(new_size, Image.Resampling.BICUBIC)
            if flip_h:
                scaled_original = ImageOps.mirror(scaled_original)
                scaled_mask = ImageOps.mirror(scaled_mask)
            if flip_v:
                scaled_original = ImageOps.flip(scaled_original)
                scaled_mask = ImageOps.flip(scaled_mask)
            final_original = ImageOps.fit(scaled_original, original_img.size, method = Image.Resampling.BICUBIC)
            final_mask = ImageOps.fit(scaled_mask, mask_img.size, method = Image.Resampling.BICUBIC)
            final_original_np = np.array(final_original)
            transformed_originals.append(final_original_np)
            transformed_masks.append(np.array(final_mask))
        return transformed_originals, transformed_masks

    def getTrainingData(self, original_folder_path, mask_folder_path):
        original_data = self.getOriginalData(original_folder_path)
        mask_data = self.getMaskData(mask_folder_path)
        training_data = []
        labels = []
        resized_original_data = [cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in original_data]
        resized_mask_data = [cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in mask_data]
        training_data.extend(resized_original_data)
        labels.extend(resized_mask_data)
        rotations = [15, 30, 45, 60]
        scales = [0.9, 1.0, 1.1]
        flips = [(False, False), (True, False), (False, True), (True, True)]

        for i in range(9):
            rotation_angle = rotations[i % len(rotations)]
            scale_factor = scales[i % len(scales)]
            flip_h, flip_v = flips[i % len(flips)]
            transformed_originals, transformed_masks = self.transformImages(self.getOriginalImages(original_folder_path),
                                                                            self.getMaskImages(mask_folder_path),
                                                                            rotation_angle,
                                                                            scale_factor,
                                                                            flip_h,
                                                                            flip_v)
            transformed_originals = [cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in transformed_originals]
            transformed_masks = [cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in transformed_masks]
            training_data.extend(transformed_originals)
            labels.extend(transformed_masks)
        training_data = np.array(training_data)
        labels = np.array(labels) / 255.0
        train_data, test_data, train_labels, test_labels = train_test_split(training_data, labels, test_size = 0.15, random_state = 42)
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size = 0.1765,random_state = 42)
        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    def train(self):
        print("Training model...")
        train_data1, train_labels1, val_data1, val_labels1, test_data1, test_labels1 = self.getTrainingData("Iris 100/OriginalData", "Iris 100/SegmentationClass")
        train_data2, train_labels2, val_data2, val_labels2, test_data2, test_labels2 = self.getTrainingData( "Iris 200/OriginalData", "Iris 200/SegmentationClass")
        train_data3, train_labels3, val_data3, val_labels3, test_data3, test_labels3 = self.getTrainingData( "Iris 300/OriginalData", "Iris 300/SegmentationClass")
        train_data = np.concatenate((train_data1, train_data2, train_data3), axis=0)
        train_labels = np.concatenate((train_labels1, train_labels2, train_labels3), axis=0)
        val_data = np.concatenate((val_data1, val_data2, val_data3), axis=0)
        val_labels = np.concatenate((val_labels1, val_labels2, val_labels3), axis=0)
        test_data = np.concatenate((test_data1, test_data2, test_data3), axis=0)
        test_labels = np.concatenate((test_labels1, test_labels2, test_labels3), axis=0)
        train_data_tensor = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1).to(self.device)
        train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        val_data_tensor = torch.tensor(val_data, dtype=torch.float32).unsqueeze(1).to(self.device)
        val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32).unsqueeze(1).to(self.device)
        train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
        val_dataset = TensorDataset(val_data_tensor, val_labels_tensor)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        train_losses = []
        val_losses = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                print(i, len(train_loader))
                if i % 100 == 0:
                    visualizeTrainingStep(inputs, outputs, targets, num_samples = 5)
            epoch_train_loss = running_loss / len(train_loader)
            train_losses.append(epoch_train_loss)
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
            epoch_val_loss = val_loss / len(val_loader)
            val_losses.append(epoch_val_loss)

            print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
            self.scheduler.step()
        plt.figure(figsize = (10, 5))
        plt.plot(range(1, self.num_epochs + 1), train_losses, label = "Training Loss")
        plt.plot(range(1, self.num_epochs + 1), val_losses, label = "Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Progress")
        plt.legend()
        plt.grid(True)
        plt.show()
        model_save_path = "unet_model.pth"
        torch.save(self.model.state_dict(), model_save_path)
        self.test(test_data, test_labels)

    def test(self, test_data, test_labels):
        test_data_tensor = torch.tensor(test_data, dtype = torch.float32).unsqueeze(1).to(self.device)
        test_labels_tensor = torch.tensor(test_labels, dtype = torch.float32).unsqueeze(1).to(self.device)
        test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
        test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
        print(f"Test Loss: {test_loss / len(test_loader):.4f}")

    def getModel(self):
        self.model.load_state_dict(torch.load("unet_model_best.pth"))
        self.model.to(self.device)
        self.model.eval()

def visualizeResults(model, test_data, test_labels, num_samples = 5):
    model.model.eval()
    indices = np.random.choice(len(test_data), num_samples, replace = False)
    test_data_sample = torch.tensor(test_data[indices], dtype = torch.float32).unsqueeze(1).to(model.device)
    test_labels_sample = torch.tensor(test_labels[indices], dtype = torch.float32).unsqueeze(1).to(model.device)
    with torch.no_grad():
        predictions = model.model(test_data_sample)
        predictions = predictions.cpu().numpy()
        test_labels_sample = test_labels_sample.cpu().numpy()
    plt.figure(figsize = (15, num_samples * 3))

    for i in range(num_samples):
        plt.subplot(num_samples, 3, 3 * i + 1)
        plt.imshow(test_data[indices[i]], cmap = "gray")
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(num_samples, 3, 3 * i + 2)
        plt.imshow(test_labels_sample[i].squeeze(), cmap = "gray")
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(num_samples, 3, 3 * i + 3)
        plt.imshow(predictions[i].squeeze(), cmap = "gray")
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def checkModel():
    model = Model()
    model.getModel()
    test_data1, test_labels1, _, _, _, _ = model.getTrainingData("Iris 100/OriginalData", "Iris 100/SegmentationClass")
    test_labels1 = np.array([cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in test_labels1]) / 255.0
    test_data2, test_labels2, _, _, _, _ = model.getTrainingData("Iris 200/OriginalData", "Iris 200/SegmentationClass")
    test_labels2 = np.array([cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in test_labels2]) / 255.0
    test_data3, test_labels3, _, _, _, _ = model.getTrainingData("Iris 300/OriginalData", "Iris 300/SegmentationClass")
    test_labels3 = np.array([cv2.resize(img, (256, 256), interpolation = cv2.INTER_CUBIC) for img in test_labels3]) / 255.0
    test_data = np.concatenate((test_data1, test_data2, test_data3), axis = 0)
    test_labels = np.concatenate((test_labels1, test_labels2, test_labels3), axis = 0)
    visualizeResults(model, test_data, test_labels, num_samples = 5)

def checkImage(image_path):
    model = Model()
    model.getModel()
    input_image = Image.open(image_path).convert("L")
    input_image = np.array(input_image)
    input_image_resized = cv2.resize(input_image, (256, 256), interpolation = cv2.INTER_CUBIC)
    input_image_tensor = torch.tensor(input_image_resized, dtype = torch.float32).unsqueeze(0).unsqueeze(0).to(model.device)
    model.model.eval()
    with torch.no_grad():
        predicted_mask = model.model(input_image_tensor)
        predicted_mask = predicted_mask.squeeze().cpu().numpy()
    plt.figure(figsize = (10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image_resized, cmap = "gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap = "gray")
    plt.title("Predicted Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def visualizeTrainingStep(inputs, predictions, targets, num_samples = 5):
    inputs = inputs.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().numpy()
    plt.figure(figsize = (15, num_samples * 4))

    for i in range(min(num_samples, len(predictions))):
        plt.subplot(num_samples, 4, 4 * i + 1)
        plt.imshow(inputs[i].squeeze(), cmap = "gray")
        plt.title("Original Image")
        plt.axis("off")
        plt.subplot(num_samples, 4, 4 * i + 2)
        plt.imshow(targets[i].squeeze(), cmap = "gray")
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(num_samples, 4, 4 * i + 3)
        plt.imshow(predictions[i].squeeze(), cmap = "gray")
        plt.title("Prediction")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# model = Model()
# model.train()
# checkModel()

# checkImage("C:/Users/adria/Desktop/Left Eye Prediction.jpg")
