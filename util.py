import os
import datetime
import matplotlib.pyplot as plt
import torch
import colour


class Logger:
    def __init__(self, base_folder="results", learning_rate=None):
        os.makedirs(base_folder, exist_ok=True)
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_folder = os.path.join(base_folder, current_time)
        os.makedirs(self.log_folder, exist_ok=True)
        self.log_file = os.path.join(self.log_folder, "training_log.txt")
        with open(self.log_file, 'w') as f:
            if learning_rate is not None:
                f.write(f"Learning Rate: {learning_rate}\n")

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")
        print(message)

    def log_image_lab(self, images, target1, output1, epoch, iteration, imagename):
        fig, axes = plt.subplots(1, 3, figsize=(16, 8))

        image = images[0].detach().cpu().permute(1, 2, 0)
        target = target1[0].detach().cpu().permute(1, 2, 0)
        output = output1[0].detach().cpu().permute(1, 2, 0)

        target[:, :, 0] = target[:, :, 0] * 100
        target[:, :, 1] = target[:, :, 1] * 200 - 100
        target[:, :, 2] = target[:, :, 2] * 200 - 100
        target = colour.Lab_to_XYZ(target)
        target = colour.XYZ_to_sRGB(target)

        output[:, :, 0] = output[:, :, 0] * 100
        output[:, :, 1] = output[:, :, 1] * 200 - 100
        output[:, :, 2] = output[:, :, 2] * 200 - 100
        output = colour.Lab_to_XYZ(output)
        output = colour.XYZ_to_sRGB(output)

        titles = ["Image", "Diffuse Reflectance", "Output (Diffuse Reflectance)"]
        imgs = [image, target, output]

        for ax, img, title in zip(axes.flatten(), imgs, titles):
            ax.imshow(img.clip(0, 1))
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        image_path = os.path.join(self.log_folder, f"epoch_{epoch}_iteration_{iteration}_{imagename[0]}.png")
        plt.savefig(image_path)
        plt.close()

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        os.makedirs(os.path.join(self.log_folder, 'checkpoints'), exist_ok=True)
        checkpoint_path = os.path.join(self.log_folder, 'checkpoints', f'model_{epoch}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)
