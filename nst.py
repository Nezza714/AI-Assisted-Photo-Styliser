import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import tkinter as tk
from tkinter import filedialog, Label, Button, Scale, HORIZONTAL
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 356

# Define normalization transform (VGG expects this)
vgg_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

# De-normalization transform to display/save images properly
def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(tensor.device).view(3, 1, 1)
    return tensor * std + mean

# Image loader with normalization
loader = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    vgg_normalize  # Apply normalization
])

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = loader(image).unsqueeze(0)
        return image.to(device)
    except Exception as e:
        raise ValueError(f"Error loading image: {str(e)}")

# VGG model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        return features

def run_style_transfer(content_path, style_path, steps=6000, output_path="generated.png"):
    content = load_image(content_path)
    style = load_image(style_path)

    model = VGG().to(device).eval()
    generated = content.clone().requires_grad_(True)

    learning_rate = 0.001
    alpha = 1
    beta = 0.01

    optimizer = optim.Adam([generated], lr=learning_rate)

    final_content_loss = 0
    final_style_loss = 0
    final_total_loss = 0

    for step in range(steps):
        gen_features = model(generated)
        content_features = model(content)
        style_features = model(style)

        content_loss = style_loss = 0

        for gen_feat, cont_feat, styl_feat in zip(gen_features, content_features, style_features):
            _, c, h, w = gen_feat.shape
            content_loss += torch.mean((gen_feat - cont_feat) ** 2)

            G = gen_feat.view(c, h * w).mm(gen_feat.view(c, h * w).t())
            A = styl_feat.view(c, h * w).mm(styl_feat.view(c, h * w).t())

            style_loss += torch.mean((G - A) ** 2)

        total_loss = alpha * content_loss + beta * style_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Save final values for printing
        if step == steps - 1:
            final_content_loss = content_loss.item()
            final_style_loss = style_loss.item()
            final_total_loss = total_loss.item()

    # De-normalize before saving for correct colors
    final_img = denormalize(generated.squeeze(0).detach().cpu()).clamp(0, 1)
    save_image(final_img, output_path)

    # Print loss values to console
    print("\n--- Stylization Complete ---")
    print(f"Final Content Loss: {final_content_loss:.4f}")
    print(f"Final Style Loss:   {final_style_loss:.4f}")
    print(f"Final Total Loss:   {final_total_loss:.4f}")

    return output_path

# GUI Setup
class StyleTransferApp:
    def __init__(self, master):
        self.master = master
        master.title("Neural Style Transfer")

        self.content_path = None
        self.style_path = None
        self.result_path = "generated.png"
        self.steps = tk.IntVar(value=600)

        self.label = Label(master, text="Neural Style Transfer Tool", font=("Helvetica", 16))
        self.label.pack(pady=10)

        self.upload_content_btn = Button(master, text="Upload Content Image", command=self.upload_content)
        self.upload_content_btn.pack(pady=5)

        self.upload_style_btn = Button(master, text="Upload Style Image", command=self.upload_style)
        self.upload_style_btn.pack(pady=5)

        self.slider_label = Label(master, text="Select Number of Iterations")
        self.slider_label.pack(pady=5)

        self.iter_slider = Scale(master, from_=200, to=6000, resolution=200, orient=HORIZONTAL, variable=self.steps)
        self.iter_slider.pack(pady=5)

        self.generate_btn = Button(master, text="Generate Stylised Image", command=self.generate)
        self.generate_btn.pack(pady=10)

        self.result_label = Label(master, text="")
        self.result_label.pack(pady=10)

        self.preview_label = Label(master)
        self.preview_label.pack(pady=5)

        self.download_btn = Button(master, text="Download Result", command=self.download_result, state=tk.DISABLED)
        self.download_btn.pack(pady=5)

    def upload_content(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
            if not path:
                return
            Image.open(path).convert("RGB")
            self.content_path = path
            self.result_label.config(text="Content image loaded.")
        except Exception as e:
            self.result_label.config(text=f"Invalid content image: {str(e)}")
            self.content_path = None

    def upload_style(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
            if not path:
                return
            Image.open(path).convert("RGB")
            self.style_path = path
            self.result_label.config(text="Style image loaded.")
        except Exception as e:
            self.result_label.config(text=f"Invalid style image: {str(e)}")
            self.style_path = None

    def generate(self):
        if not self.content_path or not self.style_path:
            self.result_label.config(text="Please upload both images.")
            return

        steps = self.steps.get()
        self.result_label.config(text=f"Processing for {steps} iterations... This may take a few minutes.")
        self.master.update()

        output_path = run_style_transfer(self.content_path, self.style_path, steps, self.result_path)
        self.result_label.config(text=f"Stylised image generated after {steps} iterations.")
        self.download_btn.config(state=tk.NORMAL)

        # Show image preview in GUI
        img = Image.open(output_path)
        img = img.resize((256, 256))  # Resize for display
        img_tk = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=img_tk)
        self.preview_label.image = img_tk

    def download_result(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png")])
        if save_path:
            Image.open(self.result_path).save(save_path)
            self.result_label.config(text="Image saved successfully.")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferApp(root)
    root.mainloop()