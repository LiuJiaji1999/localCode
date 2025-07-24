import timm
import torch
model = timm.create_model("resnet50", pretrained=True)
torch.save(model.state_dict(), "resnet50.pth")
