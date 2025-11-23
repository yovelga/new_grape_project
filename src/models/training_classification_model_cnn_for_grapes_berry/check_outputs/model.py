import torch
from torchvision.models import efficientnet_b0
from config import MODEL_SAVE_PATH


def get_model(num_classes=2):
    """
    טוען את המודל EfficientNet-B0, מגדיר את השכבה הסופית ל-2 קלאסים,
    ומטען את המשקולות שאימנו בפרוייקט הקודם.
    """
    # אין שימוש במשקולות ברירת מחדל – נטען את האדריכלות בלבד
    model = efficientnet_b0(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)

    # טעינת המשקולות המאומנות
    state_dict = torch.load(MODEL_SAVE_PATH, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)

    model.eval()
    return model
