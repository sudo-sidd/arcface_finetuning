import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from face_recognition.arcface.model import iresnet100
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model
model = iresnet100(pretrained=False).to(device)
model.load_state_dict(torch.load("face_recognition/arcface/weights/glink360k_cosface_r100_fp16_0.1.pth"), strict=False)
model.eval()

# Remove classification layer to extract embeddings
class FaceEmbeddingModel(torch.nn.Module):
    def __init__(self, model):
        super(FaceEmbeddingModel, self).__init__()
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer

    def forward(self, x):
        x = self.backbone(x)
        return x / torch.norm(x, dim=1, keepdim=True)  # Normalize embeddings

model = FaceEmbeddingModel(model)

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def get_embedding(face_image_path):
    face_image = Image.open(face_image_path).convert('RGB')
    face_tensor = transform(face_image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(face_tensor)

    return embedding.cpu().numpy().flatten()



def recognize_faces(test_image_path, features_path="datasets/face_features/feature.npz"):
    test_embedding = get_embedding(test_image_path)

    # Load saved features (embeddings) of the known faces
    features = np.load(features_path)
    known_names = features['images_name']
    known_embeddings = features['images_emb']

    # Calculate cosine similarity
    similarities = cosine_similarity([test_embedding], known_embeddings)

    # Get the index of the most similar face
    most_similar_idx = np.argmax(similarities)
    most_similar_name = known_names[most_similar_idx]
    similarity_score = similarities[0][most_similar_idx]

    print(f"Predicted: {most_similar_name}, Similarity: {similarity_score:.4f}")
    return most_similar_name, similarity_score


if __name__ == "__main__":
    test_image_path = "datasets/test/person1.jpg"  # Path to test image
    recognized_name, score = recognize_faces(test_image_path)
