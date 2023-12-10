import torch
import cv2
from torchvision import transforms
from PIL import Image
from  aligned_reid.model.Model import Model

def remove_fc(state_dict):
    for key in list(state_dict.keys()):
        if key.startswith('fc.'):
            del state_dict[key]
    return state_dict

def load_pretrained_weights(pretrained_path,model=Model()):
    # Load the pretrained weights
    weights = torch.load(pretrained_path)
    
    # Remove fc layer weights (adjust if necessary)
    weights = remove_fc(weights)
    
    # Load the modified weights into the model
    model.load_state_dict(weights)

    # Set the model to evaluation mode (if necessary)
    model.eval()

    return model
def extract_features(model, processed_image):
    model.eval()
    with torch.no_grad():
        # Assuming 'model' is your pre-trained model
        global_feat, local_feat = model(processed_image)

    return global_feat, local_feat

def preprocess_image(image_path, resize_h_w=(256, 128), normalize=True):
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize and normalize the image
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize_h_w),
        transforms.ToTensor(),
    ])

    if normalize:
        transform = transforms.Compose([
            transform,
            transforms.Normalize(mean=[0.486, 0.459, 0.408], std=[0.229, 0.224, 0.225]),
        ])

    processed_image = transform(image)

    # Add batch dimension
    processed_image = processed_image.unsqueeze(0)

    return processed_image

def calculate_local_distance(f, g):
    """
    Calculate the local distance between two sets of local features.
    Args:
    - f: Local features of the first image (list of torch tensors).
    - g: Local features of the second image (list of torch tensors).
    Returns:
    - Local distance between the two sets of local features.
    """
    H = len(f)
    D = torch.zeros((H, H))

    for i in range(H):
        for j in range(H):
            D[i, j] = torch.exp(torch.norm(f[i] - g[j]) - 1) / (torch.exp(torch.norm(f[i] - g[j]) + 1))

    # Dynamic programming to find the shortest path
    S = torch.zeros((H, H))
    for i in range(H):
        for j in range(H):
            if i == 0 and j == 0:
                S[i, j] = D[i, j]
            elif i == 0:
                S[i, j] = S[i, j - 1] + D[i, j]
            elif j == 0:
                S[i, j] = S[i - 1, j] + D[i, j]
            else:
                S[i, j] = torch.min(S[i - 1, j], S[i, j - 1]) + D[i, j]

    local_distance = S[-1, -1]
    return local_distance

def calculate_global_distance(f_global, g_global):
    """
    Calculate the global distance between two global features.
    Args:
    - f_global: Global feature of the first image (torch tensor).
    - g_global: Global feature of the second image (torch tensor).
    Returns:
    - Global distance between the two global features.
    """
    global_distance = torch.norm(f_global - g_global)
    return global_distance

def calculate_net_distance(f_global, f_local, g_global, g_local):
    """
    Calculate the net distance between two images using global and local features.
    Args:
    - f_global: Global feature of the first image (torch tensor).
    - f_local: Local features of the first image (list of torch tensors).
    - g_global: Global feature of the second image (torch tensor).
    - g_local: Local features of the second image (list of torch tensors).
    Returns:
    - Net distance between the two images.
    """
    local_distance = calculate_local_distance(f_local, g_local)
    global_distance = calculate_global_distance(f_global, g_global)

    net_distance = local_distance + global_distance
    return net_distance

def total_distance_between_images(model, image_path_a, image_paths_bs):
    processed_image_a = preprocess_image(image_path_a)
    global_feat_a, local_feat_a = extract_features(model, processed_image_a)

    net_distances = []

    for image_path_b in image_paths_bs:
        processed_image_b = preprocess_image(image_path_b)
        global_feat_b, local_feat_b = extract_features(model, processed_image_b)

        net_distance = calculate_net_distance(global_feat_a, local_feat_a, global_feat_b, local_feat_b)
        net_distances.append(net_distance)

    return net_distances

if __name__=="__main__":
    # Example usage:
    f_global = torch.rand(2048)  # Placeholder for global feature of image A
    f_local = [torch.rand(128) for _ in range(7)]  # Placeholder for local features of image A
    g_global = torch.rand(2048)  # Placeholder for global feature of image B
    g_local = [torch.rand(128) for _ in range(7)]  # Placeholder for local features of image B

    net_distance = calculate_net_distance(f_global, f_local, g_global, g_local)
    print("Net Distance between Image A and Image B:", net_distance.item())
