import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List

def extract_glcm_features(patch: np.ndarray) -> List[float]:
    if patch is None or patch.size == 0: return [0.0] * 6
    if len(patch.shape) > 2:
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    else:
        patch_gray = patch

    if np.all(patch_gray == patch_gray[0,0]):
        return [0.0] * 6

    glcm = graycomatrix(patch_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    asm = graycoprops(glcm, 'ASM').mean()
    return [contrast, dissimilarity, homogeneity, energy, correlation, asm]

def extract_hist_features(patch: np.ndarray, bins=16) -> List[float]:
    if patch is None or patch.size == 0: return [0.0] * bins
    if len(patch.shape) == 2 or patch.shape[2] == 1:
        gray_patch = patch
    else:
        gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    hist = cv2.calcHist([gray_patch], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten().tolist()

def count_cells_feature_based(image: np.ndarray, params: Dict[str, Any]) -> int:
    if image is None or image.size == 0: return 0

    patch_size = params.get("patch_size", 32)
    feature_type = params.get("feature_type", "glcm")
    n_clusters = params.get("n_clusters", 2)
    cell_cluster_heuristic = params.get("cell_cluster_heuristic", "brightest")


    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    features_list = []
    patch_coords = []

    for y in range(0, gray_image.shape[0] - patch_size + 1, patch_size):
        for x in range(0, gray_image.shape[1] - patch_size + 1, patch_size):
            patch = gray_image[y:y+patch_size, x:x+patch_size]
            patch_color = image[y:y+patch_size, x:x+patch_size]

            if feature_type == "glcm":
                features = extract_glcm_features(patch)
            elif feature_type == "hist":
                features = extract_hist_features(patch_color if params.get("hist_use_color", False) else patch)
            else:
                return 0

            features_list.append(features)
            patch_coords.append({'y': y, 'x': x, 'patch': patch_color})

    if not features_list:
        return 0

    features_array = np.array(features_list)

    scaler = StandardScaler()
    try:
        features_scaled = scaler.fit_transform(features_array)
    except ValueError:
        return 0


    if n_clusters <= 0 or n_clusters > len(features_scaled):
        return 0

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    try:
        labels = kmeans.fit_predict(features_scaled)
    except ValueError:
        return 0

    cluster_properties = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        patch_mean_intensity = np.mean(patch_coords[i]['patch'])
        cluster_properties[label].append(patch_mean_intensity)

    mean_intensity_per_cluster = [np.mean(props) if props else 0 for props in cluster_properties]

    cell_label_idx = -1
    if cell_cluster_heuristic == "brightest":
        if mean_intensity_per_cluster:
            cell_label_idx = np.argmax(mean_intensity_per_cluster)
    elif cell_cluster_heuristic == "largest":
        if labels.size > 0:
            counts = np.bincount(labels)
            if n_clusters > 1 and len(counts) == n_clusters:
                sorted_intensity_indices = np.argsort(mean_intensity_per_cluster)
                darkest_label = sorted_intensity_indices[0]

                potential_cell_labels = [l for l in range(n_clusters) if l != darkest_label]
                if potential_cell_labels:
                    cell_label_idx = potential_cell_labels[np.argmax(counts[potential_cell_labels])]
                else: # получается только один кластер остался или он самый тёмный
                    cell_label_idx = np.argmax(counts)
            else: # только один или counts не подходят
                cell_label_idx = np.argmax(counts) if counts.size > 0 else 0


    if cell_label_idx == -1 and n_clusters == 1:
        cell_label_idx = 0
    elif cell_label_idx == -1: # heuristic провалился
        return 0

    segmentation_map = np.zeros((gray_image.shape[0], gray_image.shape[1]), dtype=np.uint8)
    for i, label in enumerate(labels):
        if label == cell_label_idx:
            y, x = patch_coords[i]['y'], patch_coords[i]['x']
            segmentation_map[y:y+patch_size, x:x+patch_size] = 255

    # post-process
    kernel_size = params.get("seg_map_morph_kernel", 3)
    open_iter = params.get("seg_map_open_iter", 1)
    close_iter = params.get("seg_map_close_iter", 1)

    if kernel_size > 0 and (open_iter > 0 or close_iter > 0):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if open_iter > 0:
            segmentation_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if close_iter > 0:
            segmentation_map = cv2.morphologyEx(segmentation_map, cv2.MORPH_CLOSE, kernel, iterations=close_iter)

    num_components, _, stats, _ = cv2.connectedComponentsWithStats(segmentation_map, connectivity=8)

    # фильтр по area
    min_area = params.get("min_cell_area_pixels", patch_size*patch_size // 4)
    max_area = params.get("max_cell_area_pixels", patch_size*patch_size * 10)

    actual_cells = 0
    for i in range(1, num_components):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            actual_cells +=1

    return actual_cells
