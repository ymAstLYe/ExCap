In the implementation, ANI injects noise into image features when loading data features during training, as detailed below. These methods, defined within a Dataset subclass, can be found in lines 110–138 of Train_inference.py.

def add_noise_gauss(self, img_feature, img_feature_fix):
    noisy = torch.randn_like(img_feature)
    noisy_level1 = args.noisy_level1
    delta = noisy * noisy_level1
    output_feat = img_feature + delta
    return output_feat, noisy

def add_noise_neighbor(self, neighbor, img_feature, img_feature_fix, noise_gauss):
    noisy_level2 = args.noisy_level2
    ngb_noise = torch.zeros_like(img_feature, device=img_feature.device, dtype=img_feature.dtype)
    if neighbor:
        norm_of_noisy = noise_gauss.norm(dim=-1, keepdim=True).clamp_min(EPS)
        noise_gauss_norm = noise_gauss / norm_of_noisy  # [D]
        neighbor_features = self.support_features_img[neighbor]  # [k, D]
        differences = neighbor_features - img_feature_fix  # [k, D]
        norm_of_differences = differences.norm(dim=-1, keepdim=True).clamp_min(EPS)  # [k, 1]
        difference_norms = differences / norm_of_differences  # [k, D]
        similarities = noise_gauss_norm @ difference_norms.T  # [k,]
        mask = similarities > 0  # [k,]

        ngb_noise = (noise_gauss.unsqueeze(0) * similarities.unsqueeze(-1) * noisy_level2 / len(neighbor))[mask].sum(dim=0)
        # [1, D] * [k, 1] * [k, 1] = [k, D]
        out_feature = img_feature + ngb_noise
    else:
        out_feature = img_feature

    return out_feature, ngb_noise