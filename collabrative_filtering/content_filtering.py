# Step 1: Build user profile --> user interaction, item_feature and user_id

import numpy as np
def build_user_profile(user_interaction, item_features, user_id):
    user_item_row = user_interaction[user_id]

    user_profile = np.zeros(item_features.shape[1])
    for idx, val in enumerate(user_item_row):
        user_profile += user_item_row[idx] * item_features[idx]

    tot_rating = np.sum(user_item_row)
    if tot_rating == 0:
        return user_profile
    else:
        return user_profile/tot_rating


def cos_sim(u, v):
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    if u_norm==0 or v_norm==0:
        return 0

    dot = np.dot(u,v)
    return dot/(u_norm*v_norm)

def user_item_sim(user_interaction, item_features):
    sim_matrix = np.zeros((len(user_interaction), len(item_features)))

    for user_idx in range(len(user_interaction)):
        user_profile = build_user_profile(user_interaction, item_features, user_idx)
        for item_idx in range(len(item_features)):
            sim_matrix[user_idx][item_idx] = cos_sim(user_profile, item_features[item_idx])

    sim_matrix[user_interaction>0] = -np.inf
    return sim_matrix


if __name__ == "__main__":
    item_features = np.array([
    [1, 1, 0, 0],  # I1
    [1, 0, 0, 1],  # I2
    [0, 1, 1, 0],  # I3
    [0, 1, 0, 1],  # I4
    ], dtype=float)

    user_interaction = np.array([
                [5, 4, 4, 0],
                [4, 5, 3, 0],
                [0, 0, 5, 4],
                [0, 0, 4, 5]
            ], dtype=float)


    print(build_user_profile(user_interaction, item_features, 1))
    print(user_item_sim(user_interaction, item_features))