'''
Trains a discriminator between the LearnedEnv generated trajectory (from the behavior policy) and a trajectory generated
from the original environment (from the behavior policy)
Update: distinguish between start states and returns of the trajectories.
'''
import torch
import pickle
import numpy as np

class TrajectoryClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=150):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    def get_trajectory_weight(self, trajectory):
        with torch.no_grad():
            p_hat = self.forward(trajectory.flatten())
            return p_hat/(1-p_hat + 1e-8)

if __name__ == 'main':
    state_dim=17
    return_dim=1
    pi_b = 9
    pi_e = 10

    model = TrajectoryClassifier(state_dim + return_dim)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()

    # Trajectories from Target Policy and Learned Env = 1
    first_term_trajs = pickle.load(open('./saved_trajectories/' + str(pi_e) + "_first_term.pkl", 'rb'))

    trajectories_target = first_term_trajs['trajectories']
    returns_target = first_term_trajs['returns']

    # Each sample is [17-dim first state] + [1-dim return] flattened
    trajectory_inputs = np.asarray(list(trajectories_target))[:150, 0, :]
    return_inputs = np.asarray(returns_target)[:150]
    inputs_pi_e = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    labels_pi_e = torch.Tensor(np.ones(inputs_pi_e.shape[0]))

    # Trajectories from Original Env and Behavior Policy = 0
    behavior_trajectories_o = pickle.load(open('./saved_trajectories/' + str(pi_b) + "_o.pkl", 'rb')) # This is always going to be similar?
    trajectories_behavior = behavior_trajectories_o['trajectories']
    returns_behavior = behavior_trajectories_o['returns']
    trajectory_inputs = np.asarray(list(trajectories_behavior))[:150, 0, :]
    return_inputs = np.asarray(returns_behavior)[:150]
    inputs_pi_b = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    labels_pi_b = torch.Tensor(np.zeros(inputs_pi_b.shape[0]))

    # Validation set
    trajectory_inputs = np.asarray(list(trajectories_target))[150:, 0, :]
    return_inputs = np.asarray(returns_target)[150:]
    val_pi_e = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    val_label_pi_e = torch.Tensor(np.ones(val_pi_e.shape[0]))

    trajectory_inputs = np.asarray(list(trajectories_behavior))[150:, 0, :]
    return_inputs = np.asarray(returns_behavior)[150:]
    val_pi_b = torch.Tensor(np.hstack((trajectory_inputs, return_inputs.reshape(-1, 1))))
    val_label_pi_b = torch.Tensor(np.zeros(val_pi_b.shape[0]))


    # Training Loop
    for epoch in range(300):
        preds = model(torch.cat([inputs_pi_e, inputs_pi_b])).squeeze(-1)
        loss = criterion(preds, torch.cat([labels_pi_e, labels_pi_b]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        val_inputs = torch.cat([val_pi_e, val_pi_b])
        val_labels = torch.cat([val_label_pi_e, val_label_pi_b])

        # Model predictions
        val_preds = model(val_inputs).squeeze(-1)  # (B,) instead of (B, 1)

        # Compute loss
        val_loss = criterion(val_preds, val_labels)

        # Optional: Compute accuracy (assumes sigmoid + 0.5 threshold)
        predicted_labels = (torch.sigmoid(val_preds) > 0.5).float()
        accuracy = (predicted_labels == val_labels).float().mean()

    print(f"Validation Loss: {val_loss.item():.4f}, Accuracy: {accuracy.item():.4f}") # This is not bad as a model.




