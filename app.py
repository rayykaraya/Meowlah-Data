import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv

# ======== Model GCN ========
class GCNModel(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=64, output_dim=2):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.embedding_output = None

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        self.embedding_output = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ======== Dummy class untuk GCN inference ========
class DummyData:
    def __init__(self, x):
        self.x = x
        self.edge_index = torch.tensor([[0], [0]])

# ======== Model DQN ========
class DQN(nn.Module):
    def __init__(self, input_dim=64, output_dim=2):  # input 64 = dim dari GCN embedding
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ======== Load kedua model ========
model_gcn = GCNModel()
model_gcn.load_state_dict(torch.load("model_gcn.pt", map_location="cpu"))
model_gcn.eval()

model_dqn = DQN()
model_dqn.load_state_dict(torch.load("model_dqn.pt", map_location="cpu"))
model_dqn.eval()

# ======== Fungsi prediksi ========
def predict(input_str, model_type):
    try:
        # konversi input jadi tensor
        arr = np.array([float(x) for x in input_str.strip().split(',')], dtype=np.float32)
        x_tensor = torch.tensor(arr).unsqueeze(0)

        if model_type == "GCN":
            data = DummyData(x_tensor)
            with torch.no_grad():
                out = model_gcn(data)
                pred = torch.argmax(out, dim=1).item()
                return "Normal" if pred == 0 else "Attack"

        elif model_type == "DQN":
            with torch.no_grad():
                out = model_dqn(x_tensor)
                pred = torch.argmax(out, dim=1).item()
                return "Normal" if pred == 0 else "Attack"

    except Exception as e:
        return f"Error input: {e}"

# ======== Gradio interface ========
gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(label="Masukkan 45 fitur (pisahkan koma)"),
        gr.Radio(["GCN", "DQN"], label="Pilih Model")
    ],
    outputs="text",
    title="Deteksi Intrusi Jaringan (GCN + DQN)",
    description="Klasifikasi trafik jaringan: serangan atau normal menggunakan GCN dan DQN"
).launch()
