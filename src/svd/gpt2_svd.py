import torch
import functools
from transformers import AutoModelForCausalLM, AutoTokenizer
import pysvelte
from tabulate import tabulate
from matplotlib import pyplot as plt
from src.utils.gpt2_utils import top_tokens


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def get_model_tokenizer_embedding(model_name="gpt2-medium"):
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  if device == 'cpu':
    print("WARNING: you should probably restart on a GPU runtime")

  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  emb = model.get_output_embeddings().weight.data.T.detach()
  return model, tokenizer, emb, device


def get_model_info(model):
  num_layers = model.config.n_layer
  num_heads = model.config.n_head
  hidden_dim = model.config.n_embd
  head_size = hidden_dim // num_heads
  return num_layers, num_heads, hidden_dim, head_size

def get_mlp_weights(model,num_layers, hidden_dim):
  Ks = []
  Vs = []
  for j in range(num_layers):
    K = model.get_parameter(f"transformer.h.{j}.mlp.c_fc.weight").T.detach()
    # fuse the layernorm
    ln_2_weight = model.get_parameter(f"transformer.h.{j}.ln_2.weight").detach()
    K = torch.einsum("oi,i -> oi", K, ln_2_weight)

    V = model.get_parameter(f"transformer.h.{j}.mlp.c_proj.weight")
    Ks.append(K)
    Vs.append(V)

  Ks =  torch.cat(Ks)
  Vs = torch.cat(Vs)
  K_heads = Ks.reshape(num_layers, -1, hidden_dim)
  V_heads = Vs.reshape(num_layers, -1, hidden_dim)
  return K_heads, V_heads

def get_attention_heads(model, num_layers, hidden_dim, num_heads, head_size):
  qkvs = []
  for j in range(num_layers):
    qkv = model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight").detach().T
    ln_weight_1 = model.get_parameter(f"transformer.h.{j}.ln_1.weight").detach()

    qkv = qkv - torch.mean(qkv, dim=0)
    qkv = torch.einsum("oi,i -> oi", qkv, ln_weight_1)
    qkvs.append(qkv.T)

  W_Q, W_K, W_V = torch.cat(qkvs).chunk(3, dim=-1)
  W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") for j in range(num_layers)]).detach()
  W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
  W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  return W_Q_heads, W_K_heads, W_V_heads, W_O_heads

def top_singular_vectors(mat, emb, all_tokens, k = 20, N_singular_vectors = 10, with_negative = False,use_visualization=True, filter="topk"):
  U,S,V = torch.linalg.svd(mat)
  Vs = []
  for i in range(N_singular_vectors):
      acts = V[i,:].float() @ emb
      Vs.append(acts)
  if use_visualization:
    Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
    pysvelte.TopKTable(tokens=all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
  else:
    Vs = [top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
    print(tabulate([*zip(*Vs)]))
  if with_negative:
    Vs = []
    for i in range(N_singular_vectors):
      acts = -V[i,:].float() @ emb
      Vs.append(acts)
    if use_visualization:
      Vs = torch.stack(Vs, dim=1).unsqueeze(1) # n_tokens, n_layers (1), n_directions
      pysvelte.TopKTable(tokens=all_tokens, activations=Vs, obj_type="SVD direction", k=k, filter=filter).show()
    else:
      Vs = [top_tokens(Vs[i].float().cpu(), k = k, pad_to_maxlen=True) for i in range(len(Vs))]
      print(tabulate([*zip(*Vs)]))

def plot_MLP_singular_vectors(K,layer_idx, max_rank=None):
  W_matrix = K[layer_idx, :,:]
  U,S,V = torch.linalg.svd(W_matrix,full_matrices=False)
  if not max_rank:
    max_rank = len(S)
  if max_rank > len(S):
    max_rank = len(S) -1
  plt.plot(S[0:max_rank].detach().cpu().numpy())
  plt.yscale('log')
  plt.ylabel("Singular value")
  plt.xlabel("Rank")
  plt.title("Distribution of the singular vectors")
  plt.show()

def cosine_sim(x,y):
    return torch.dot(x,y) / (torch.norm(x) * torch.norm(y))


def normalize_and_entropy(V, eps=1e-6):
    absV = torch.abs(V)
    normV = absV / torch.sum(absV)
    entropy = torch.sum(normV * torch.log(normV + eps)).item()
    return -entropy
