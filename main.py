import torch
import os
from transformers import LlamaTokenizer, LlamaForCausalLM
import pickle
from pathlib import Path
import deepcopy
import numpy as np
import tabulate

def keep_k(x, k=100, absolute=True, dim=-1):
    shape = x.shape
    x_ = x
    if absolute:
        x_ = abs(x)
    values, indices = torch.topk(x_, k=k, dim=dim)
    res = torch.zeros_like(x)
    res.scatter_(dim, indices, x.gather(dim, indices))
    return res

def get_max_token_length(tokens):
  maxlen = 0
  for t in tokens:
    l = len(t)
    if l > maxlen:
      maxlen = l
  return maxlen

def pad_with_space(t, maxlen):
  spaces_to_add = maxlen - len(t)
  for i in range(spaces_to_add):
    t += " "
  return t

def convert_to_tokens(indices, tokenizer, extended, extra_values_pos, strip=True, pad_to_maxlen=False):
    if extended:
        res = [tokenizer.convert_ids_to_tokens([idx])[0] if idx < len(tokenizer) else 
               (f"[pos{idx-len(tokenizer)}]" if idx < extra_values_pos else f"[val{idx-extra_values_pos}]") 
               for idx in indices]
    else:
        res = tokenizer.convert_ids_to_tokens(indices)
    if strip:
        res = list(map(lambda x: x[1:] if x[0] == 'Ġ' else "#" + x, res))
    if pad_to_maxlen:
      maxlen = get_max_token_length(res)
      res = list(map(lambda t: pad_with_space(t, maxlen), res))
    return res


def top_tokens(v_tok, tokenizer, k=100, only_english=False, only_ascii=True, with_values=False, 
               exclude_brackets=False, extended=True, extra_values=None, pad_to_maxlen=False):
    v_tok = deepcopy(v_tok)
    ignored_indices = []
    if only_ascii:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not val.strip('Ġ').isascii()]
    if only_english: 
        ignored_indices =[key for val, key in tokenizer.vocab.items() if not (val.strip('Ġ').isascii() and val.strip('Ġ[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    v_tok[ignored_indices] = -np.inf
    extra_values_pos = len(v_tok)
    if extra_values is not None:
        v_tok = torch.cat([v_tok, extra_values])
    values, indices = torch.topk(v_tok, k=k)
    res = convert_to_tokens(indices, tokenizer, extended=extended, extra_values_pos=extra_values_pos,pad_to_maxlen = pad_to_maxlen)
    if with_values:
        res = list(zip(res, values.cpu().numpy()))
    return res


def top_matrix_tokens(mat, tokenizer, k=100, rel_thresh=None, thresh=None, 
                      sample_entries=10000, alphabetical=True, only_english=False,
                      exclude_brackets=False, with_values=True, extended=True):

    mat = deepcopy(mat)
    ignored_indices = []
    if only_english:
        ignored_indices = [key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.strip('[]').isalnum())]
    if exclude_brackets:
        ignored_indices = set(ignored_indices).intersection(
            {key for val, key in tokenizer.vocab.items() if not (val.isascii() and val.isalnum())})
        ignored_indices = list(ignored_indices)
    mat[ignored_indices, :] = -np.inf
    mat[:, ignored_indices] = -np.inf
    cond = torch.ones_like(mat).bool()
    if rel_thresh:
        cond &= (mat > torch.max(mat) * rel_thresh)
    if thresh:
        cond &= (mat > thresh)
    entries = torch.nonzero(cond)
    if sample_entries:
        entries = entries[np.random.randint(len(torch.nonzero(cond)), size=sample_entries)]
    res_indices = sorted(entries, 
                         key=lambda x: x[0] if alphabetical else -mat[x[0], x[1]])
    res = [*map(partial(convert_to_tokens, extended=extended, tokenizer=tokenizer), res_indices)]
            
    if with_values:
        res_ = []
        for (x1, x2), (i1, i2) in zip(res, res_indices):
            res_.append((x1, x2, mat[i1][i2].item()))
        res = res_    
    return res

def get_llama_info(model):
    info = {
        "num_layers" : model.config.num_hidden_layers, #number of hidden layers in the transformer encoder
        "hidden_dim" : model.config.hidden_size,
        "num_heads" : model.config.num_attention_heads, #number of attention heaqds for each attention layer in Transformer encoder
        "head_size" : model.config.hidden_size // model.config.num_attention_heads
    }
    return info


def load_llama(model_path,
               tokenizer_path,
               load_from_disk=True): 
    if os.path.exists(model_path) and \
        os.path.exists(tokenizer_path) and load_from_disk:
        model = LlamaForCausalLM()
        model.load_state_dict(model_path)
        with open(tokenizer_path, "rb") as file:
            tokenizer = pickle.load(file)
        return (model, tokenizer)

    BASE_MODEL = "decapoda-research/llama-7b-hf"
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    #save state dict and tokenizer
    model_path = Path("checkpoints/llama_model.pth")
    torch.save(model.state_dict(), model_path)
    tokenizer_path = Path("checkpoints/llama_tokenizer.pkl")
    with open(tokenizer, "wb") as file:
        pickle.dump(tokenizer, file)
    return (model, tokenizer)

def SVD_OV_circuit(W_O, W_V, layer_idx, head_idx, embeddings, all_tokens, k=20, N_singular_vectors=10):
    num_layers, num_heads, hidden_size, head_dim = W_V.shape
    W_V_head = W_V[layer_idx, head_idx]
    W_O_head = W_O[layer_idx, head_idx]
    OV_circuit = W_V_head @ W_O_head
    U,S,V = torch.linalg.svd(OV_circuit)
    print(U.shape)
    print(V.shape)
    #check dim of U,V. 
    singular_embeddings = []
    for i in range(N_singular_vectors): 
        singular_emb = V[i,:].float() @ embeddings
        singular_embeddings.append(singular_emb)
    singular_vector_tokens = [
        top_tokens(singular_embeddings[i].float().cpu(), k=k) for i in range(N_singular_vectors)
    ]
    return tabulate([*singular_embeddings])
    
    

def get_KV_circuits(model, num_layers, hidden_dim, num_heads, head_size):
    pass

def get_MLPin_circuits(model, num_layers, hidden_dim, num_heads, head_size):
    pass

def get_MLPout_circuits(model, num_layers, hidden_dim, num_heads, head_size):
    pass

def get_attention_heads(model, num_layers, hidden_dim, num_heads, head_size):
    W_Q_heads = [], W_K_heads = [], W_V_heads = [], W_O_heads = []
    for j in num_layers: 
        Q = model.get_paramter("model.layers.{j}.self_attn.q_proj.weight").detach()
        K = model.get_parameter("model.layers.{j}.self_attn.k_proj.weight").detach()
        V = model.get_parameter("model.layers.{j}.self_attn.v_proj.weight").detach()
        O = model.get_parameter("model.layers.{j}.self_attn.o_proj.weight").detach()
        
        #normalizing the weights
        # Q = Q - torch.mean(Q, dim=0)
        # K = K - torch.mean(K, dim=0)
        # V = V - torch.mean(V, dim=0)
        # O = O - torch.mean(O, dim=0)
        
        Q = Q.reshape(hidden_dim, num_heads, head_size).permute(1, 0, 2)
        K = K.reshape(hidden_dim, num_heads, head_size).permute(1, 0, 2)
        V = V.reshape(hidden_dim, num_heads, head_size).permute(1, 0, 2)
        O = O.reshape(hidden_dim, num_heads, head_size).permute(1, 0, 2)
    
        W_Q_heads.append(Q)
        W_K_heads.append(K)
        W_V_heads.append(V)
        W_O_heads.heads.append(O)
    return (
        torch.cat(W_Q_heads, dim=0),
        torch.cat(W_K_heads, dim=0),
        torch.cat(W_V_heads, dim=0),
        torch.cat(W_O_heads, dim=0),
    )
        
        


def get_attention_heads(model, num_layers, hidden_dim, num_heads, head_size):
  qkvs = []
  for j in range(num_layers):
    qkv = model.get_parameter(f"transformer.h.{j}.attn.c_attn.weight").detach().T
    ln_weight_1 = model.get_parameter(f"transformer.h.{j}.ln_1.weight").detach()

    qkv = qkv - torch.mean(qkv, dim=0)
    qkv = torch.einsum("oi,i -> oi", qkv, ln_weight_1)
    qkvs.append(qkv.T)

  W_Q, W_K, W_V = torch.cat(qkvs).chunk(3, dim=-1)
  W_O = torch.cat([model.get_parameter(f"transformer.h.{j}.attn.c_proj.weight") for j in range(num_layers)]).etach()
  W_V_heads = W_V.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_O_heads = W_O.reshape(num_layers, num_heads, head_size, hidden_dim)
  W_Q_heads = W_Q.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  W_K_heads = W_K.reshape(num_layers, hidden_dim, num_heads, head_size).permute(0, 2, 1, 3)
  return W_Q_heads, W_K_heads, W_V_heads, W_O_heads



def main():
    print("Updated")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(DEVICE))
    
    model_path = Path("checkpoints/llama_model.pth")
    tokenizer_path = Path("checkpoints/llama_tokenizer.pkl")
    
    model, tokenizer = load_llama(model_path, tokenizer_path)

    model_info = get_llama_info(model)
    print(model_info)
    
    



if __name__ == "__main__":
    main()