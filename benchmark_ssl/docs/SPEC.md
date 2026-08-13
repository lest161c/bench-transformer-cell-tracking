# SPEC — Refactor `visualize_embeddings.py` Naming Conventions

## Goal

Rename all short-form and `_t`/`_n` suffixed variables in
`src/cnn_encoder/visualize_embeddings.py` to the descriptive `anchor`/`query`
convention established in `src/edge_probing/harness/evaluation.py`.

**No logic changes** — pure rename.

## Motivation

`evaluation.py` uses the following convention (from its module docstring):

> Anchor (frame t) is the reference; Query (frame t+1) is the candidate.

`visualize_embeddings.py` predates this convention and uses `_t` (frame t)
and `_n` (frame t+1) suffixes, plus short forms like `mt`, `mn`, `ct`, `lt`,
`cn`, `ln`, `enc`, `dec`, `sd`, `ckpt`, `e_c`, `e_r`, `min_n`, `arr`, `pe_mod`.

## Renaming table

### `build_model`
| Old | New |
|-----|-----|
| `enc` | `encoder` |
| `dec` | `decoder` |

### `load_checkpoint`
| Old | New |
|-----|-----|
| `ckpt` | `checkpoint` |
| `ckpt_path` | `checkpoint_path` |

### `try_load_model`
| Old | New |
|-----|-----|
| `cp` | `candidate_path` |
| `ckpt` | `checkpoint` |
| `sd` | `state_dict` |
| `k` (in comprehension) | `key` |

### `extract_embeddings_from_pairs`
| Old | New |
|-----|-----|
| `mt` | `mask_path_anchor` |
| `mn` | `mask_path_query` |
| `img_t_path` | `img_path_anchor` |
| `img_n_path` | `img_path_query` |
| `rt` | `frame_anchor` |
| `rn` | `frame_query` |
| `ct` | `coords_anchor` |
| `lt` | `labels_anchor` |
| `imgt` | `img_anchor` |
| `cn` | `coords_query` |
| `ln` | `labels_query` |
| `imgn` | `img_query` |
| `idx_t` | `idx_anchor` |
| `idx_n` | `idx_query` |
| `ct_s` | `coords_anchor_shared` |
| `lt_s` | `labels_anchor_shared` |
| `cn_s` | `coords_query_shared` |
| `ln_s` | `labels_query_shared` |
| `mask_t` | `mask_anchor` |
| `mask_n` | `mask_query` |
| `feat_7d_t` | `feat_7d_anchor` |
| `feat_7d_n` | `feat_7d_query` |
| `pt_s` | `patches_anchor` |
| `pn_s` | `patches_query` |
| `cnn_t` | `cnn_feat_anchor` |
| `cnn_n` | `cnn_feat_query` |
| `feat_t` | `feats_anchor` |
| `feat_n` | `feats_query` |
| `coords_t` | `coords_tensor_anchor` |
| `coords_n` | `coords_tensor_query` |
| `pe_t` | `pe_anchor` |
| `pe_n` | `pe_query` |
| `enc_t` | `embeddings_anchor` |
| `enc_n` | `embeddings_query` |
| `emb_t_np` | `embeddings_anchor_np` |
| `emb_n_np` | `embeddings_query_np` |
| `l` (in comprehension) | `label` |

### `compute_cka`
| Old | New |
|-----|-----|
| `n` | `num_samples` |
| `m` | `min_samples` |
| `K` | `gram_matrix_x` |
| `L` | `gram_matrix_y` |
| `H` | `centering_matrix` |
| `K_c` | `gram_matrix_x_centered` |
| `L_c` | `gram_matrix_y_centered` |

`X`, `Y` kept as established math notation (representation matrices).
`hsic_xy`, `hsic_xx`, `hsic_yy` kept as established math notation.
`cka` kept.

### `main`
| Old | New |
|-----|-----|
| `ckpt_path` | `checkpoint_path` |
| `alt_path` | `alt_checkpoint_path` |
| `ckpt_dir` | `checkpoint_dir` |
| `alt_dir` | `alt_checkpoint_dir` |
| `pe_mod` | `fourier_pe` |
| `e_c` | `embeddings_mode_c` |
| `e_r` | `embeddings_mode_r` |
| `min_n` | `min_samples` |
| `n` | `num_embeddings` |
| `arr` | `embeddings_array` |
| `embs` | `mode_embeddings` |
| `embed_dim` | `embedding_dim` |

## Acceptance criteria

1. No `_t` or `_n` suffix variable names remain.
2. No short-form variable names remain (`mt`, `mn`, `rt`, `rn`, `ct`, `lt`,
   `cn`, `ln`, `imgt`, `imgn`, `pt_s`, `pn_s`, `enc`, `dec`, `sd`, `ckpt`,
   `cp`, `e_c`, `e_r`, `min_n`, `arr`, `pe_mod`, `n`, `l`, `k`, `m`).
3. Module docstring updated with the anchor/query naming convention.
4. File parses without syntax errors.
5. Module imports without runtime errors (only `main` is imported externally).
6. No logic changes — behavior is identical.
