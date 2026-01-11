import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, get_peft_model
from transformers import GPT2Model

# ===================== Constant Definition (Extract magic numbers for unified adjustment) =====================
CONST_LORA_RANK = 16
CONST_LORA_ALPHA = 32
CONST_LAYER_NORM_EPS = 1e-5
CONST_CROSS_MODAL_SCALE = 0.01
CONST_DEFAULT_TIME_STEPS = 144
CONST_SPATIAL_EMB_DIM_DEFAULT = 64
CONST_DROPOUT_DEFAULT = 0.1


# ===================== Data Class Definition =====================
@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    """
    Custom output class for GPT2-based model with cross-modal attention
    Contain the hidden states and attention outputs from the transformer backbone
    """
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# ===================== Temporal Embedding Module =====================
class TemporalEmbedding(nn.Module):
    """
    Learnable temporal embedding module for spatio-temporal data
    Integrates daily time embedding and weekly time embedding to capture periodic temporal features
    """
    def __init__(
        self, time_steps: int, embed_dim: int, time_day_idx: int = -2, week_emb_idx: int = -1
    ):
        super().__init__()
        self.time_steps = time_steps
        self.time_day_idx = time_day_idx
        self.week_emb_idx = week_emb_idx

        # Learnable embedding parameters for temporal features
        self.time_day_emb = nn.Parameter(torch.empty(time_steps, embed_dim))
        self.time_week_emb = nn.Parameter(torch.empty(7, embed_dim))
        nn.init.xavier_uniform_(self.time_day_emb)
        nn.init.xavier_uniform_(self.time_week_emb)

        print(f"🔧 TemporalEmbedding Initialized | Daily time steps: {time_steps} | Embedding dim: {embed_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation of temporal embedding module
        Args:
            x: Input tensor with shape [batch_size, in_steps, num_nodes, feat_dim]
        Returns:
            Concatenated temporal embedding with shape [batch_size, 2*embed_dim, num_nodes, 1]
        """
        batch_size, _, num_nodes, _ = x.shape

        # Daily time feature embedding
        day_feat = x[..., self.time_day_idx]
        day_indices = (day_feat[:, -1, :] * self.time_steps).long()
        day_indices = torch.clamp(day_indices, 0, self.time_steps - 1)
        time_day = self.time_day_emb[day_indices]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)

        # Weekly time feature embedding
        week_feat = x[..., self.week_emb_idx]
        week_indices = week_feat[:, -1, :].long()
        week_indices = torch.clamp(week_indices, 0, 6)
        time_week = self.time_week_emb[week_indices]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)

        # Concatenate daily and weekly temporal embeddings
        concatenated = torch.cat([time_day, time_week], dim=1)
        return concatenated


# ===================== Core PFA Module (GPT2+LoRA+Cross-Modal Attention) =====================
class PFA(nn.Module):
    """
    Core Position-aware Fusion Attention (PFA) Module
    Integrate GPT2 backbone with LoRA lightweight fine-tuning and cross-modal attention mechanism
    Realize the bidirectional alignment of time-series feature and text feature in the last U layers
    """
    def __init__(
        self,
        device="cuda:0",
        gpt_layers=6,
        U=1,
        dropout_rate=CONST_DROPOUT_DEFAULT,
        cross_modal_config=None,
    ):
        super(PFA, self).__init__()
        # Load pretrained GPT2 model with eager attention implementation
        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", attn_implementation="eager", output_attentions=True, output_hidden_states=True
        )
        # Truncate GPT2 to specified layers for efficiency
        self.gpt2.h = self.gpt2.h[:gpt_layers]
        self.U = U
        self.device = device
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.lora_rank = CONST_LORA_RANK

        # LoRA configuration for lightweight fine-tuning (freeze main backbone, only train adapter)
        self.lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=CONST_LORA_ALPHA,
            lora_dropout=self.dropout_rate,
            target_modules=["q_attn", "c_attn"],
            bias="none",
        )

        # Default configuration for cross-modal attention layer
        default_cross_modal_config = {
            "nhead": 8,
            "num_layers": 1,
            "norm_first": True,
            "batch_first": True,
        }
        if cross_modal_config is not None:
            default_cross_modal_config.update(cross_modal_config)

        # Initialize layer-wise cross-modal multi-head attention for text alignment
        self.cross_modal_align_text = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.gpt2.config.n_embd,
                    num_heads=default_cross_modal_config.get("nhead", 8),
                    dropout=self.dropout_rate,
                    batch_first=default_cross_modal_config.get("batch_first", True),
                )
                for _ in range(gpt_layers)
            ]
        )
        self.cross_modal_align_text_ln = nn.ModuleList(
            [
                nn.LayerNorm(self.gpt2.config.n_embd, eps=CONST_LAYER_NORM_EPS)
                for _ in range(gpt_layers)
            ]
        )

        # Initialize layer-wise cross-modal multi-head attention for time-series alignment
        self.cross_modal_align_time_series = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.gpt2.config.n_embd,
                    num_heads=default_cross_modal_config.get("nhead", 8),
                    dropout=self.dropout_rate,
                    batch_first=default_cross_modal_config.get("batch_first", True),
                )
                for _ in range(gpt_layers)
            ]
        )
        self.cross_modal_align_time_series_ln = nn.ModuleList(
            [
                nn.LayerNorm(self.gpt2.config.n_embd, eps=CONST_LAYER_NORM_EPS)
                for _ in range(gpt_layers)
            ]
        )

        # Apply LoRA adapter to GPT2 backbone
        self.gpt2 = get_peft_model(self.gpt2, self.lora_config)

        # Layer-wise parameter freezing for fine-grained training control
        # Strategy: partial unfreeze for shallow layers, selective unfreeze for deep layers
        for layer_index, layer in enumerate(self.gpt2.h):
            for name, param in layer.named_parameters():
                if layer_index < gpt_layers - self.U:
                    param.requires_grad = True if ("ln" in name or "wpe" in name) else False
                else:
                    param.requires_grad = False if "mlp" in name else True

    def custom_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        adjacency_matrix: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, dict]:
        """
        Custom forward function for GPT2 with cross-modal attention integration
        Add bidirectional cross-modal alignment between time-series and text features in the last U layers
        Args:
            input_ids: Input token ids for text modality
            inputs_embeds: Precomputed input embeddings (alternative of input_ids)
            adjacency_matrix: Adjacency matrix for spatial graph attention mask
            others: Standard GPT2 forward parameters
        Returns:
            Model output with fused cross-modal features
        """
        output_attentions = output_attentions or self.gpt2.config.output_attentions
        output_hidden_states = output_hidden_states or self.gpt2.config.output_hidden_states
        use_cache = use_cache or self.gpt2.config.use_cache
        return_dict = return_dict or self.gpt2.config.use_return_dict

        # Input validity check: mutually exclusive for input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both 'input_ids' and 'inputs_embeds'")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Must specify either 'input_ids' or 'inputs_embeds'")

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        past_length = past_key_values[0][0].size(-2) if past_key_values else 0
        past_key_values = (
            tuple([None] * len(self.gpt2.h)) if past_key_values is None else past_key_values
        )

        # Position encoding initialization (standard GPT2 absolute position encoding)
        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_shape[-1] + past_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        # Token embedding + position embedding fusion
        if inputs_embeds is None:
            inputs_embeds = self.gpt2.wte(input_ids)
        position_embeds = self.gpt2.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        all_self_attentions = () if output_attentions else None
        presents = () if use_cache else None
        total_layers = len(self.gpt2.h)

        # GPT layer-wise forward propagation + cross-modal interaction in last U layers
        for i, (block, layer_past) in enumerate(zip(self.gpt2.h, past_key_values)):
            if i >= total_layers - self.U:
                # Use adjacency matrix as spatial attention mask for graph structure constraint
                attention_mask = (
                    adjacency_matrix.to(hidden_states.device).float()
                    if adjacency_matrix is not None
                    else None
                )
                # Split hidden states into time-series and text modality features
                hidden_states_time_series, hidden_states_text = hidden_states.chunk(2)

                # LayerNorm + Cross-modal attention alignment (time-series -> text)
                h_ts_ln = self.cross_modal_align_time_series_ln[i](hidden_states_time_series)
                h_tx_ln = self.cross_modal_align_text_ln[i](hidden_states_text)

                outputs_time_series, _ = self.cross_modal_align_time_series[i](
                    query=h_ts_ln, key=h_tx_ln, value=h_tx_ln
                )
                outputs_time_series = (
                    hidden_states_time_series + CONST_CROSS_MODAL_SCALE * outputs_time_series
                )

                # LayerNorm + Cross-modal attention alignment (text -> time-series)
                outputs_text, _ = self.cross_modal_align_text[i](
                    query=h_tx_ln, key=h_ts_ln, value=h_ts_ln
                )
                outputs_text = hidden_states_text + CONST_CROSS_MODAL_SCALE * outputs_text

                # Concatenate aligned cross-modal features
                hidden_states = torch.cat([outputs_time_series, outputs_text], dim=0)

            # Standard GPT2 block forward pass
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i] if head_mask is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]

            # Cache key-value pairs and attention weights for subsequent inference
            if use_cache and len(outputs) > 1:
                presents = presents + (outputs[1],)
            if output_attentions and len(outputs) > 2:
                all_self_attentions = all_self_attentions + (outputs[2],)

        # Final layer normalization and tensor reshape
        hidden_states = self.gpt2.ln_f(hidden_states)
        hidden_states = hidden_states.view((-1,) + input_shape[1:] + (hidden_states.size(-1),))

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            attentions=all_self_attentions,
        )

    def forward(self, x, adjacency_matrix):
        """
        Wrapper for custom forward function
        Args:
            x: Input feature tensor for PFA module
            adjacency_matrix: Spatial adjacency matrix for graph attention
        Returns:
            Dropout-applied fused feature tensor
        """
        batch_size = x.shape[0]
        num_heads = self.gpt2.config.n_head

        # Build attention mask from adjacency matrix
        if adjacency_matrix is not None:
            attention_mask = adjacency_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
            attention_mask = attention_mask.unsqueeze(1).repeat(1, num_heads, 1, 1)
        else:
            attention_mask = (
                torch.ones(batch_size, num_heads, x.shape[1], x.shape[1]).to(self.device).float()
            )

        # Forward propagation and dropout regularization
        output = self.custom_forward(
            inputs_embeds=x, attention_mask=attention_mask
        ).last_hidden_state
        output = self.dropout(output)
        return output


# ===================== Main Model: MobileLLM =====================
class MobileLLM(nn.Module):
    """
    Main Spatio-Temporal Large Language Model (MobileLLM)
    End-to-end model for spatio-temporal forecasting task
    Integrate temporal embedding, PFA cross-modal fusion, transformer encoder/decoder for external feature alignment
    """
    def __init__(
        self,
        config: Dict[str, Any],
        device,
        adj_mx=None,
        input_dim=3,
        num_nodes=170,
        in_steps=12,
        out_steps=12,
        output_dim=None,
        gpt_layers=6,
        U=1,
        dataset_name: str = "milan",
    ):
        super().__init__()
        self.config = self._load_config(config)
        self.device = device
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.output_dim = output_dim if output_dim is not None else max(1, input_dim - 2)
        self.gpt_layers = gpt_layers
        self.U = U
        self.dataset_name = dataset_name

        # Load and preprocess adjacency matrix for spatial graph structure
        loaded_adj = self._load_adjacency_matrix(adj_mx)
        self.adj_mx = (
            loaded_adj if loaded_adj is not None else torch.eye(self.num_nodes).to(self.device)
        )
        # Initialize architecture hyperparameters from config
        self._init_architecture_params()
        # Print model configuration for verification
        self._print_model_info()
        # Initialize all submodules of the model
        self._init_submodules()
        # Move model to target device (GPU/CPU)
        self.to(self.device)

    def _load_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load and parse model configuration dictionary"""
        return config

    def _process_adjacency_matrix(self, adj_matrix):
        """
        Adjacency matrix preprocessing pipeline for spatial graph
        Including: symmetrization, sparsification, normalization, self-loop addition
        Args:
            adj_matrix: Raw adjacency matrix (numpy array / torch tensor)
        Returns:
            Preprocessed torch tensor adjacency matrix on target device
        """
        adj_config = self.config.get("model_architecture", {}).get("adjacency_matrix", {})
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.cpu().numpy()

        # Symmetrize adjacency matrix for undirected graph
        if adj_config.get("preprocessing", {}).get("symmetrize", True):
            adj_matrix = (adj_matrix + adj_matrix.T) / 2
        # Sparsify matrix by thresholding small values to reduce computation
        if adj_config.get("preprocessing", {}).get("sparsify", False):
            threshold = adj_config["preprocessing"].get("sparsity_threshold", 0.01)
            adj_matrix[adj_matrix < threshold] = 0

        # Normalize adjacency matrix to avoid gradient explosion/vanishing
        norm_method = adj_config.get("normalization", {}).get("method", "symmetric")
        if norm_method == "symmetric":
            D = np.diag(np.sum(adj_matrix, axis=1))
            D_inv_sqrt = np.power(D, -0.5)
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
            adj_matrix = D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        elif norm_method == "row_normalized":
            row_sums = np.sum(adj_matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            adj_matrix = adj_matrix / row_sums
        elif norm_method == "softmax":
            adj_matrix = np.exp(adj_matrix) / np.sum(np.exp(adj_matrix), axis=1, keepdims=True)

        # Add self-loop to preserve node self-feature
        if adj_config.get("normalization", {}).get("add_self_loop", True):
            np.fill_diagonal(adj_matrix, 1.0)

        return torch.from_numpy(adj_matrix).float().to(self.device)

    def _load_adjacency_matrix(self, adj_mx=None):
        """
        Multi-source adjacency matrix loading strategy
        Priority: Custom matrix > Dynamic embedding > Predefined file
        Args:
            adj_mx: Custom adjacency matrix input
        Returns:
            Preprocessed adjacency matrix tensor
        """
        if adj_mx is not None:
            return self._process_adjacency_matrix(adj_mx)

        adj_config = self.config.get("model_architecture", {}).get("adjacency_matrix", {})
        adj_type = adj_config.get("type", "predefined")
        if adj_type == "dynamic_embedding":
            print("🔄 Using dynamically generated adjacency matrix (node embedding based)")
            return None

        # Load predefined adjacency matrix from pickle file
        file_paths = adj_config.get("file_paths", {})
        adj_file = file_paths.get(self.dataset_name, file_paths.get("default"))
        with open(adj_file, "rb") as f:
            _, _, adj_matrix = pickle.load(f)
        print(f"✅ Loaded adjacency matrix from file: {adj_file}")
        return self._process_adjacency_matrix(adj_matrix)

    def _init_architecture_params(self):
        """Initialize model architecture hyperparameters from config file"""
        model_arch = self.config.get("model_architecture", {})
        dataset_map = {177: "milan", 320: "shanghai", 94: "trento"}
        dataset_type = dataset_map.get(self.num_nodes, "default")

        self.time_steps = model_arch.get("time_steps", {}).get(
            dataset_type, CONST_DEFAULT_TIME_STEPS
        )
        self.gpt_channel = model_arch.get("gpt_channel", 256)
        self.to_gpt_channel = model_arch.get("to_gpt_channel", 768)
        self.time_day_idx = model_arch.get("feature_indices", {}).get("time_day", 1)
        self.week_emb_idx = model_arch.get("feature_indices", {}).get("week_emb", 2)
        self.dropout_rate = model_arch.get("dropout", CONST_DROPOUT_DEFAULT)
        self.activation = model_arch.get("activation", "leaky_relu")
        self.spatial_embedding_dim = 0  # Remove node spatial embedding feature

    def _print_model_info(self) -> None:
        """Print model configuration and hyperparameters for logging"""
        print("=" * 60)
        print(f"📊 ST-LLM-Plus Model Configuration | Dataset: {self.dataset_name} | Device: {self.device}")
        print(
            f"🔹 Spatio-temporal Params: Nodes={self.num_nodes} | Input steps={self.in_steps} | Output steps={self.out_steps}"
        )
        print(
            f"🔹 Feature Dim: Input={self.input_dim} | Output={self.output_dim} | GPT channel={self.gpt_channel}"
        )
        print(
            f"🔹 LLM Params: GPT layers={self.gpt_layers} | Trainable layers={self.U} | Time steps={self.time_steps}"
        )
        print(f"🔹 Regularization: Dropout={self.dropout_rate} | Activation={self.activation}")
        print("=" * 60)

    def _init_submodules(self):
        """
        Initialize all submodules of MobileLLM
        Critical Fix: Pre-create all trainable layers to avoid dynamic layer creation during forward pass
        """
        # Temporal embedding layer
        self.temporal_emb = TemporalEmbedding(
            time_steps=self.time_steps, embed_dim=self.gpt_channel
        )

        # Input feature projection layer (dim reduction/expansion)
        self.input_proj = nn.Conv2d(
            self.input_dim * self.in_steps, self.gpt_channel, kernel_size=(1, 1)
        )

        # Core PFA fusion layer
        self.pfa_layer = PFA(
            device=self.device, gpt_layers=self.gpt_layers, U=self.U, dropout_rate=self.dropout_rate
        )

        # Feature fusion layer: input projection + temporal embedding (2*gpt_channel for day+week)
        fusion_in_chan = self.gpt_channel + (self.gpt_channel * 2)
        self.feat_fusion = nn.Conv2d(fusion_in_chan, self.to_gpt_channel, kernel_size=(1, 1))

        # Dimension adjustment convolution layer
        self.dim_adjust = nn.Conv2d(self.to_gpt_channel, self.to_gpt_channel, kernel_size=(1, 1))

        # External embedding encoder/decoder for auxiliary feature alignment
        self.external_proj = None
        self.external_emb_dropout = nn.Dropout(p=self.dropout_rate)
        self.prompt_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.to_gpt_channel,
                nhead=8,
                batch_first=True,
                norm_first=True,
                dropout=self.dropout_rate,
            ),
            num_layers=2,
        )
        self.cross_modal_align = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.to_gpt_channel,
                nhead=8,
                batch_first=True,
                norm_first=True,
                dropout=self.dropout_rate,
            ),
            num_layers=1,
        )

        # Final regression head for spatio-temporal forecasting
        self.regressor = nn.Conv2d(
            self.to_gpt_channel * 2, self.out_steps * self.output_dim, kernel_size=(1, 1)
        )

    def param_num(self):
        """Calculate total number of model parameters"""
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        """Calculate number of trainable model parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_params(self) -> int:
        """Print parameter statistics (total/trainable) and return total params"""
        total_params = self.param_num()
        trainable_params = self.count_trainable_params()
        print(
            f"\n📈 Model Parameter Statistics | Total Params: {total_params:,} | Trainable Params: {trainable_params:,} ({trainable_params/total_params:.1%})"
        )
        return total_params

    def forward(self, history_data, external_emb=None):
        """
        End-to-end forward propagation of MobileLLM for spatio-temporal forecasting
        Args:
            history_data: Input spatio-temporal sequence with shape [batch, in_steps, num_nodes, input_dim]
            external_emb: Optional external auxiliary embedding with shape [batch, num_nodes, emb_dim]
        Returns:
            Forecasting result with shape [batch, out_steps, num_nodes, output_dim]
        """
        batch_size = history_data.shape[0]

        # Step 1: Input feature permutation and projection
        x_permuted = history_data.permute(0, 3, 2, 1)
        x_time_flat = x_permuted.transpose(1, 2).contiguous().view(batch_size, self.num_nodes, -1)
        x_time_flat = x_time_flat.transpose(1, 2).unsqueeze(-1)
        x_proj = self.input_proj(x_time_flat)

        # Step 2: Extract temporal embedding features
        tem_emb = self.temporal_emb(history_data)

        # Step 3: Multi-feature fusion (input projection + temporal embedding)
        fusion_inputs = [x_proj, tem_emb]
        fused_feat = self.feat_fusion(torch.cat(fusion_inputs, dim=1))
        gpt_input = fused_feat.squeeze(-1).transpose(1, 2)

        # Step 4: Optional external embedding fusion and alignment
        if external_emb is not None:
            external_emb = external_emb.to(self.device)
            # Dynamic projection layer for external embedding (adaptive to different dim)
            if self.external_proj is None or external_emb.shape[-1] != self.to_gpt_channel:
                self.external_proj = nn.Linear(external_emb.shape[-1], self.to_gpt_channel).to(
                    self.device
                )
            external_emb = self.external_proj(external_emb)

            external_emb = self.external_emb_dropout(external_emb)
            encoded_prompt = self.prompt_encoder(external_emb)
            aligned_feat = self.cross_modal_align(tgt=encoded_prompt, memory=gpt_input)
            aligned_feat = aligned_feat.transpose(1, 2).unsqueeze(-1)

            fused_feat = self.dim_adjust(torch.cat([fused_feat, aligned_feat], dim=0))
            gpt_input = fused_feat.squeeze(-1).transpose(1, 2)

        # Step 5: Core PFA cross-modal fusion forward pass
        gpt_output = self.pfa_layer(gpt_input, None)
        gpt_output = torch.cat(gpt_output.chunk(2, dim=0), dim=-1)
        gpt_output = gpt_output.transpose(1, 2).unsqueeze(-1)

        # Step 6: Regression prediction and tensor reshape to target output shape
        pred_flat = self.regressor(gpt_output).squeeze(-1)
        pred = pred_flat.transpose(1, 2)
        pred = pred.view(-1, self.num_nodes, self.out_steps, self.output_dim).transpose(1, 2)

        return pred

