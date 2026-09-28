"""DB-only SchemaMem on native Mistral blocks, without trace compression.

Full-sequence training/evaluation and aligned chunk continuation are supported.
There is deliberately no token-generation KV cache in this experimental model.
Each chunk has native causal attention; only S crosses chunk boundaries.
"""
import math

import torch
from torch import nn
from transformers import MistralConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.mistral.modeling_mistral import (
    MistralDecoderLayer, MistralModel, MistralForCausalLM, MistralPreTrainedModel,
)

from modeling_schema_mem import SchemaCache, _phase_features


class SchemaMemDBConfig(MistralConfig):
    model_type = 'schema_mem_db'

    def __init__(self, schema_size=128, schema_update_scale=math.pi,
                 schema_layer_indices=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(schema_size,int) or isinstance(schema_size,bool) or schema_size<1:
            raise ValueError('schema_size must be a positive integer')
        if not self.sliding_window or self.sliding_window<1:
            raise ValueError('sliding_window must be a positive commit period')
        if not math.isfinite(schema_update_scale) or schema_update_scale<=0:
            raise ValueError('schema_update_scale must be finite and positive')
        self.schema_size=schema_size
        self.schema_update_scale=schema_update_scale
        indices=list(range(self.num_hidden_layers)) if schema_layer_indices is None else list(schema_layer_indices)
        if (not indices or any(not isinstance(i,int) or isinstance(i,bool) or
                not 0<=i<self.num_hidden_layers for i in indices) or len(set(indices))!=len(indices)):
            raise ValueError('schema_layer_indices must contain unique valid indices and at least one schema layer')
        self.schema_layer_indices=sorted(indices)


class SchemaTraceAttentionLayer(MistralDecoderLayer):
    """Native Mistral attention/MLP with dense phasor reads and independent writes."""

    def __init__(self,config,layer_idx):
        super().__init__(config,layer_idx)
        self.config=config;self.layer_idx=layer_idx;h=config.hidden_size
        self.schema_enabled=layer_idx in config.schema_layer_indices
        if not self.schema_enabled:
            # Native attention/MLP only. The outer model still restricts this
            # layer to the current chunk and never passes a previous KV cache.
            return
        self.schema_embedding=nn.Parameter(torch.empty(config.schema_size,h))
        self.schema_key_proj=nn.Linear(2*h,h,bias=False)
        self.schema_value_proj=nn.Linear(2*h,h,bias=False)
        self.schema_read_query_proj=nn.Linear(h,h,bias=False)
        self.schema_write_query_proj=nn.Linear(h,h,bias=False)
        self.schema_write_delta_proj=nn.Linear(h,h,bias=False)
        self.schema_write_delta_proj._schema_zero_init=True
        self.schema_update_output_proj=nn.Linear(h,h,bias=False)

    def _project_schema(self,state):
        angles=self.schema_embedding.float().unsqueeze(0)
        if state is not None: angles=angles+state.float()
        features=_phase_features(angles).to(self.schema_key_proj.weight.dtype)
        return self.schema_key_proj(features),self.schema_value_proj(features)

    def _address(self,query,keys):
        return (query.float()@keys.float().transpose(-1,-2)/math.sqrt(self.hidden_size)).softmax(-1).to(query.dtype)

    def _encode_traces(self,hidden_states,schema_state):
        keys,values=self._project_schema(schema_state)
        weights=self._address(self.schema_read_query_proj(self.input_layernorm(hidden_states)),keys)
        return weights@values,keys,values

    def _extract_schema_deltas(self,normalized):
        # Own-layer S and own-layer attention output NEVER enter the writer.
        keys,values=self._project_schema(None)
        weights=self._address(self.schema_write_query_proj(normalized),keys)
        features=self.schema_write_delta_proj(normalized+weights@values)
        phase=self.config.schema_update_scale*features.float().tanh()
        updates=weights.float().transpose(1,2)@phase
        return features,updates

    def forward(self,hidden_states,attention_mask=None,position_ids=None,
                position_embeddings=None,schema_cache=None,**kwargs):
        if not self.schema_enabled:
            return super().forward(hidden_states,attention_mask=attention_mask,
                position_ids=position_ids,position_embeddings=position_embeddings,
                past_key_values=None,use_cache=False)
        if schema_cache is None: raise ValueError('SchemaMemDBModel must supply schema_cache')
        normalized=self.input_layernorm(hidden_states)
        read,_,_=self._encode_traces(hidden_states,schema_cache.get_state(self.layer_idx))
        attended,_=self.self_attn(hidden_states=normalized+read,
            attention_mask=attention_mask,position_ids=position_ids,
            position_embeddings=position_embeddings,past_key_values=None,use_cache=False)
        features,updates=self._extract_schema_deltas(normalized)
        result=hidden_states+attended+self.schema_update_output_proj(features)
        result=result+self.mlp(self.post_attention_layernorm(result))
        schema_cache.update(updates,self.layer_idx,num_tokens=hidden_states.shape[1])
        return result


class SchemaMemDBModel(MistralModel):
    config_class=SchemaMemDBConfig

    def __init__(self,config):
        super().__init__(config)
        self.layers=nn.ModuleList([SchemaTraceAttentionLayer(config,i) for i in range(config.num_hidden_layers)])
        self.post_init()

    def _init_weights(self,module):
        super()._init_weights(module)
        if isinstance(module,SchemaTraceAttentionLayer) and module.schema_enabled:
            nn.init.uniform_(module.schema_embedding,-math.pi,math.pi)
        if getattr(module,'_schema_zero_init',False): nn.init.zeros_(module.weight)

    def iter_trace_layers(self):
        return (layer for layer in self.layers if layer.schema_enabled)

    def forward(self,input_ids=None,inputs_embeds=None,schema_cache=None,
                attention_mask=None,position_ids=None,past_key_values=None,use_cache=False,**kwargs):
        if use_cache or past_key_values is not None:
            raise ValueError('DB experiment supports full-sequence evaluation, not generation KV caching')
        if attention_mask is not None:
            raise ValueError('DB padding is a real blank token; external padding masks are not supported')
        if (input_ids is None)==(inputs_embeds is None):
            raise ValueError('Supply exactly one of input_ids and inputs_embeds')
        cache=SchemaCache() if schema_cache is None else schema_cache
        schema_indices=self.config.schema_layer_indices
        if any(cache.get_pending_length(i) for i in schema_indices):
            raise ValueError('Continuation must start at a committed chunk boundary')
        offset=cache.get_seq_length(schema_indices[0])
        if any(cache.get_seq_length(i)!=offset for i in schema_indices):
            raise ValueError('All layer states must belong to the same sequence')
        hidden=self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        if position_ids is None:
            position_ids=torch.arange(offset,offset+hidden.shape[1],device=hidden.device)[None]
        outputs=[];window=self.config.sliding_window
        for start in range(0,hidden.shape[1],window):
            end=min(start+window,hidden.shape[1])
            # Reuse native mask construction, model-level RoPE, final norm and
            # layer loop. No KV from earlier chunks is supplied or retained.
            part=super().forward(inputs_embeds=hidden[:,start:end],
                position_ids=position_ids[:,start:end],schema_cache=cache,use_cache=False)
            outputs.append(part.last_hidden_state)
            if end-start==window: cache.commit()
        return BaseModelOutputWithPast(last_hidden_state=torch.cat(outputs,dim=1))


class SchemaMemDBForCausalLM(MistralForCausalLM):
    config_class=SchemaMemDBConfig

    def __init__(self,config):
        MistralPreTrainedModel.__init__(self,config)
        self.model=SchemaMemDBModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
        self.post_init()
