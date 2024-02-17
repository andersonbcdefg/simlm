import copy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import nullcontext
from torch import Tensor
from torch.distributions import Categorical
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoModelForMaskedLM, ElectraModel
from transformers.modeling_outputs import MaskedLMOutput, ModelOutput
from transformers.models.bert import BertForMaskedLM
from transformers.configuration_bert import BertConfig

from logger_config import logger
from config import Arguments
from utils import slice_batch_dict

class BertFlashAttention(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = config.attention_probs_dropout_prob
        self.position_embedding_type = "absolute"
        self.max_position_embeddings = config.max_position_embeddings
        self.is_decoder = False

    @classmethod
    def from_bert_attention(
        cls,
        config,
        bert_attention_layer,
    ):
        config = config
        new_layer = cls(config)

        # Copy parameters
        with torch.no_grad():
            new_layer.query.weight.copy_(bert_attention_layer.query.weight)
            new_layer.query.bias.copy_(bert_attention_layer.query.bias)
            new_layer.key.weight.copy_(bert_attention_layer.key.weight)
            new_layer.key.bias.copy_(bert_attention_layer.key.bias)
            new_layer.value.weight.copy_(bert_attention_layer.value.weight)
            new_layer.value.bias.copy_(bert_attention_layer.value.bias)

        # Delete original tensors to free up memory
        del bert_attention_layer

        return new_layer

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        reshapes tensor from (batch_size, seq_len, all_head_size) to (batch_size, num_heads, seq_len, head_size)
        for xformers, input tensors must be in format [B, M, H, K], where B is the batch size, M the sequence length,
        H is the number of heads, and K the embeding size per head
        """
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        return x.view(new_x_shape).permute(0, 2, 1, 3)  # b, h, m, k

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        bias = attention_mask.view(batch_size, 1, seq_len)  # b, 1, k_len
        bias = bias.expand(-1, seq_len, -1).unsqueeze(1)  # b, q_len, k_len
        # print bias
        out = F.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            dropout_p=self.dropout,
            attn_mask=bias,  # b, 1, q_len, k_len, broadcast happens automatically
        ).transpose(
            1, 2
        )  # b, m, h, k

        out = out.flatten(2)  # b, m, h * k
        return out, None

@dataclass
class ReplaceLMOutput(ModelOutput):
    loss: Optional[Tensor] = None
    encoder_mlm_loss: Optional[Tensor] = None
    decoder_mlm_loss: Optional[Tensor] = None
    g_mlm_loss: Optional[Tensor] = None
    replace_ratio: Optional[Tensor] = None


class ReplaceLM(nn.Module):
    def __init__(self, args: Arguments,
                 bert: BertForMaskedLM):
        super(ReplaceLM, self).__init__()
        self.encoder = bert
        self.decoder = copy.deepcopy(self.encoder.bert.encoder.layer[-args.rlm_decoder_layers:])
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.generator: ElectraModel = AutoModelForMaskedLM.from_pretrained(args.rlm_generator_model_name)

        # replace all attn layers in encoder, decoder, generator with flash attention
        for idx, layer in enumerate(self.encoder.bert.encoder.layer):
            print("Replacing attention in encoder layer", idx, "with flash attention.")
            layer.attention.self = BertFlashAttention.from_bert_attention(
                self.encoder.bert.config,
                layer.attention.self
            )

        for idx, layer in enumerate(self.decoder):
            print("Replacing attention in decoder layer", idx, "with flash attention.")
            layer.attention.self = BertFlashAttention.from_bert_attention(
                self.encoder.bert.config,
                layer.attention.self
            )

        for idx, layer in enumerate(self.generator.electra.encoder.layer):
            print("Replacing attention in generator layer", idx, "with flash attention.")
            layer.attention.self = BertFlashAttention.from_bert_attention(
                self.generator.electra.config,
                layer.attention.self
            )

        # pad all vocabs to multiple of 128
        self.encoder.resize_token_embeddings(pad_to_multiple_of=128)
        self.generator.resize_token_embeddings(pad_to_multiple_of=128)

        if args.rlm_freeze_generator:
            self.generator.eval()
            self.generator.requires_grad_(False)

        self.args = args

        from trainers.rlm_trainer import ReplaceLMTrainer
        self.trainer: Optional[ReplaceLMTrainer] = None

    def forward(self, model_input: Dict[str, torch.Tensor]) -> ReplaceLMOutput:
        enc_prefix, dec_prefix = 'enc_', 'dec_'
        encoder_inputs = slice_batch_dict(model_input, enc_prefix)
        decoder_inputs = slice_batch_dict(model_input, dec_prefix)
        labels = model_input['labels']

        enc_sampled_input_ids, g_mlm_loss = self._replace_tokens(encoder_inputs)
        if self.args.rlm_freeze_generator:
            g_mlm_loss = torch.tensor(0, dtype=torch.float, device=g_mlm_loss.device)
        dec_sampled_input_ids, _ = self._replace_tokens(decoder_inputs, no_grad=True)

        encoder_inputs['input_ids'] = enc_sampled_input_ids
        decoder_inputs['input_ids'] = dec_sampled_input_ids
        # use the un-masked version of labels
        encoder_inputs['labels'] = labels
        decoder_inputs['labels'] = labels

        is_replaced = (encoder_inputs['input_ids'] != labels) & (labels >= 0)
        replace_cnt = is_replaced.long().sum().item()
        total_cnt = (encoder_inputs['attention_mask'] == 1).long().sum().item()
        replace_ratio = torch.tensor(replace_cnt / total_cnt, device=g_mlm_loss.device)

        encoder_out: MaskedLMOutput = self.encoder(
            **encoder_inputs,
            output_hidden_states=True,
            return_dict=True)

        # batch_size x 1 x hidden_dim
        cls_hidden = encoder_out.hidden_states[-1][:, :1]
        # batch_size x seq_length x embed_dim
        dec_inputs_embeds = self.encoder.bert.embeddings(decoder_inputs['input_ids'])
        hiddens = torch.cat([cls_hidden, dec_inputs_embeds[:, 1:]], dim=1)

        attention_mask = self.encoder.get_extended_attention_mask(
            encoder_inputs['attention_mask'],
            encoder_inputs['attention_mask'].shape,
            encoder_inputs['attention_mask'].device
        )

        for layer in self.decoder:
            layer_out = layer(hiddens, attention_mask)
            hiddens = layer_out[0]

        decoder_mlm_loss = self.mlm_loss(hiddens, labels)

        loss = decoder_mlm_loss + encoder_out.loss + g_mlm_loss * self.args.rlm_generator_mlm_weight

        return ReplaceLMOutput(loss=loss,
                               encoder_mlm_loss=encoder_out.loss.detach(),
                               decoder_mlm_loss=decoder_mlm_loss.detach(),
                               g_mlm_loss=g_mlm_loss.detach(),
                               replace_ratio=replace_ratio)

    def _replace_tokens(self, batch_dict: Dict[str, torch.Tensor],
                        no_grad: bool = False) -> Tuple:
        with torch.no_grad() if self.args.rlm_freeze_generator or no_grad else nullcontext():
            outputs: MaskedLMOutput = self.generator(
                **batch_dict,
                return_dict=True)

        with torch.no_grad():
            sampled_input_ids = Categorical(logits=outputs.logits).sample()
            is_mask = (batch_dict['labels'] >= 0).long()
            sampled_input_ids = batch_dict['input_ids'] * (1 - is_mask) + sampled_input_ids * is_mask

        return sampled_input_ids.long(), outputs.loss

    def mlm_loss(self, hiddens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pred_scores = self.encoder.cls(hiddens)
        mlm_loss = self.cross_entropy(
            pred_scores.view(-1, self.encoder.config.vocab_size),
            labels.view(-1))
        return mlm_loss

    @classmethod
    def from_pretrained(cls, all_args: Arguments,
                        model_name_or_path: str, *args, **kwargs):
        hf_model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, *args, **kwargs)
        model = cls(all_args, hf_model)
        decoder_save_path = os.path.join(model_name_or_path, 'decoder.pt')
        if os.path.exists(decoder_save_path):
            logger.info('loading extra weights from local files')
            state_dict = torch.load(decoder_save_path, map_location="cpu")
            model.decoder.load_state_dict(state_dict)
        return model

    def save_pretrained(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)
        torch.save(self.decoder.state_dict(), os.path.join(output_dir, 'decoder.pt'))
