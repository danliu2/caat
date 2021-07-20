import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from fairseq import search, utils
from fairseq.data import data_utils
from fairseq.models import FairseqIncrementalDecoder
from fairseq.sequence_generator import SequenceGenerator, EnsembleModel



class StageGenerator(SequenceGenerator):
    def __init__(
        self,
        models,
        **kwargs
    ):
        self.asr_1best=False
        models= JointEnsembel(models)
        super().__init__(models, **kwargs)
    

    def finalize_hypos(
        self,
        prev_steps:Tensor,
        prev_positions:Tensor,
        step: int,
        bbsz_idx,
        eos_scores,
        tokens,
        scores,
        finalized: List[List[Dict[str, Tensor]]],
        finished: List[bool],
        beam_size: int,
        attn: Optional[Tensor],
        max_len: int,
    ):
        """Finalize hypothesis, store finalized information in `finalized`, and change `finished` accordingly.
        A sentence is finalized when {beam_size} finished items have been collected for it.

        Returns number of sentences (not beam items) being finalized.
        These will be removed from the batch and not processed further.
        Args:
            bbsz_idx (Tensor):
        """
        assert bbsz_idx.numel() == eos_scores.numel()

        # clone relevant token and attention tensors.
        # tokens is (batch * beam, max_len). So the index_select
        # gets the newly EOS rows, then selects cols 1..{step + 2}
        tokens_clone = tokens.index_select(0, bbsz_idx)[
            :, 1 : step + 1
        ]  # skip the first index, which is EOS

        tokens_clone[:, step-1] = self.eos
        attn_clone = (
            attn.index_select(0, bbsz_idx)[:, :, 1 : step + 2]
            if attn is not None
            else None
        )
        selected_prev_steps= prev_steps[bbsz_idx]
        selected_prev_positions= prev_positions[bbsz_idx]

        # compute scores per token position
        pos_scores = scores.index_select(0, bbsz_idx)[:, 1: step + 1]
        pos_scores[:, step-1] = eos_scores
        # convert from cumulative to per-position scores
        pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

        # normalize sentence-level scores
        if self.normalize_scores:
            eos_scores /= (step + selected_prev_steps) ** self.len_penalty

        # cum_unfin records which sentences in the batch are finished.
        # It helps match indexing between (a) the original sentences
        # in the batch and (b) the current, possibly-reduced set of
        # sentences.
        cum_unfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cum_unfin.append(prev)

        # The keys here are of the form "{sent}_{unfin_idx}", where
        # "unfin_idx" is the index in the current (possibly reduced)
        # list of sentences, and "sent" is the index in the original,
        # unreduced batch
        # set() is not supported in script export
        sents_seen: Dict[str, Optional[Tensor]] = {}

        # For every finished beam item
        for i in range(bbsz_idx.size()[0]):
            idx = bbsz_idx[i]
            score = eos_scores[i]
            # sentence index in the current (possibly reduced) batch
            unfin_idx = idx // beam_size
            # sentence index in the original (unreduced) batch
            sent = unfin_idx + cum_unfin[unfin_idx]
            # Cannot create dict for key type '(int, int)' in torchscript.
            # The workaround is to cast int to string
            seen = str(sent.item()) + "_" + str(unfin_idx.item())
            if seen not in sents_seen:
                sents_seen[seen] = None
            # An input sentence (among those in a batch) is finished when
            # beam_size hypotheses have been collected for it
            if len(finalized[sent]) < beam_size:
                if attn_clone is not None:
                    # remove padding tokens from attn scores
                    hypo_attn = attn_clone[i]
                else:
                    hypo_attn = torch.empty(0)

                finalized[sent].append(
                    {
                        "tokens": tokens_clone[i],
                        "score": score,
                        "attention": hypo_attn,  # src_len x tgt_len
                        "alignment": torch.empty(0),
                        "positional_scores": pos_scores[i],
                        "prev_positions": selected_prev_positions[i]
                    }
                )

        newly_finished: List[int] = []

        for seen in sents_seen.keys():
            # check termination conditions for this sentence
            sent: int = int(float(seen.split("_")[0]))
            unfin_idx: int = int(float(seen.split("_")[1]))

            if not finished[sent] and self.is_finished(
                step, unfin_idx, max_len, len(finalized[sent]), beam_size
            ):
                finished[sent] = True
                newly_finished.append(unfin_idx)

        return newly_finished

    
    def _gen_one_stage(
        self,
        encoder_outs,
        incremental_states,
        prev_scores,
        prev_lengths,
        max_len,
        sample_id
    ):
        beam_size= self.beam_size
        bsz = len(sample_id)
        assert beam_size* bsz == len(prev_scores)

        # initialize buffers
        scores = (
            torch.zeros(bsz * beam_size, max_len +2).to(prev_scores).float()
        )  # +1 for eos; pad is never chosen for scoring
        scores[:,0] = prev_scores
        tokens = (
            torch.zeros(bsz * beam_size, max_len + 2)
            .to(prev_scores)
            .long()
            .fill_(self.pad)
        )  # +2 for eos and pad
        tokens[:, 0] = self.eos 

        prev_positions= torch.arange(beam_size).repeat(bsz, 1).to(prev_scores.device).view(-1)
       
        cands_to_ignore = (
            torch.zeros(bsz, beam_size).to(tokens).eq(-1)
        )  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(bsz)],
        )  # contains lists of dictionaries of infomation about the hypothesis being finalized at each step

        finished = [
            False for i in range(bsz)
        ]  # a boolean array indicating if the sentence at the index is finished or not
        num_remaining_sent = bsz  # number of sentences remaining

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens).to(prev_scores.device)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens).to(prev_scores.device)

        reorder_state: Optional[Tensor] = None
        batch_idxs: Optional[Tensor] = None

        original_batch_idxs: Optional[Tensor] = None
        original_batch_idxs = sample_id
        

        for step in range(1, max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(
                        batch_idxs
                    )
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size
                    )
                    original_batch_idxs = original_batch_idxs[batch_idxs]
                self.model.reorder_incremental_state(incremental_states, reorder_state)
                encoder_outs = self.model.reorder_encoder_out(
                    encoder_outs, reorder_state
                )

            lprobs, avg_attn_scores = self.model.forward_decoder(
                tokens[:, : step ],
                encoder_outs,
                incremental_states,
                self.temperature,
            )


            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1 :] = -math.inf

            
            if step < self.min_len+1:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            scores = scores.type_as(lprobs)
            eos_bbsz_idx = torch.empty(0).to(
                tokens
            )  # indices of hypothesis ending with eos (finished sentences)
            eos_scores = torch.empty(0).to(
                scores
            )  # scores of hypothesis ending with eos (finished sentences)


            # Shape: (batch, cand_size)
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
                tokens[:, : step + 1],
                original_batch_idxs,
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos
            # Shape of eos_mask: (batch size, beam size)
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][cands_to_ignore] = torch.tensor(0).to(eos_mask)

            # only consider eos when it's among the top beam_size indices
            # Now we know what beam item(s) to finish
            # Shape: 1d list of absolute-numbered
            eos_bbsz_idx = torch.masked_select(
                cand_bbsz_idx[:, :beam_size], mask=eos_mask[:, :beam_size]
            )

            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(
                    cand_scores[:, :beam_size], mask=eos_mask[:, :beam_size]
                )

                finalized_sents = self.finalize_hypos(
                    prev_lengths,
                    prev_positions,
                    step,
                    eos_bbsz_idx,
                    eos_scores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beam_size,
                    None,
                    max_len,
                )
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            if self.search.stop_on_max_len and step >= max_len:
                break
            assert step < max_len, f"{step} < {max_len}"

            # Remove finalized sentences (ones for which {beam_size}
            # finished hypotheses have been generated) from the batch.
            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = torch.ones(
                    bsz, dtype=torch.bool, device=cand_indices.device
                )
                batch_mask[finalized_sents] = False
                # TODO replace `nonzero(as_tuple=False)` after TorchScript supports it
                batch_idxs = torch.arange(
                    bsz, device=cand_indices.device
                ).masked_select(batch_mask)

                # Choose the subset of the hypothesized constraints that will continue
                self.search.prune_sentences(batch_idxs)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]

                # if prefix_tokens is not None:
                #     prefix_tokens = prefix_tokens[batch_idxs]
                # src_lengths = src_lengths[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]
                prev_lengths = prev_lengths.view(bsz,-1)[batch_idxs].view(-1)
                prev_positions = prev_positions.view(bsz,-1)[batch_idxs].view(-1)

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                # if attn is not None:
                #     attn = attn.view(bsz, -1)[batch_idxs].view(
                #         new_bsz * beam_size, attn.size(1), -1
                #     )
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos hypos
            # and values < cand_size indicate candidate active hypos.
            # After, the min values per row are the top candidate active hypos

            # Rewrite the operator since the element wise or is not supported in torchscript.

            eos_mask[:, :beam_size] = ~((~cands_to_ignore) & (~eos_mask[:, :beam_size]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )

            # get the top beam_size active hypotheses, which are just
            # the hypos with the smallest values in active_mask.
            # {active_hypos} indicates which {beam_size} hypotheses
            # from the list of {2 * beam_size} candidates were
            # selected. Shapes: (batch size, beam size)
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=beam_size, dim=1, largest=False
            )

            # update cands_to_ignore to ignore any finalized hypos.
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :beam_size]
            # Make sure there is at least one active item for each sentence in the batch.
            assert (~cands_to_ignore).any(dim=1).all()

            # update cands_to_ignore to ignore any finalized hypos

            # {active_bbsz_idx} denotes which beam number is continued for each new hypothesis (a beam
            # can be selected more than once).
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses

            # Set the tokens for each beam (can select the same row more than once)
            tokens[:, : step ] = torch.index_select(
                tokens[:, : step ], dim=0, index=active_bbsz_idx
            )
            # Select the next token for each of them
            tokens.view(bsz, beam_size, -1)[:, :, step ] = torch.gather(
                cand_indices, dim=1, index=active_hypos
            )
            prev_lengths = prev_lengths[active_bbsz_idx]
            prev_positions = prev_positions[active_bbsz_idx]
            scores[:, :step] = torch.index_select(
                scores[:, :step], dim=0, index=active_bbsz_idx
            )
            scores.view(bsz, beam_size, -1)[:, :, step] = torch.gather(
                cand_scores, dim=1, index=active_hypos
            )

            # Update constraints based on which candidates were selected for the next beam
            self.search.update_constraints(active_hypos)

            # copy attention for active hypotheses
            # if attn is not None:
            #     attn[:, :, : step + 2] = torch.index_select(
            #         attn[:, :, : step + 2], dim=0, index=active_bbsz_idx
            #     )

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            scores = torch.tensor(
                [float(elem["score"].item()) for elem in finalized[sent]]
            )
            _, sorted_scores_indices = torch.sort(scores, descending=True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sorted_scores_indices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, Tensor]], finalized[sent]
            )
        return finalized

    
    def _generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        constraints: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        incremental_states = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {})
                for i in range(self.model.models_size)
            ],
        )
        net_input = sample["net_input"]
        src_tokens = net_input["fbank"]
        src_lengths= net_input['fbk_lengths']

        # bsz: total number of sentences in beam
        # Note that src_tokens may have more than 2 dimenions (i.e. audio features)
        bsz, src_len = src_tokens.size()[:2]
        beam_size = self.beam_size

        # Initialize constraints, when active
        self.search.init_constraints(constraints, beam_size)
        max_len = min(
            int(self.max_len_a * src_len + self.max_len_b),
            # exclude the EOS marker
            self.model.max_decoder_positions() - 1,
        )
        assert (
            self.min_len <= max_len
        ), "min_len cannot be larger than max_len, please adjust these!"
        # compute the encoder output for each beam
        encoder_outs = self.model.forward_encoder(net_input)

        # placeholder of indices for bsz * beam_size to hold tokens and accumulative scores
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = self.model.reorder_encoder_out(encoder_outs, new_order)
        prev_scores= torch.zeros(bsz * beam_size).to(src_tokens).float()
        prev_scores.view(bsz,beam_size)[:,1:] = -math.inf
        prev_lengths= torch.zeros(bsz*beam_size).to(src_tokens.device).float()
        # ensure encoder_outs is a List.
        assert encoder_outs is not None
        original_batch_idxs: Optional[Tensor] = None
        if "id" in sample and isinstance(sample["id"], Tensor):
            original_batch_idxs = sample["id"]
        else:
            original_batch_idxs = torch.arange(0, bsz).to(src_tokens.device).float()
        self.len_penalty =1.0
        hypos_asr = self._gen_one_stage(
            encoder_outs,
            incremental_states,
            prev_scores,
            prev_lengths,
            max_len,
            original_batch_idxs
        )
        asr_tokens, prev_scores, prev_lengths = self.extract_hypos(hypos_asr)
        if self.asr_1best:
            prev_scores.view(bsz,beam_size)[:,1:] = -math.inf
            prev_scores.view(bsz,beam_size)[:,0] = 0
        self.len_penalty =2
        self.model.init_for_second_decoding(encoder_outs, asr_tokens, incremental_states)
       
        hypos_mt = self._gen_one_stage(
            encoder_outs,
            incremental_states,
            prev_scores,
            prev_lengths,
            max_len,
            original_batch_idxs
        )
        hypos=[]
        for i in range(bsz):
            asrs = hypos_asr[i]
            mts = hypos_mt[i]
            results=[]
            for j in range(len(mts)):
                asr_pos = mts[j]["prev_positions"]
                rlt= {
                    "mt_tokens": mts[j]["tokens"],
                    "mt_score": mts[j]["score"],
                    "mt_positional_scores": mts[j]['positional_scores'],
                    "asr_tokens":asrs[asr_pos]["tokens"],
                    "asr_score":asrs[asr_pos]["score"]
                }
                results.append(rlt)
            hypos.append(results)
        return hypos

    
    def extract_hypos(self, hypos_list):
        """
        {
            "tokens": tokens_clone[i],
            "score": score,
            "attention": hypo_attn,  # src_len x tgt_len
            "alignment": torch.empty(0),
            "positional_scores": pos_scores[i],
            "prev_positions": selected_prev_positions[i]
        }
        """
        bsz = len(hypos_list)
        beam_size= self.beam_size
        pad= self.pad
        all_tokens, all_scores= [],[]
        for i in range(bsz):
            hypos= hypos_list[i]
            tokens= [h["tokens"] for h in hypos]
            scores= [h['score']*len(h['tokens']) for h in hypos]
            #scores= [h['score'] for h in hypos]
            if len(tokens)< beam_size:
                to_add = beam_size - len(tokens)
                tokens.extend([tokens[-1]]*to_add)
                scores.extend([scores[-1]]*to_add)
            tokens= tokens[:beam_size]
            scores = scores[:beam_size]
            all_tokens.extend(tokens)
            all_scores.extend(scores)
        prev_lengths = [len(t) for t in all_tokens]
        max_len = max(prev_lengths)
        bbsz= len(all_tokens)
        src_tokens=all_tokens[0].new(bbsz, max_len).fill_(pad)
        for i, token in enumerate(all_tokens):
            src_tokens[i,:prev_lengths[i]] = token
        prev_scores= torch.stack(all_scores)
        
        prev_lengths = torch.LongTensor(prev_lengths).to(prev_scores.device)
        #prev_lengths.fill_(0)
        return src_tokens, prev_scores, prev_lengths



        


class JointEnsembel(EnsembleModel):
    def __init__(self, models):
        super().__init__(models)
        self.curr_stage = 0
    
    @torch.jit.export
    def forward_encoder(self, net_input: Dict[str, Tensor]):
        self.curr_stage = 0
        if not self.has_encoder():
            return None
        return [model.encoder.forward(**net_input) for model in self.models]
    
    def init_for_second_decoding(self, encoder_outs, prev_source, incremental_states):
        self.curr_stage = 1
        for i, model in enumerate(self.models):
            encoder_outs[i], incremental_states[i] = model.init_for_second_decoding(
                encoder_outs[i], prev_source, incremental_states[i]
            )
    
    @torch.jit.export
    def reorder_incremental_state(
        self,
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        new_order,
    ):
        if not self.has_incremental_states():
            return
        for i, model in enumerate(self.models):
            if self.curr_stage == 1 and hasattr(model, "decoder2"):
                model.decoder2.reorder_incremental_state_scripting(
                    incremental_states[i], new_order
                )
            else:
                model.decoder.reorder_incremental_state_scripting(
                    incremental_states[i], new_order
                )

    def forward_decoder(
        self,
        tokens,
        encoder_outs: List[Dict[str, List[Tensor]]],
        incremental_states: List[Dict[str, Dict[str, Optional[Tensor]]]],
        temperature: float = 1.0,
    ):
        assert self.has_incremental_states(), "stage decoding needs incremental states"
        log_probs = []
        avg_attn: Optional[Tensor] = None
        encoder_out: Optional[Dict[str, List[Tensor]]] = None
        for i, model in enumerate(self.models):
            if self.has_encoder():
                encoder_out = encoder_outs[i]
            # decode each model
            if self.curr_stage == 0:
                decoder_out = model.decode1(tokens, encoder_out, incremental_states[i])
            else:
                decoder_out= model.decode2(tokens,encoder_out, incremental_states[i])
            decoder_len = len(decoder_out)
            decoder_out_tuple = (
                decoder_out[0][:, -1:, :].div_(temperature),
                None if decoder_len <= 1 else decoder_out[1],
            )
            
            probs = model.get_normalized_probs(
                decoder_out_tuple, log_probs=True, sample=None
            )
            probs = probs[:, -1, :]
            if self.models_size == 1:
                return probs, None
            log_probs.append(probs)

        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(
            self.models_size
        )

        return avg_probs, None

    
