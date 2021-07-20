import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss
)

@register_criterion("fake_loss")
class FakeCriterion(FairseqCriterion):
    def __init__(self,task):
        super().__init__(task)
        
    
    def forward(self, loss_info,sample,model):
        """
         {"loss":losses[0], "loss_prob":losses[1], "loss_delay":losses[2], "loss_me":losses[3], "sample_size": B}
        """
        if "ntokens" not in sample:
            sample["ntokens"]= model.get_ntokens(sample)
        sample_size= loss_info["sample_size"]
        loss= loss_info["loss"]
        loss_prob= loss_info["loss_prob"]
        loss_delay= loss_info["loss_delay"]
        #loss_me= loss_info["loss_me"]
        nll_loss = loss_info["nll_loss"]
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "prob_loss": loss_prob.data,
            "delay_loss":loss_delay.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if "dec2_nll_loss" in loss_info:
            logging_output['dec2_nll_loss'] = loss_info['dec2_nll_loss'].data
            
        return loss, sample_size, logging_output
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        prob_loss_sum = sum(log.get("prob_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        delay_loss_sum= sum(log.get("delay_loss", 0) for log in logging_outputs)
        # loss_me_sum = sum(log.get("me_loss",0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "delay_loss", delay_loss_sum / sample_size , sample_size, round=3
        )
        # transducer nll loss should normalized by token number
        metrics.log_scalar(
            "prob_loss", prob_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        # metrics.log_scalar(
        #     "me_loss", loss_me_sum/sample_size, sample_size, round=3
        # )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        if "dec2_nll_loss" in logging_outputs[0]:
            dec2_loss_sum = sum(log.get("dec2_nll_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "dec2_nll_loss", dec2_loss_sum / ntokens / math.log(2), ntokens, round=3
            )
        
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True