import itertools

import torch
from transformers import BertTokenizer, BertForSequenceClassification


class PredictEntailment:
    """
    Entailment task predictor
    """

    def __init__(self, model_or_path, tokenisor_or_path, do_lower_case=False, max_length=512, batch_size=8):

        self.batch_size = batch_size
        self.device = "cuda" if torch.has_cuda else "cpu"
        self.max_length = max_length

        if isinstance(model_or_path, str):
            model_or_path = BertForSequenceClassification.from_pretrained(model_or_path, output_attentions=False)
        self._model = model_or_path

        if isinstance(tokenisor_or_path, str):
            tokenisor_or_path = BertTokenizer.from_pretrained(tokenisor_or_path, do_lower_case=do_lower_case)
        self._tokenizer = tokenisor_or_path

    def __call__(self, sentences_a, sentences_b, labels=None, transform_result_func=None):
        transform_result_func = transform_result_func
        result_pred_prob_batches = []
        result_gt_batches = []
        result_inputs = []
        labels = labels or [0] * len(sentences_a)

        self._model.to(self.device)

        for (batch_a, batch_b, batch_label) in zip(self._chunk(sentences_a, self.batch_size),
                                                   self._chunk(sentences_b, self.batch_size),
                                                   self._chunk(labels, self.batch_size)):

            inputs = self._tokenizer(batch_a, batch_b, return_tensors='pt', padding=True, truncation=True,
                                     max_length=512)

            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            classification_logits = self._model(**inputs)[0]

            pred_prob = torch.softmax(classification_logits, dim=1)

            result_pred_prob_batches.append(pred_prob)
            result_gt_batches.append(batch_label)
            result_inputs.append([sentences_a, sentences_b])

        if transform_result_func:
            transform_result_func(result_pred_prob_batches, result_inputs, result_gt_batches)

        return torch.cat(result_pred_prob_batches, dim=0), list(itertools.chain(*result_gt_batches))

    def _chunk(self, data, chunk_size):
        accumulator = []
        for i in data:
            if (len(accumulator) + 1) % chunk_size == 0:
                yield accumulator
                accumulator = []
            else:
                accumulator.append(i)

        # Final remaining partial chunk
        if len(accumulator) > 0:
            print(accumulator)
            yield accumulator
