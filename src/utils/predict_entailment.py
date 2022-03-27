import csv
import itertools

import numpy
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
        # Set model in eval mode
        self._model.eval()

        for (batch_a, batch_b, batch_label) in zip(self._chunk(sentences_a, self.batch_size),
                                                   self._chunk(sentences_b, self.batch_size),
                                                   self._chunk(labels, self.batch_size)):

            inputs = self._tokenizer(batch_a, batch_b, return_tensors='pt', padding=True, truncation=True,
                                     max_length=512)

            for k, v in inputs.items():
                inputs[k] = v.to(self.device)

            with torch.no_grad():
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

    def write_to_file(self, output_file, pred_prob_batched, inputs_batched, result_gt_batched, labels_order=None):

        def label_indexer(x):
            if labels_order:
                return labels_order.index(x)
            return x

        with open(output_file, "w") as f:
            csv_f = csv.writer(f, delimiter='\t', quotechar='"')
            csv_f.writerow(["confidence", "pred_index", "sentence_a", "sentence_b", "label_index", "label_name"])
            for pred_prob, inputs, labels in zip(pred_prob_batched, inputs_batched, result_gt_batched):
                rows = [[max(p.cpu().numpy()), numpy.argmax(p.cpu().numpy()), i[0], i[1], label_indexer(l), l]
                        for (p, i, l) in zip(pred_prob, inputs, labels)]
                csv_f.writerows(rows)
