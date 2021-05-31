# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 14:35:18 2021

@author: Sterling Engle
@uid: 904341227

Developed and tested only on Windows 10 under Python 3.8.5.

penalcode.py is a PyTorch implementation using ALBERT NLP pre-trained
in SQUAD 2.0 to answer questions: "ktrapeznikov/albert-xlarge-v2-squad-v2"
to investigate machine reading comprehension of the California Penal Code.

These parameters may be specified via command line arguments:

usage: penalcode.py [-h] [-j JSON] [-o OUTPUT] [-s SKIP]

optional arguments:
  -h, --help            show this help message and exit
  -j JSON, --json JSON  JSON file path
  -o OUTPUT, --output OUTPUT
                        Q&A log output file path
  -s SKIP, --skip SKIP  number of questions to skip

References:
[1] The Hugging Face Team, "transformers / examples / question-answering",
    https://github.com/huggingface/transformers/tree/master/examples/question-answering
[2] Cloudera Fast Forward Labs, "Evaluating QA: Metrics, Predictions, and
    the Null Response",
    https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html#F1
[2] Persson, Alladin, "Useful Tensor Operations",
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_tensorbasics.py
[3] PyTorch.org, "PyTorch Documentation",
    https://pytorch.org/docs/stable/index.html
[4] Stack Overflow, "Get total amount of free GPU memory and available using
    pytorch",
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
"""

import torch
import time
import argparse  # command-line parsing library
import json
import string
import re  # regular expressions
import textstat  # calculates statistics from text such as reading level
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch.cuda.profiler as profiler
import pyprof


def printlog(s):
    print(s)
    if logFile is not None:
        print(s, file=logFile)


# prints GPU memory usage and availability [4]
def printMemory():
    if torch.cuda.is_available():
        totalGPU = torch.cuda.get_device_properties(0).total_memory
        reservedGPU = torch.cuda.memory_reserved(0)
        allocatedGPU = torch.cuda.memory_allocated(0)
        freeGPU = reservedGPU - allocatedGPU  # free inside reserved
        printlog(f"     total GPU memory: {totalGPU}")
        printlog(f"  reserved GPU memory: {reservedGPU}")
        printlog(f" allocated GPU memory: {allocatedGPU}")
        printlog(f"      free GPU memory: {freeGPU}")


# these functions are based upon the HF squad_metrics.py script
def normalize_text(s):
    # Removing articles and punctuation, and standardizing whitespace are
    # all typical text processing steps.

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    # if either the prediction or the truth is no-answer then f1 = 1
    # if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)

    common_tokens = set(pred_tokens) & set(truth_tokens)

    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0.0

    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)

    return 2.0 * (prec * rec) / (prec + rec)


def get_gold_answers(example):
    # helper function that retrieves all possible true answers from a squad2.0
    # example

    gold_answers = [answer["text"] for answer in example.answers
                    if answer["text"]]

    # if gold_answers doesn't exist it's because this is a negative example -
    # the only correct answer is an empty string
    if not gold_answers:
        gold_answers = [""]

    return gold_answers


def main():
    # argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-j", "--json", type=open, default='CalPenalCodeQA.json',
                    help="JSON file path")
    ap.add_argument("-o", "--output", type=argparse.FileType('a'),
                    help="Q&A log output file path")
    ap.add_argument("-p", "--pyprof", action='store_true',
                    help="run pyprof profiling")
    ap.add_argument("-s", "--skip", type=int, default=0,
                    help="number of questions to skip")
    ap.add_argument("-t", "--textstats", action='store_true',
                    help="outputs text statistics instead of Q&A")
    args = vars(ap.parse_args())
    global logFile  # output file for printlog(s)
    logFile = args['output']  # optional output file used by printlog(s)
    printlog("AutoModelForQuestionAnswering with ALBERT xLarge pretrained "
             "on SQuAD2.0 by ktrapeznikov")
    printlog("Reading CA Penal Code Q&A JSON sorted by section "
             "including unanswerables")
    law = json.load(args['json'])  # returns a dictionary
    args['json'].close()

    # Can we use Cuda to run faster on a GPU or just use the slower CPU?
    device = "cuda" if torch.cuda.is_available() else "cpu"  # [2] line 22
    printlog(f"           running on: {device}")
    printMemory()

    # record start time
    if device == "cuda":
        torch.cuda.synchronize()
    startTime = int(round(time.time() * 1000))  # time in ms

    skip = args['skip']
    if skip > 0:
        printlog(f"   skipping the first: {skip} questions")

    if args['textstats']:  # calculate text statistics like reading levels
        sentences = 0
        syllables = 0
        fleschReadingEase = 0.0
        fleschGradeLevel = 0.0
        gunningFOG = 0.0
        autoReadability = 0.0
        colemanLiau = 0.0
        linsearWrite = 0.0
        daleChall = 0.0
        # textStandard = 0.0

        for i in range(len(law['data'])):
            if i < skip:
                continue  # skip first --skip=n questions for validating new
            entry = law['data'][i]
            question = entry['question']
            if question == "QQQ":  # skip non-existant questions
                skip += 1
                continue
            id = entry['id']
            if id != i:
                printlog(f"WARNING: ? {i} JSON ID {id} mismatch.")
            context = entry['context']
            sentences += textstat.sentence_count(context)
            syllables += textstat.syllable_count(context)
            fleschReadingEase += textstat.flesch_reading_ease(context)
            fleschGradeLevel += textstat.flesch_kincaid_grade(context)
            gunningFOG += textstat.gunning_fog(context)
            autoReadability += textstat.automated_readability_index(context)
            colemanLiau += textstat.coleman_liau_index(context)
            linsearWrite += textstat.linsear_write_formula(context)
            daleChall += textstat.dale_chall_readability_score(context)
            # textStandard += textstat.text_standard(context)
        i -= skip
        i += 1  # first ? is index 0
        printlog("")
        printlog(f"       CalPenalCodeQA v0.1 contains: {sentences} "
                 f"sentences and {syllables} syllables")
        printlog("                Flesch Reading Ease: "
                 f"{fleschReadingEase / i:.1f} (Very Confusing)")
        printlog("         Flesch-Kincaid Grade Level: "
                 f"{fleschGradeLevel / i:.1f}")
        printlog("            Gunning FOG Grade Level: "
                 f"{gunningFOG / i:.1f}")
        printlog("  Automated Readability Grade Level: "
                 f"{autoReadability / i:.1f}")
        printlog("           Coleman-Liau Grade Level: "
                 f"{colemanLiau / i:.1f}")
        printlog("          Linsear Write Grade Level: "
                 f"{linsearWrite / i:.1f}")
        printlog(f"             Dale-Chall Grade Level: {daleChall / i:.1f}")
        meanGradeLevel = (fleschGradeLevel + gunningFOG + autoReadability +
                          colemanLiau + linsearWrite + daleChall) / 6.0 / i
        printlog("Six-metric mean reading grade level: "
                 f"{meanGradeLevel:.1f}")
        printlog("")
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2")
        model = AutoModelForQuestionAnswering.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2")
        model.to(device)        # run on GPU if available
        noAnswerCount = 0       # number of questions that have no answer
        noAnswerCorrect = 0     # number of "no answer" questions correct
        exactMatch = 0
        totalF1 = 0.0

        if args['pyprof']:
            printlog("enabling pyprof profiling")
            pyprof.init()  # initializes PyProf
            profiler.start()
        startTime = int(round(time.time() * 1000))  # time in ms
        for i in range(len(law['data'])):
            if i < skip:
                startTime = int(round(time.time() * 1000))  # time in ms
                continue  # skip first --skip=n questions for validating new
            entry = law['data'][i]
            question = entry['question']
            if question == "QQQ":  # skip non-existant questions
                skip += 1
                continue
            id = entry['id']
            if id != i:
                printlog(f"WARNING: ? {i} JSON ID {id} mismatch.")

            context = entry['context']
            input_dict = tokenizer.encode_plus(question, context,
                                               return_tensors="pt")
            tokenCount = input_dict['input_ids'].size(1)
            if tokenCount > 512:
                skip += 1
                printlog(f"? {i} ERROR: context longer than 512 tokens "
                         f"({tokenCount}) - skipped: {context}")
                continue
            input_dict.to(device)   # run on GPU if available
            modelOutput = model(**input_dict)
            start_scores = modelOutput.start_logits
            end_scores = modelOutput.end_logits
            input_ids = input_dict["input_ids"].tolist()
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            AlbertAnswer = ''.join(all_tokens[torch.argmax(start_scores):
                                              torch.argmax(end_scores)
                                              + 1]).replace('▁', ' ').strip()
            if AlbertAnswer == "[CLS]":
                AlbertAnswer = ""
            elif AlbertAnswer[0:5] == "[CLS]":
                sep = AlbertAnswer.find("[SEP]")
                if sep > 4:  # remove [CLS] ... [SEP]
                    AlbertAnswer = AlbertAnswer[(sep + 5):].strip()
            goldAnswers = [answer['text'] for answer
                           in law['data'][i]['answers'] if answer['text']]
            if not goldAnswers:
                goldAnswers = ['']

            em = max((compute_exact_match(AlbertAnswer, answer)) for answer
                     in goldAnswers)
            if em == 1:
                exactMatch += 1
                f1 = 1.0  # F1 always equals 1.0 for an exact match
                totalF1 += f1
                if goldAnswers == ['']:  # no answer was the correct answer
                    noAnswerCount += 1
                    noAnswerCorrect += 1
                printlog(f"EM ? {i}: {question}")
                printlog(f"ALBERT: {AlbertAnswer}")
                # printlog(f"answer: {goldAnswers}")
                # printlog(f"context: {context}")
            else:  # print F1, golden answers and context when not an EM
                f1 = max((compute_f1(AlbertAnswer, answer))
                         for answer in goldAnswers)
                totalF1 += f1
                if goldAnswers == ['']:  # no answer was the correct answer
                    noAnswerCount += 1
                printlog(f"F1 {f1} ? {i}: {question}")
                printlog(f"ALBERT: {AlbertAnswer}")
                printlog(f"answer: {goldAnswers}")
                printlog(f"context: {context}")

        if args['pyprof']:
            profiler.stop()
        i -= skip
        i += 1  # first ? is index 0
        printlog("")
        printlog(f"{exactMatch}/{i} exact matches = "
                 "{exactMatch / i * 100:.2f}%")
        printlog(f"F1 = {totalF1 / i * 100:.2f}%")
        printlog(f"{noAnswerCorrect}/{noAnswerCount} correct \"no answers\" = "
                 f"{noAnswerCorrect / noAnswerCount * 100:.2f}%")
        if (i - noAnswerCount) > 0:
            printlog(f"{exactMatch - noAnswerCorrect}/{i - noAnswerCount} "
                     "\"has answer\" exact matches = "
                     f"{(exactMatch - noAnswerCorrect) / (i - noAnswerCount) * 100:.2f}%")

    if device == "cuda":
        torch.cuda.synchronize()
    elapsedTime = int(round(time.time() * 1000)) - startTime
    if args['textstats']:
        printlog(f"elapsed text statistical analysis time {elapsedTime} ms")
        printlog(f"{elapsedTime / i / 1000:.3f} seconds per question")
    else:
        printlog(f"elapsed answering time: {elapsedTime} ms")
        printlog(f"{elapsedTime / i / 1000:.3f} seconds per question")
    printMemory()
    if logFile is not None:
        logFile.close()

    with torch.no_grad():
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
