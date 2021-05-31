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

# import multiprocessing as mp
import sys
import os
import gc  # garbage collector
import torch
import time
import argparse  # command-line parsing library
import json
import string
import re  # regular expressions
import textstat  # calculates statistics from text such as reading level
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import torch.cuda.profiler as profiler
# import pyprof


def printlog(s):
    print(s)
    if logFile is not None:
        print(s, file=logFile)


def printlogFile(s):
    print(s)
    if logFile is not None:
        print(s, file=logFile)


def printProcessInfo(title):
    printlog(title)
    printlog(f"module name: {__name__}")
    printlog(f"parent process: {os.getppid()}")
    printlog(f"process id: {os.getpid()}")


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


# From https://goshippo.com/blog/measure-real-size-any-python-object/
# Recursively finds size of objects
def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes,
                                                           bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


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


def getTextStats(law, skip):
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
    return i, skip, startTime


def getAnswers(lawFileName, batchSize, skip, first):
    # global logFile  # output file for printlog(s)
    # logFile = open(logFileName, "a")
    # printProcessInfo("function getAnswers")
    printlogFile("AutoModelForQuestionAnswering with ALBERT xLarge "
                 "pretrained on SQuAD2.0 by ktrapeznikov")
    printlogFile("Reading CA Penal Code Q&A JSON sorted by section "
                 "including unanswerables")

    lawFile = open(lawFileName, "r")
    law = json.load(lawFile)  # returns a dictionary
    lawFile.close()
    # lawSize = get_size(law)
    # printlogFile(f"law size: {lawSize} bytes")

    # Can we use Cuda to run faster on a GPU or just use the slower CPU?
    device = "cuda" if torch.cuda.is_available() else "cpu"  # [2] line 22
    printlogFile(f"           running on: {device}")
    printMemory()

    # record start time
    if device == "cuda":
        torch.cuda.synchronize()
    tokenizer = AutoTokenizer.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2")
    model = AutoModelForQuestionAnswering.from_pretrained(
            "ktrapeznikov/albert-xlarge-v2-squad-v2")
    # modelSize = get_size(model)
    # printlogFile(f" model size CPU: {modelSize} bytes")
    model.to(device)        # run on GPU if available
    printMemory()
    noAnswerCount = 0       # number of questions that have no answer
    noAnswerCorrect = 0     # number of "no answer" questions correct
    exactMatch = 0
    totalF1 = 0.0
    q = 1
    batchNum = 0
    # record start time
    if device == "cuda":
        torch.cuda.synchronize()
    startTime = int(round(time.time() * 1000))  # time in ms
    for i in range(len(law['data'])):
        if i < skip:
            startTime = int(round(time.time() * 1000))  # time in ms
            continue  # skip first --skip=n questions for validating new
        if q > first:
            q -= 1
            break
        entry = law['data'][i]
        question = entry['question']
        # questions = [question['text'] for question
        #             in entry['questions'] if question['text']]
        if question == "QQQ":  # skip non-existant questions
            skip += 1
            continue
        # print(questions)
        id = entry['id']
        if id != i:
            printlogFile(f"WARNING: ? {i} JSON ID {id} mismatch.")
        context = entry['context']
        if batchNum == 0:
            question_context_for_batch = []
            answer_for_batch = []
        batchNum += 1
        question_context_for_batch.append((question, context))
        goldAnswers = [answer['text'] for answer
                       in law['data'][i]['answers'] if answer['text']]
        if not goldAnswers:
            goldAnswers = ['']
        answer_for_batch.append((goldAnswers))
        if (batchNum < batchSize):
            continue
        else:
            batchNum = 0
        # print(question_context_for_batch)
        # print(answer_for_batch)
        encoding = tokenizer.batch_encode_plus(question_context_for_batch,
                                               padding="longest",
                                               return_tensors="pt")
        input_ids = encoding["input_ids"]
        # attention_mask = encoding["attention_mask"]
        # input_ids.to(device)
        # attention_mask.to(device)
        encoding.to(device)
        # start_scores, end_scores = model(**encoding)
        """
        tokenCount = input_dict['input_ids'].size(1)
        if tokenCount > 512:
            skip += 1
            printlogFile(f"? {i} ERROR: context longer than 512 tokens "
                         f"({tokenCount}) - skipped: {context}")
            continue
        printMemory()
        """
        modelOutput = model(**encoding)
        start_scores = modelOutput.start_logits
        end_scores = modelOutput.end_logits

        for b in range(batchSize):
            max_startscore = torch.argmax(start_scores[b])
            max_endscore = torch.argmax(end_scores[b])
            ans_tokens = input_ids[b][max_startscore: max_endscore + 1]
            answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens,
                                                    skip_special_tokens=True)
            AlbertAnswer = tokenizer.convert_tokens_to_string(answer_tokens)
            """
            # AlbertAnswer = ''.join(all_tokens[torch.argmax(start_scores[b]):
            #                                  torch.argmax(end_scores[b])
            #                                  + 1]).replace('▁', ' ').strip()
            if AlbertAnswer == "[CLS]":
                AlbertAnswer = ""
            elif AlbertAnswer[0:5] == "[CLS]":
                sep = AlbertAnswer.find("[SEP]")
                if sep > 4:  # remove [CLS] ... [SEP]
                    AlbertAnswer = AlbertAnswer[(sep + 5):].strip()
            """
            em = max((compute_exact_match(AlbertAnswer, answer)) for answer
                     in answer_for_batch[b])
            if em == 1:
                exactMatch += 1
                f1 = 1.0  # F1 always equals 1.0 for an exact match
                totalF1 += f1
                if answer_for_batch[b] == ['']:  # no answer was correct
                    noAnswerCount += 1
                    noAnswerCorrect += 1
                printlogFile(f"EM ? {q}: {question_context_for_batch[b][0]}")
                printlogFile(f"ALBERT: {AlbertAnswer}")
            else:  # print F1, golden answers and context when not an EM
                f1 = max((compute_f1(AlbertAnswer, answer))
                         for answer in answer_for_batch[b])
                totalF1 += f1
                if answer_for_batch[b] == ['']:  # no answer was correct
                    noAnswerCount += 1
                printlogFile(f"F1 {f1} ? {q}: "
                             f"{question_context_for_batch[b][0]}")
                printlogFile(f"ALBERT: {AlbertAnswer}")
                printlogFile(f"answer: {answer_for_batch[b]}")
                printlogFile(f"context: {question_context_for_batch[b][1]}")
            q += 1
        """
        printlog("after model(**encoding)")
        printMemory()
        del encoding
        printlog("after del encoding")
        printMemory()
        del input_ids, start_scores, end_scores
        del modelOutput
        printlog("after del modelOutput")
        printMemory()
        gc.collect()
        printlog("after gc.collect()")
        printMemory()
        # torch.cuda.synchronize(device)
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.empty_cache()
        printlog("after clearing cache")
        printMemory()
        # printMemory()
        # with torch.no_grad():
        #    if device == "cuda":
        #        torch.cuda.empty_cache()
        # printlog("after clearing cache")
        # printMemory()
        """
    elapsedTime = int(round(time.time() * 1000)) - startTime
    # i -= skip
    # i += 1  # first ? is index 0
    printlogFile("")
    printlogFile(f"{exactMatch}/{q} exact matches = "
                 f"{exactMatch / q * 100:.2f}%")
    printlogFile(f"F1 = {totalF1 / q * 100:.2f}%")
    if (noAnswerCount > 0):
        printlogFile(f"{noAnswerCorrect}/{noAnswerCount} "
                     "correct \"no answers\" = "
                     f"{noAnswerCorrect / noAnswerCount * 100:.2f}%")
    if (q - noAnswerCount) > 0:
        printlogFile(f"{exactMatch - noAnswerCorrect}/{q - noAnswerCount} "
                     "\"has answer\" exact matches = "
                     f"{(exactMatch - noAnswerCorrect) / (q - noAnswerCount) * 100:.2f}%")
    printlogFile(f"elapsed answering time: {elapsedTime} ms")
    printlogFile(f"{elapsedTime / q / 1000:.4f} seconds per question")
    printMemory()
    # if logFile is not None:
    #    logFile.close()
    # return (len(law['data']) - 1), skip, noAnswerCount, noAnswerCorrect, \
    #       exactMatch, totalF1, startTime
    return


def main():
    # argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", type=int, default=1)
    ap.add_argument("-f", "--first", type=int, default=300)
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
    # printlog(f"           running on: {device}")
    # printMemory()

    # record start time
    # if device == "cuda":
    #    torch.cuda.synchronize()
    startTime = int(round(time.time() * 1000))  # time in ms

    batchSize = args['batch']
    printlog(f"           batch size: {batchSize} question(s)")
    skip = args['skip']
    if skip > 0:
        printlog(f"   skipping the first: {skip} questions")
    first = args['first']
    printlog(f"  answering the first: {first} question(s)")

    if args['textstats']:  # calculate text statistics like reading levels
        i, skip, startTime = getTextStats(law, skip)
    else:
        getAnswers("CalPenalCodeQA.json", batchSize, skip, first)

    # if device == "cuda":
    #    torch.cuda.synchronize()

    elapsedTime = int(round(time.time() * 1000)) - startTime

    if args['textstats']:
        elapsedTime = int(round(time.time() * 1000)) - startTime
        printlog(f"elapsed text statistical analysis time {elapsedTime} ms")
        printlog(f"{elapsedTime / i / 1000:.4f} seconds per question")

    printMemory()
    if logFile is not None:
        logFile.close()

    with torch.no_grad():
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
