# CalPenalCodeQA: A machine comprehension dataset

Course: Machine Learning Algorithms (CS 260) - Professor: Dr. Cho-Jui Hsieh
Due date: March 17th, 2021 11:59 PM

Introduction

Machine Reading Comprehension (MRC) is a very active area of research in the Natural Language
Processing (NLP) branch of AI. The Stanford Question Answering Dataset (SQuAD2.0) has proven
very useful for comparing the performance of various MRC algorithms [8].

SQuAD2.0 combines 100,000 questions in SQuAD1.1 with over 50,000 unanswerable ones
written adversarially by crowdworkers to look similar to answerable ones. To do well on
SQuAD2.0, systems must not only answer questions when possible, but also determine
when no answer is supported by the paragraph and abstain from answering. ([7])

These questions are about speci?c paragraphs drawn from popular Wikipedia articles. The goal of
this investigation is to determine how well a model pre-trained on SQuAD2.0 performs against a
domain-speci?c `real world' dataset such as the California Penal Code it has never seen before. The
penal code was chosen because if arti?cial superintelligence (greater than humans) ever becomes a
reality, it will need to know and abide by our criminal laws.

The algorithm ?ALBERT: A Lite BERT for Self-supervised Learning of Language Representations?
[4] has performed well on SQuAD2.0. SA-Net on ALBERT (ensemble) was the top-ranked model
from April 6, 2020 until February 21, 2021 on the SQuAD2.0 Leaderboard [7] with an exact match
(EM) of 90.724% and F1 [10] score of 93.011. For this study, albert-xlarge-v2-squad-v2, an
albert-xlarge-v2 model [2] trained against SQuAD2.0 for question answering using 4 NVIDIA
GeForce RTX 2080 Ti 11Gb GPUs by Kirill Trapeznikov [9] was used. albert-xlarge-v2 has ?24
repeating layers, 128 embedding dimensions, 2048 hidden dimensions, 16 attention heads, and 58M
parameters? [2]. Trapeznikov's pre-trained model scored 84.4 EM and 87.5 F1 on SQuAD2.0.

California Penal Code dataset creation

The California Penal Code contains 1,346,842 words organized into 5,596 sections. The Code was
obtained by `scraping' it from the California Legislative Information website [5]. The Scrapy Python
package [11] was used by a custom spider script to download it to a JSON ?le. Scrapy locates HTML
tags identi?ed in the custom spider script that encapsulate desired information, such as between the
paragraph <p></p> tag pairs following this division <div> tag:
<div id="display_code_many_law_sections" class="displaycodeleftmargin">
<p style="margin:0;display:inline;">This Act shall be known as The Penal Code&nbsp;of California,
and is divided into four parts, as follows: <br></p>

The Scrapy spider script output the Penal Code to a simple JSON ?le with code section and law
text ?elds corresponding to each paragraph of the code as seen in Figure 2:

{"sec": "1", "law": "Section 1. This Act shall be known as The Penal Code of California, and is
divided into four parts, as follows: I.-OF CRIMES AND PUNISHMENTS. II.-OF CRIMINAL PROCEDURE.
III.-OF THE STATE PRISON AND COUNTY JAILS. IV.-OF PREVENTION OF CRIMES AND APPREHENSION OF
CRIMINALS."},

Figure 2: Sample Penal Code section penalcode_spider.py JSON output.

Next, the Linux stream editor sed(1) and number lines nl(1) shell pipeline shown in Figure 3 was
run. It transformed the spider JSON output format in Figure 2 into one similar to SQuAD2.0 to
support the manual addition of questions with none, one, or more answers as seen in Figure 4.

sed --expression="s/^{//" $1 | nl --starting-line-number=-1 --number-format=ln --number-width=1
--number-separator=", " | sed --expression="s/^./{\"id\": &/" --expression="s/\"sec\"/\"title\"/"
--expression="s/\"law\":/\n \"question\": \"QQQ\",\n \"answers\": \[{\"text\": \"AAA\",
\"answer_start\": -1}], \"is_impossible\": false,\n \"context\":/"
--expression="1c\{\"version\": \"v0.1\", \"data\": [" --expression="\$c\]}"

Figure 3: sed(1) script to add empty SQUAD2.0-style Q&A ?elds.

{"version": "v0.1", "data": [
{"id": 0, "title": "1",
"question": "QQQ", "answers": [{"text": "AAA", "answer_start": -1}], "is_impossible": false,
"context": "Section 1. This Act shall be known as The Penal Code of California, and is divided
into four parts, as follows: I.-OF CRIMES AND PUNISHMENTS. II.-OF CRIMINAL PROCEDURE. III.-OF
THE STATE PRISON AND COUNTY JAILS. IV.-OF PREVENTION OF CRIMES AND APPREHENSION OF CRIMINALS."},

Figure 4: Sample empty question and answer, and penal code section.

The penalcode.py Python script that ran the questions through albert-xlarge-v2-squad-v2
skipped all entries where the "question" ?eld is still "QQQ". This indicates a question has not
been created for that Code section yet. For CalPenalCodeQA v0.1, three hundred questions and an-
swers were manually added to the JSON ?le, 259 with correct answer(s) and 41 that have no correct
answer1. When writing the unanswerable questions, care was taken to follow the ?two desiderata?
goals for them posited by SQuAD2.0, namely relevance and existence of plausible answers [8].
The ALBERT model can only process a maximum of 512 tokens (semantic units) at once.
All
penal code sections used for questions that exceeded this limit (as reported by the script) were
manually subdivided into multiple JSON records. Some other sections were duplicated in order to
pose di?erent questions about them.
California Penal Code dataset question answering performance
As seen in Figure 5, exact match (EM) for the entire 300-question paragraph-based dataset was
89%, with an F1 score of 89.8%. The model scored 91.12% EM on the 259 questions that have
answer(s), as compared to only 75.61% EM for the 41 unanswerable questions.
Running on anNVIDIA GeForce RTX 3070 laptop GPU with 5,120 CUDA 3rd generation tensor cores, and 8 GB
of GDDR6 memory at 1290-1620 MHz boost clock, model performance is 0.085 seconds per question.
Running on an Intel Core i7-10750H CPU at 2.6 Ghz with 6 cores mean performance is 2.63 seconds
per question.
267/300 exact matches = 89.00%
F1 = 89.80%
31/41 correct "no answers" = 75.61%
236/259 "has answer" exact matches = 91.12%
elapsed answering time: 25428 ms
0.085 seconds per question
Figure 5: CalPenalCodeQA Performance on GeForce RTX 3070 laptop GPU
Figure 6 is an example of a question that ALBERT answered correctly. Notice that the question
noun, `o?cer' is presented without either of its adjectives, and is only one of four nouns in the
sentence. The question verb, `receives' is only one of three verbs in the answer-containing sentence.
The answer is located 41 words away from the closest question word, `bribe' in this 195-word sentence.
Sentences of this length and complexity are not found in the SQUAD2.0 Wikipedia paragraphs that
our ALBERT model was trained to answer questions on.
EM ? 20: What is the punishment for an officer who receives a bribe?
ALBERT: imprisonment in the state prison for two, three, or four years
{"id": 20, "title": "68",
"question": "What is the punishment for an officer who receives a bribe?",
"answers": [{"text": "imprisonment in the state prison for two, three, or four years",
"answer_start": -1}], "is_impossible": false,
"context": "Section 68. (a) Every executive or ministerial officer, employee, or appointee of
the State of California, a county or city therein, or a political subdivision thereof, who asks,
receives, or agrees to receive, any bribe, upon any agreement or understanding that his or her
vote, opinion, or action upon any matter then pending, or that may be brought before him or her
in his or her official capacity, shall be influenced thereby, is punishable by imprisonment in
the state prison for two, three, or four years and, in cases in which no bribe has been actually
received, by a restitution fine of not less than two thousand dollars ($2,000) or not more than
ten thousand dollars ($10,000) or, in cases in which a bribe was actually received, by a
restitution fine of at least the actual amount of the bribe received or two thousand dollars
($2,000), whichever is greater, or any larger amount of not more than double the amount of any
bribe received or ten thousand dollars ($10,000), whichever is greater, and, in addition thereto,
forfeits his or her office, employment, or appointment, and is forever disqualified from holding
any office, employment, or appointment, in this state. (b) In imposing a restitution fine pursuant
to this section, the court shall consider the defendant's ability to pay the fine."},
Figure 6: Sample question, ALBERT answer, and penal code section from CalPenalCodeQA.json ?le
One reason ALBERT scored signi?cantly lower on questions without an answer was due to the
inclusion of questions that test ALBERT's ability to detect a single `nonsense word' replacing the
topic of an otherwise reasonable question. For example, in question 292 in Figure 7, ALBERT is ?rst
asked a question about a person in Section 186.33 and answers correctly. In question 293, the noun
`person' is replaced with `airplane': ?In section 186.33, any airplane required to register pursuant to
Section 186.30 who knowingly violates any of its provisions is guilty of what?? ALBERT cannot tell
the di?erence between a person and an airplane and gives the same answer, a misdemeanor instead
of no answer. We hypothesize that ALBERT is not able to detect the single nonsense word because
all the other words `?t' the pattern of a sentence that contains the answer to those words.
EM ? 292: In section 186.33, any person required to register pursuant to Section 186.30 who
knowingly violates any of its provisions is guilty of what?
ALBERT: a misdemeanor
F1 0.0 ? 293: In section 186.33, any airplane required to register pursuant to Section 186.30 who
knowingly violates any of its provisions is guilty of what?
ALBERT: a misdemeanor
answer: ['']
{"id": 293, "title": "186.33",
"question": "In section 186.33, any airplane required to register pursuant to Section 186.30 who
knowingly violates any of its provisions is guilty of what?",
"answers": [{"text": "", "answer_start": -1}], "is_impossible": true,
"context": "Section 186.33. (a) Any person required to register pursuant to Section 186.30 who
knowingly violates any of its provisions is guilty of a misdemeanor. (b) (1) Any person who
knowingly fails to register pursuant to Section 186.30 and is subsequently convicted of, or any
person for whom a petition is subsequently sustained for a violation of, any of the offenses
specified in Section 186.30, shall be punished by an additional term of imprisonment in the state
prison for 16 months, or two or three years. The court shall select the sentence enhancement which,
in the court's discretion, best serves the interests of justice and shall state the reasons for its
choice on the record at the time of sentencing in accordance with the provisions of subdivision (d)
of Section 1170.1. (2) The existence of any fact bringing a person under this subdivision shall be
alleged in the information, indictment, or petition, and be either admitted by the defendant or
minor in open court, or found to be true or not true by the trier of fact. (c) This section shall
remain in effect only until January 1, 2022, and as of that date is repealed, unless a later
enacted statute, that is enacted before January 1, 2022, deletes or extends that date."},
Figure 7: Normal and `nonsense word' questions, ALBERT answer, and penal code section
CalPenalCodeQA v0.1 and SQuAD2.0 Flesch Reading Ease Comparison
SQuAD2.0 contains paragraphs from popular English Wikipedia articles. An analysis of 1,710,752
suitable (more than 5 sentence) English Wikipedia articles as of 17 August 2010 found ?the average
reading ease score was 51.18 (SD = 13.84)? [6]. This score is interpreted as ?fairly di?cult? by the
Flesch Reading Ease test that was used [3].
The same (and additional) grade level tests were applied to the CalPenalCodeQA v0.1 dataset with
the results seen in Figure 8. These metrics were calculated by the textstat 0.7.0 Python package
[1]. The Flesch Reading Ease score for our dataset is -37.8, which is interpreted as ?very confusing?.
CalPenalCodeQA v0.1 contains: 612 sentences and 83527 syllables
Flesch Reading Ease: -37.8 (Very Confusing)
Flesch-Kincaid Grade Level: 45.2
Gunning FOG Grade Level: 47.7
Automated Readability Grade Level: 55.9
Coleman-Liau Grade Level: 11.7
Linsear Write Grade Level: 47.4
Dale-Chall Grade Level: 12.4
Six-metric mean reading grade level: 36.7

Discussion
Achieving 91.12% exact match on the 259 CalPenalCodeQA v0.1 answerable questions demonstrates
the utility of ALBERT machine reading highly-complex, domain-speci?c legal text when pre-trained
on SQuAD2.0, which is based upon general text with a much lower human reading di?culty. AL-
BERT's weakest area uncovered by this investigation is detecting `nonsense' word(s) in an otherwise
sensible question and withholding an answer.
Areas for future research include:
1. Add a section index front-end to allow ALBERT to try to answer any question about the entire
California Penal Code given the section number.
2. Build a reverse keyword index of the Penal Code that maps keywords to sections. Use this
index to return the ?rst n section(s) where keywords in the question appear. Ask ALBERT
to answer the question given each of these sections. Return the answer that has the highest
score, if any.
3. Scrape the California Constitution and the text of all California Laws from the California
Legislative Information website. Apply 1. and 2. indexing strategies to create a general purpose
California law-answering system. The same could be done with any group of laws or bodies of
text.
4. Add at least another 700 questions to CalPenalCodeQA in order to pre-train, verify, and test
the ALBERT SQuAD2.0 model on it to see if EM and F1 increase.
References
[1] Shivam Bansal and Chaitanya Aggarwal.
textstat 0.7.0.
https://pypi.org/project/textstat,
2020. Accessed: 2021-03-15.
[2] Hugging Face. albert-xlarge-v2. https://huggingface.co/albert-xlarge-v2, 2019. Accessed: 2021-
03-13.
[3] Rudolph Flesch. A new readability yardstick. Journal of Applied Psychology, volume 32, number
3, pp. 221?233, 1948.
[4] Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu
Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. CoRR,
abs/1909.11942, 2019.
[5] California State Legislature. The penal code of california. https://leginfo.legislature.ca.gov/
faces/codedisplayexpand.xhtml?tocCode=PEN, 2021. Accessed: 2021-02-26.
[6] Teun Lucassen, Roald Dijkstra, and Jan Maarten Schraagen. Readability of wikipedia. First
Monday, Aug. 2012.
[7] Pranav Rajpurkar and Robin Jia.
Squad2.0:
The stanford question answering dataset.
https://rajpurkar.github.io/SQuAD-explorer, 2021. Accessed: 2021-03-13.
[8] Pranav Rajpurkar, Robin Jia, and Percy Liang. Know what you don't know: Unanswerable
questions for squad. ArXiv, abs/1806.03822, 2018.
[9] Kirill Trapeznikov.
albert-xlarge-v2-squad-v2.
https://huggingface.co/ktrapeznikov/albert-
xlarge-v2-squad-v2, 2020. Accessed: 2021-03-13.
[10] Thomas Wood. F-score. https://deepai.org/machine-learning-glossary-and-terms/f-score. Ac-
cessed: 2021-03-13.
[11] Zyte. Scrapy. https://scrapy.org, 2021. Accessed: 2021-02-26.
