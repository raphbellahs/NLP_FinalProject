from project_evaluate import *
import evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import sacrebleu
from datasets import *
import spacy
import string
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def read_modified_file(file_path):
    """
    This function reads a modified file format containing German sentences,
    their corresponding roots in English, and their corresponding modifiers in English.
    The function reads the file and returns a tuple containing lists of the German sentences, roots, and modifiers.

    Parameters:
    file_path (str): A string specifying the path to the file to be read.

    Returns:
    tuple: A tuple containing the following lists:
    - german_sentence: A list of German sentences.
    - roots: A list of English roots corresponding to the German sentences.
    - modifiers: A list of English modifiers corresponding to the German sentences.
    """
    with open(file_path, 'r', encoding='utf8') as f:
        german_sentence = []
        roots = []
        modifiers = []
        # Split the file contents into pairs of German and English sentences
        raw_data = f.read().strip().split('\n\n')
        i = 0
        for pair in raw_data:
            pair = pair.split('\nModifiers in English:')
            modifiers.append(pair[1])
            pair = pair[0].split("\nRoots in English:")
            roots.append(pair[1])
            pair = pair[0].split('German:\n')
            german_sentence.append(pair[1])
        return german_sentence, roots, modifiers

def load_dataset(files):
    """
    This function loads data from files containing modified file format with German sentences,
    their corresponding roots in English, and their corresponding modifiers in English.
    The function returns a list of Dataset objects, each Dataset object containing the German sentences,
    their corresponding roots, and their corresponding modifiers.

    Parameters:
    files (list): A list of strings specifying the paths to the files to be loaded.

    Returns:
    list: A list of Dataset objects, each containing the following features:
    - de: A string containing the German sentence.
    - root: A string containing the English root corresponding to the German sentence.
    - mod: A string containing the English modifier corresponding to the German sentence.
    """
    raw_datasets = []
    for path in files:
        with open(path, 'r', encoding='utf8') as f:
            dataset = {'translation': []}
            de_data, root, modifiers =read_modified_file(path)
            for(de_sen,root,mod) in zip(de_data,root, modifiers):
                dataset['translation'].append({'de': de_sen, 'root':root, 'mod':mod})
            # Create a Dataset object from the dictionary and add it to the list
            dataset = Dataset.from_dict(dataset)
            raw_datasets.append(dataset)
    return raw_datasets

def preprocess_function(dataset):
    """
        Preprocesses a dataset of translations by generating tokenized inputs that consist of a prefix
        concatenated with the source language sentences, the roots in English, their POS tags,
        and the modifiers in English.

        Args:
            dataset (Dataset): A dataset of translations that contains a list of dictionaries, w
            here each dictionary has the keys "de" (German sentence),
            "root" (roots of the German sentence in English), and "mod" (modifiers of the German sentence in English).

        Returns:
            A dictionary containing the tokenized inputs, where the key "input_ids" corresponds to the token IDs of the
            input sequences, and the key "attention_mask" corresponds to the attention masks of the input sequences.
        """
    inputs = []
    for sen in dataset["translation"]:
        rootsPrefix = " Roots in English are : " + sen['root'] + ', '
        # --------- POS TAGGING FOR ROOT --------
        # words = word_tokenize(sen['root'].replace(",", " "))
        # pos_tags = pos_tag(words)
        # pos_tags_str = ','.join([tag for (word, tag) in pos_tags])

        # posPrefix = " The tags for the roots in English are : " + pos_tags_str

        modifPrefix = " Modifiers in English are: " + sen['mod']
        sentencePrefix = sen[source_lang]
        inputs.append(prefix + " " + sentencePrefix + rootsPrefix + modifPrefix)
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    return model_inputs

def read_file(file_path):
    """
    This function reads a text file in UTF-8 encoding, which contains English and German language texts.
    The file should have the following format:
    The function returns two lists, file_en and file_de, containing the English and German texts respectively.

    Parameters:
    file_path (str): The path of the text file to be read.

    Returns:
    tuple: A tuple containing two lists of strings
    - file_en and file_de -
    which contain the English and German texts respectively.
    """
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            cur_str += line + ' '
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de

def load_dataset_val(files):
    """
    This function loads a list of text files containing English and German texts and creates a list of datasets.
    Each dataset contains a list of translations, where each translation is a dictionary with 'en' and 'de' keys
    for the English and German text respectively.

    Parameters:
    files (list): A list of file paths to be loaded.

    Returns:
    list: A list of datasets, where each dataset is a Dataset object containing a list of translations.
    """
    raw_datasets = []
    for path in files:
        with open(path, 'r', encoding='utf8') as f:
            dataset = {'translation': []}
            en_data, de_data =read_file(path)
            for en_sen, de_sen in zip(en_data, de_data):
                dataset['translation'].append({'de': de_sen, 'en': en_sen})
            # Create a Dataset object from the dictionary and add it to the list
            dataset = Dataset.from_dict(dataset)
            raw_datasets.append(dataset)
    return raw_datasets

def preprocess_function_val(dataset):
    """
    This function preprocesses a dataset of translations by tokenizing the inputs and targets using a tokenizer object.
    The inputs are generated by concatenating a prefix with the source language sentences,
    while the targets are the target language sentences.

    Parameters:
    dataset (Dataset): A Dataset object containing a list of translations.

    Returns:
    dict: A dictionary containing tokenized model inputs and labels for the translation task.
    The dictionary contains the following keys:
    - input_ids: A list of tokenized input sequences.
    - attention_mask: A list of binary values indicating which tokens should be attended to by the model.
    - labels: A list of tokenized target sequences.
    """
    inputs = [prefix + sen[source_lang] for sen in dataset["translation"]]
    targets = [sen[target_lang] for sen in dataset["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__ == '__main__':
    file = sys.argv[1]
    if "val" in file:
        # Step 0: Load the data
        TRAIN = 'data/train.labeled'
        VAL = 'data/val.labeled'
        DF = load_dataset_val([TRAIN, VAL])
        raw_datasets = DatasetDict({"train": DF[0], "validation": DF[1]})
        # Step: Tokenization
        model_name = "t5-base"
        # model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        max_input_length = 500
        max_target_length = 500
        source_lang = "de"
        target_lang = "en"
        prefix = "translate German to English: "
        tokenized_datasets = raw_datasets.map(preprocess_function_val, batched=True)
        tokenized_datasets.set_format('torch')
        """
        sample = tokenized_datasets["validation"]
        print(sample)
        input_ids = sample["input_ids"]
        print(type(input_ids))
        input_ids = sample["input_ids"][0]
        print(input_ids)
        quit()
        """

        # Step 1: Load our model
        print("Loading t5-base-7epoch-model")
        model = AutoModelForSeq2SeqLM.from_pretrained("t5-base-7epoch-model").to("cuda:0")
        metric = evaluate.load("sacrebleu")

        # Step 2: Translate every sentences from val_unlabeled
        print('Translating!')
        generated_translations = []
        for i in range(len(tokenized_datasets["validation"])):
            if i % 25 == 0:
                print('translating sentence ' + str(i) + '/1000')
            sample = tokenized_datasets["validation"]
            input_ids = sample["input_ids"][i].unsqueeze(0)
            attention_mask = sample["attention_mask"][i].unsqueeze(0)

            input_ids = input_ids.to("cuda:0")
            attention_mask = attention_mask.to("cuda:0")

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=500,
            )

            # Decode the generated output and the ground truth
            generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
            # Store the translation
            generated_translations.append(generated_translation)
        print('Translation done!')

        # Step 3: Generate file val_id1_id2_labeled
        _, german_sentences = read_file(VAL)
        f = open("val_209948728_931188684_labeled.txt", "w", encoding="utf-8")
        for i in range(len(german_sentences)):
            f.write("German:\n" + german_sentences[i] + "\n" + "English:\n" + generated_translations[i] + "\n")
        f.close()

        # Step 4: calculate score
        val_labeled = "val_labeled"
        calculate_score(val_labeled, "val_209948728_931188684_labeled.txt")
    
    else:
        # # Load English language model for dependency parsing
        # nlp = spacy.load("en_core_web_sm")
        # # nltk download
        # nltk.download('averaged_perceptron_tagger')
        
        COMP = 'data/comp.unlabeled'
        german_sentence, roots, modifiers = read_modified_file(COMP)
        #print('Len de german_sentences:')
        #print(len(german_sentence))
        #print(german_sentence)
        #print(type(german_sentence))
        #print(german_sentence[0])
        #print(len(german_sentence))
        myDataset = load_dataset([COMP])
        compDS = DatasetDict({"comp": myDataset[0]})
        print(compDS)
    
        model_name = "t5-base"
        max_input_length = 750
        max_target_length = 750
        source_lang = "de"
        target_lang = "en"
        prefix = "translate German to English"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
        tokenized_DS = compDS.map(preprocess_function, batched=True)
        tokenized_DS.set_format('torch')
        sample = tokenized_DS["comp"]
        input_ids = sample["input_ids"][0].unsqueeze(0)
        #print(input_ids)
        attention_mask = sample["attention_mask"][0].unsqueeze(0)
        #print(attention_mask)
    
        # Step 1: Load our model
        print("Loading model-r&m-t5-base-5epoch")
        model = AutoModelForSeq2SeqLM.from_pretrained("model-r&m-t5-base-5epoch").to("cuda:0")
        metric = evaluate.load("sacrebleu")
    
        # Step 2: Translate every sentences from val_unlabeled
        print('Translating!')
        generated_translations = []
        for i in range(len(tokenized_DS["comp"])):
            if i % 25 == 0:
                print('translating sentence ' + str(i) + '/1000')
                # print(tokenizer.decode(sample["input_ids"][i].unsqueeze(0)))
            sample = tokenized_DS["comp"]
            input_ids = sample["input_ids"][i].unsqueeze(0)
            attention_mask = sample["attention_mask"][i].unsqueeze(0)
    
            input_ids = input_ids.to("cuda:0")
            attention_mask = attention_mask.to("cuda:0")
    
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=750,
            )
    
            # Decode the generated output and the ground truth
            generated_translation = tokenizer.decode(output[0], skip_special_tokens=True)
            # Store the translation
            generated_translations.append(generated_translation)
        print('Translation done!')
    
        # Step 3: Generate file val_id1_id2_labeled
        german_sentences, _, _,  = read_modified_file(COMP)
        print('Len de german_sentences:')
        print(len(german_sentences))
        print(('Len de generated_translations:'))
        print(len(german_sentences))
        f = open("comp_209948728_931188684.labeled", "w", encoding="utf-8")
        for i in range(len(german_sentences)):
            f.write("German:\n" + german_sentences[i] + "\n" + "English:\n" + generated_translations[i] + "\n")
        f.close()
