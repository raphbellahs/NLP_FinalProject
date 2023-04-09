from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import *
from project_evaluate import *
import spacy
import string
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def read_file(file_path):
    """
        Reads a file containing English and German sentences, and returns two lists:
        one containing all English sentences and another containing all German sentences.

        Args:
        - file_path (str): The path to the file to be read.

        Returns:
        - english_sentences (list): A list of all English sentences in the file.
        - german_sentences (list): A list of all German sentences in the file.
        """
    english_sentences, german_sentences = [], []
    with open(file_path, encoding='utf-8') as file:
        current_sentence, current_list = '', []
        for line in file.readlines():
            # If line indicates a language change, switch to the corresponding list
            if line == 'English:\n' or line == 'German:\n':
                if len(current_sentence) > 0:
                    current_list.append(current_sentence)
                    current_sentence = ''
                # Beggining of English Sentence
                if line == 'English:\n':
                    current_list = english_sentences
                # Beggining of German Sentence
                else:
                    current_list = german_sentences
                continue
            # If line is part of a sentence, add it to the current sentence
            current_sentence += line
        # Add the last sentence to the appropriate list
        if len(current_sentence) > 0:
            current_list.append(current_sentence)
    return english_sentences, german_sentences

def get_root_and_modifier(sentences):
    """
    Returns the roots and modifiers of the given list of sentences using spaCy's dependency parser.

    Args:
    sentences (list): A list of strings representing sentences.

    Returns:
    tuple: A tuple containing two lists. The first list contains the roots of each sentence, and the second list contains the modifiers of each sentence.
    """
    rootsOfSentences = []
    modifiersOfSentences = []
    for sent in sentences:

        # Create a Doc object for the sentence to be parsed
        doc = nlp(sent)

        # Iterate over the tokens in the sentence and print their dependencies
        root = ''
        NewModifiers = []
        for token in doc:
            if token.pos_ == 'PUNCT':
                continue
            if token.dep_ == 'ROOT':
                root = token.text
                modifiers = [child.text for child in token.children]
                NewModifiers = [word for word in modifiers if word not in string.punctuation]
                break
        rootsOfSentences.append(root)
        if len(NewModifiers) >= 2:
            modifiersOfSentences.append(np.random.choice(NewModifiers, 2, replace=False))
        else:
            modifiersOfSentences.append(NewModifiers)

    roots_str = "Roots in English: " + ', '.join(rootsOfSentences) + "\n"
    modifiers_str = "Modifiers in English: " + ', '.join(
        ['(' + ', '.join(modifiers) + ')' for modifiers in modifiersOfSentences])

    return roots_str + modifiers_str

def writeTrain(file_path, out_file_path):
    """
        Reads a file containing parallel English and German sentences, extracts the root and modifiers for each English sentence
        using the get_root_and_modifier function, and writes a new file with modified English sentences and their corresponding
        German sentences.

        Args:
            file_path (str): The path to the input file.
            out_file_path (str): The path to the output file to be created.

        Returns:
            None
        """
    en_sentences, de_sentences = read_file(file_path)
    out_file = open(out_file_path, 'w', encoding='utf-8')

    en_root_mod = []
    for i in tqdm(range(len(en_sentences))):
        en_root_mod.append(get_root_and_modifier(en_sentences[i].split('\n')[:-2]))
    print('SAVING NEW FILE !')
    for i in tqdm(range(len(en_sentences))):
        out_file.write("German:\n" + de_sentences[i] + en_root_mod[i] + "\nEnglish:\n" + en_sentences[i])
    print('FINISHED !')

def writeTest(file_path, out_file_path, rootsVal, modifiersVal):
    """
    Write the test data with given roots and modifiers in the specified format to a file.

    Args:
    file_path (str): The path to the test data file.
    out_file_path (str): The path to the output file.
    rootsVal (list): A list of root words corresponding to each sentence in the test data.
    modifiersVal (list): A list of modifiers corresponding to each sentence in the test data.

    Returns:
    None
    """
    en_sentences, de_sentences = read_file(file_path)
    out_file = open(out_file_path, 'w', encoding='utf-8')
    for i in tqdm(range(len(en_sentences))):
        out_file.write(
            "German:\n" + de_sentences[i] + rootsVal[i] + "\n" + modifiersVal[i] + "\nEnglish:\n" + en_sentences[i])
    print('FINISHED!')

def read_modified_file(file_path):
    """
       Reads in a modified parallel corpus file and extracts the English and German sentences along with their respective
       roots and modifiers.

       Args:
           file_path (str): The path to the modified parallel corpus file.

       Returns:
           Tuple of four lists: English sentences, German sentences, roots of English sentences, and modifiers of English
           sentences.
       """
    with open(file_path, 'r', encoding='utf8') as f:
        english_sentence = []
        german_sentence = []
        roots = []
        modifiers = []
        # Split the file contents into pairs of German and English sentences
        raw_data = f.read().strip().split('\n\n')
        i = 0
        for pair in raw_data:
            pair = pair.split('\nEnglish:\n')
            english_sentence.append(pair[1].replace("\n", " "))
            pair = pair[0].split('\nModifiers in English:')
            modifiers.append(pair[1])
            pair = pair[0].split("\nRoots in English:")
            roots.append(pair[1])
            pair = pair[0].split('German:\n')
            german_sentence.append(pair[1])

        return english_sentence, german_sentence, roots, modifiers

def load_dataset(files):
    """
        Loads the training and testing datasets from the given file paths, and returns a list of Hugging Face Datasets.

        Args:
        - files: A list of file paths.

        Returns:
        - raw_datasets: A list of Hugging Face Datasets.
        """
    raw_datasets = []
    for path in files:
        print(path)
        with open(path, 'r', encoding='utf8') as f:
            dataset = {'translation': []}
            en_data, de_data, root, modifiers =read_modified_file(path)
            for(en_sen, de_sen,rt,mod) in zip(en_data, de_data,root, modifiers):
                dataset['translation'].append({'de': de_sen, 'en': en_sen, 'root':rt, 'mod':mod})
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
      """
      words = word_tokenize(sen['root'].replace(","," "))
      pos_tags = pos_tag(words)
      pos_tags_str = ','.join([tag for (word, tag) in pos_tags])

      posPrefix = " The tags for the roots in English are : " + pos_tags_str
      """
      modifPrefix = " Modifiers in English are: " + sen['mod']
      sentencePrefix = sen[source_lang]
      inputs.append( prefix + " " + sentencePrefix + rootsPrefix + modifPrefix )

    targets = [sen[target_lang] for sen in dataset["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def postprocess_text(preds, labels):
    """
    Postprocesses the predicted and target labels by stripping any leading or trailing whitespaces.
    Args:
        preds (List[str]): A list of predicted labels.
        labels (List[List[str]]): A list of target labels.
    Returns:
        Tuple[List[str], List[List[str]]]: A tuple of two lists.
        The first list contains the postprocessed predicted labels,
        and the second list contains the postprocessed target labels.
    """
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels

def compute_metrics(eval_preds):
    """
        Computes the evaluation metrics for a given set of predictions and labels.
        Args:
            eval_preds (tuple): A tuple containing the predictions and labels.
        Returns:
            dict: A dictionary containing the evaluation metrics.
                The metrics include 'bleu' (BLEU score) and 'gen_len' (generated sequence length).
        """
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

if __name__ == '__main__':
    print('Load val.unlabeled')
    with open('data/val.unlabeled', 'r', encoding='utf8') as f:
        raw_data = f.read().strip().split('\n\n')
        rootsVal = []
        modifiersVal = []
        for sentence in raw_data:
            new_sentence = sentence.replace("\n", " ").split("Roots in English: ")
            new_sentence = new_sentence[1].split("Modifiers in English: ")

            rootsVal.append("Roots in English: " + new_sentence[0].replace(" ", ""))
            modifiersVal.append("Modifiers in English: " + new_sentence[1])
    print(len(rootsVal))
    print(len(modifiersVal))

    # Load English language model for dependency parsing
    nlp = spacy.load("en_core_web_sm")

    TRAIN = 'data/train.labeled'
    TEST = 'data/val.labeled'
    print('writeTrain')
    writeTrain(TRAIN, 'train_root_modifier.labeled')
    print('writeTest')
    writeTest(TEST, 'test_root_modifier.labeled', rootsVal, modifiersVal)

    print('Loading train & test root modifier files')
    TRAIN_ROOT_MODIFIER = 'train_root_modifier.labeled'
    TEST_ROOT_MODIFIER = 'test_root_modifier.labeled'

    print('Creating dataset')
    myDataset = load_dataset([TRAIN_ROOT_MODIFIER, TEST_ROOT_MODIFIER])
    finalDS = DatasetDict({"train": myDataset[0], "validation": myDataset[1]})

    print('Tokenization')
    model_name = "t5-base"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prefix = "translate German to English"
    max_input_length = 512
    max_target_length = 512
    nltk.download('punkt')
    string = "is,where"
    words = word_tokenize(string.replace(',', " "))
    pos_tags = pos_tag(words)
    pos_tags_str = ','.join([tag for (word, tag) in pos_tags])
    source_lang = "de"
    target_lang = "en"
    tokenized_DS = finalDS.map(preprocess_function, batched=True)
    tokenized_DS.set_format('torch')

    metric = load_metric("sacrebleu")

    # Train
    print('Training Started!')
    batch_size = 4
    args = Seq2SeqTrainingArguments(
        output_dir="modified-t5-base-10epoch-400len-500input",
        evaluation_strategy="epoch",
        save_strategy='epoch',
        logging_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=2,
        predict_with_generate=True,
        generation_max_length=512,
        fp16=True,
        logging_steps=1000
    )
    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_DS["train"],
        eval_dataset=tokenized_DS["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    model.save_pretrained("model-r&m-t5-base-1epoch")
    trainer.save_model("r&m-t5-base-1epoch-trainer")
