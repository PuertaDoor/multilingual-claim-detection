import os
import re
import glob
import csv
import transformers
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import torch
import time
from collections import defaultdict
import random
import string

transformers.logging.set_verbosity_error()

torch.cuda.empty_cache()

def delete_tsv_files(current_dir):
    """Supprime tous les fichiers .tsv dans le répertoire spécifié."""
    for f in glob.glob(os.path.join(current_dir, "*.tsv")):
        os.remove(f)

def compile_tsv_files_without_headers(current_dir, file_names):
    """Compile les fichiers TSV sans les en-têtes dans le répertoire spécifié."""
    compiled_data = {name: [] for name in file_names}
    
    for subdir, dirs, files in os.walk(current_dir):
        for file in files:
            if file in file_names:
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r', encoding='utf-8') as file_obj:
                    lines = file_obj.readlines()
                    if lines and re.match(r'^[A-Za-z\s]+\t[A-Za-z\s]+$', lines[0].strip()):
                        lines = lines[1:]  # Ignorer la ligne d'en-tête
                    compiled_data[file].extend(lines)
    
    for file_name, data in compiled_data.items():
        with open(os.path.join(current_dir, file_name), 'w', encoding='utf-8') as compiled_file:
            compiled_file.writelines(data)

def clean_eof_issues(input_file_path, temp_output_file_path):
    """Nettoie les problèmes de guillemets de fin de fichier."""
    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(temp_output_file_path, 'w', encoding='utf-8') as temp_output_file:
        for line in input_file:
            parts = line.strip().split('\t')
            if len(parts) == 2 and parts[0]:
                line += '"' * (line.count('"') % 2)  # Ajoute un guillemet si le nombre est impair
                temp_output_file.write(line)

# Usage
current_dir = '.'  # Répertoire courant
file_names = ['train.tsv', 'test.tsv', 'dev.tsv']

delete_tsv_files(current_dir)
compile_tsv_files_without_headers(current_dir, file_names)

for name in file_names:
    clean_eof_issues(os.path.join(current_dir, name), os.path.join(current_dir, f"{name}_temp"))
    os.replace(f"{name}_temp", name)

language_families = {
    'germanic': ['szl_Latn', 'nob_Latn', 'nno_Latn', 'lim_Latn', 'eng_Latn', 'deu_Latn', 'nld_Latn', 'afr_Latn', 'dan_Latn', 'swe_Latn', 'isl_Latn', 'ltz_Latn', 'fao_Latn', 'ydd_Hebr'],
    'romance': ['vec_Latn', 'lij_Latn', 'srd_Latn', 'scn_Latn', 'lmo_Latn', 'fur_Latn', 'fra_Latn', 'spa_Latn', 'ita_Latn', 'por_Latn', 'ron_Latn', 'cat_Latn', 'oci_Latn', 'glg_Latn', 'ast_Latn'],
    'baltic': ['ltg_Latn', 'lvs_Latn', 'lit_Latn', 'eng_Latn'],
    'slavic': ['pol_Latn', 'ces_Latn', 'slk_Latn', 'slv_Latn', 'srp_Cyrl', 'hrv_Latn', 'bos_Latn', 'bul_Cyrl', 'mkd_Cyrl', 'rus_Cyrl', 'bel_Cyrl', 'ukr_Cyrl'],
    'albanese': ['als_Latn', 'eng_Latn'], # langue isolée donc traduction seulement en anglais
    'armenian': ['hye_Armn', 'eng_Latn', 'rus_Cyrl'], # itou + russe car grosse influence
    'celtic': ['gla_Latn', 'gle_Latn', 'cym_Latn', 'eng_Latn'], # car traduction anglaise vers gaélique souvent
    'hellenic': ['ell_Grek', 'eng_Latn'], # isolated
    'turkish': ['crh_Latn', 'tat_Cyrl', 'tur_Latn', 'azj_Latn', 'azb_Arab', 'eng_Latn'], # not indo-euro
    'finno-ugric': ['hun_Latn', 'fin_Latn', 'est_Latn', 'eng_Latn'], # en pour augmenter la collection
    'arabic': ['arz_Arab', 'ary_Arab', 'ars_Arab', 'arb_Arab', 'apc_Arab', 'ajp_Arab', 'aeb_Arab', 'acq_Arab', 'acm_Arab', 'ace_Arab', 'mlt_Latn', 'eng_Latn'],
    'euskara': ['eus_Latn', 'spa_Latn', 'eng_Latn'],
    'georgian': ['kat_Geor', 'eng_Latn'],
    'esperanto': ['epo_Latn', 'eng_Latn']
}

# Configuration initiale du modèle
model_name = 'facebook/nllb-200-distilled-600M'
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
tokenizer = NllbTokenizer.from_pretrained(model_name)

id_to_lang_code = {valeur: cle for cle, valeur in tokenizer.lang_code_to_id.items()}

def clean_translation(text):
    """Nettoie le texte traduit en enlevant les crochets, les guillemets, et la ponctuation à la fin."""
    # Enlève les crochets et guillemets
    cleaned_text = text.strip("[]'\"")
    # Enlève la ponctuation finale
    if cleaned_text and cleaned_text[-1] in string.punctuation:
        cleaned_text = cleaned_text[:-1]
    return cleaned_text

def get_target_language(src_lang, language_families):
    """Retourne les langues cibles de la même famille que la langue source."""
    return [lang for family, languages in language_families.items() if src_lang in languages for lang in languages if lang != src_lang]

def translate_text_batch(text, src_lang, target_lang):
    """Traduit un texte vers la langue cible spécifiée, en spécifiant la langue source."""
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    translated_tokens = model.generate(
        inputs.input_ids, attention_mask=inputs.attention_mask,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang], max_length=512
    )
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return ''.join(c for c in translated_text if c not in string.punctuation) # Enlève la ponctuation

def augment_dataset_with_family_translation(input_file_path, output_file_path, language_families=language_families, num_texts=300):
    texts_by_lang = defaultdict(list)
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        reader = csv.reader(input_file, delimiter='\t')
        for row in reader:
            text, label = row[0], row[1]
            inputs = tokenizer(text, return_tensors="pt").input_ids[0, 0].item()
            src_lang_code = id_to_lang_code.get(inputs)
            if src_lang_code:
                texts_by_lang[src_lang_code].append((text, label))

    selected_texts = []
    for lang, texts in texts_by_lang.items():
        num_texts_per_lang = max(1, int((len(texts) / sum(len(t) for t in texts_by_lang.values())) * num_texts))
        selected_texts.extend(random.sample(texts, min(len(texts), num_texts_per_lang)))

    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        writer = csv.writer(output_file, delimiter='\t')
        i = 0
        start_time = time.time()
        for text, label in selected_texts:
            inputs = tokenizer(text, return_tensors="pt").input_ids[0, 0].item()
            src_lang_code = id_to_lang_code.get(inputs)
            target_langs = get_target_language(src_lang_code, language_families)
            for target_lang in target_langs:
                translated_text = translate_text_batch(text, src_lang_code, target_lang)
                writer.writerow([translated_text, label])  # Ajoute le label correspondant

            elapsed_time = time.time() - start_time
            texts_per_second = (i + 1) / elapsed_time
            remaining_time = (len(selected_texts) - (i + 1)) / texts_per_second
            print(f"Processed {i+1}/{len(selected_texts)} texts. Estimated time remaining: {remaining_time:.2f} seconds.")

            i += 1


augment_dataset_with_family_translation('train.tsv', 'train_aug.tsv')