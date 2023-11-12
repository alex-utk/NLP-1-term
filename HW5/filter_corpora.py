from natasha import Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,Doc
from nltk.corpus import stopwords
from corus import load_lenta
import re
from tqdm import tqdm
import pickle
import nltk
import pickle
    
segmenter = Segmenter()
morph_vocab = MorphVocab()
morph_tagger = NewsMorphTagger(NewsEmbedding())

records = load_lenta('lenta-ru-news.csv.gz') # 739351

texts = []

ru_stopwords = stopwords.words("russian")
allowed_symbols = re.compile(u'[0-9]|[^\u0400-\u04FF \.\-]')

total = 739351

for idx, article in tqdm(enumerate(records, total=total)):
    text = allowed_symbols.sub(' ', article.text.lower())
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    
    
    filtered_tokens = [token.lemma
                       for token
                       in doc.tokens
                       if token.pos in ['NOUN', 'ADJ', 'VERB', 'ADV']]
    
    cleared_tokens = list(filter(lambda token: token not in ru_stopwords, filtered_tokens))
    
    texts.append(cleared_tokens)
    
    if idx % 10000 == 0:
        pickle.dump(texts, open(f'lenta_processed_{idx}.pkl', 'wb'))
    


pickle.dump(texts, open('lenta_processed.pkl', 'wb'))
    
    # 4 часа