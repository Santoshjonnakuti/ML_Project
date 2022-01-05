import re
import nltk
# nltk.download()
from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


def preProcessData(text):
    # to lower the text
    text = text.lower()
    # to remove mentions
    text = re.sub(r'@[a-z0-9]+', '', text)
    # to remove hyperlinks i.e, https://, http://...
    text = re.sub(r'https?://\S+', '', text)
    # to remove punctuations
    text = re.sub(r'[^a-z\s]', '', text)
    # to remove stopwords and stemming the words
    wNL = WordNetLemmatizer()
    text = text.split()
    text = [wNL.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
