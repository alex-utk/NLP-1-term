from pymystem3 import Mystem

mystem = Mystem()

def tokenize_and_filter(text: str, ru_stopwords: list[str]) -> list[str]:
    """Лемматизация, токенизация и постпроцессинг

    Args:
        text (str): текст
        mystem (Mystem): лемматизатор яндекса
        ru_stopwords (list[str]): стоп слова nltk

    Returns:
        list[str]: очищенный список токенов
    """
    tokens_list = mystem.lemmatize(text) # токенизация и лемматизация лемматизатором яндекса
    tokens_list = [token.strip() for token in tokens_list] # очищаем от лишних пробелов
    # так как у нас правило token == 'не' идет самое первое, то за счет оптимизации
    # логических операций слово не будет даже если оно есть в списке stopwords
    cleared_tokens = list(filter(lambda token: token == 'не' or (token not in ru_stopwords and len(token) > 2),
                          tokens_list))
    return cleared_tokens