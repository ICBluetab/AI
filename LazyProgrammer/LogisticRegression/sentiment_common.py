
punctuation = [',', '.', '?', '\'', '(', ')', '-', '/', '$', '+', '"', '!', ':', '&', '*', ';', '~', '_', '`', '%', '=', '[', ']', '{', '}']

blacklist = ['i']


def split_text(text):
    text = text.lower().strip()
    for p in punctuation:
        text = text.replace(p, ' ')

    return [word for word in text.split() if word  not in blacklist ]
