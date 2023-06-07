import nltk


class IPAReplacement(object):
    pass


class IPAReplacementOperation(IPAReplacement):
    def __init__(self, source: str, replacement: str):
        self.source = source
        self.replacement = replacement

    def applyTo(self, word):
        return word.replace(self.source, self.replacement)


class SwadeshList(object):
    def __init__(self, language: str, items: tuple[str]) -> None:
        self.language = language
        self.items = items
        # if len(self.items) != 207:
        #     raise ValueError("Are you nuts? This ain't a Swadesh list!")

    def copy(self):
        return SwadeshList(self.language, self.items)


def similarityScore(listA: SwadeshList, listB: SwadeshList):
    return sum(nltk.edit_distance(a, b) for a, b in zip(listA.items, listB.items))
