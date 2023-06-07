import pathlib
import re

import pandas as pd

from .model import SwadeshList

ATTICUS_RE = re.compile(r"\/[^\/]+\/")
CLDF_FOLDER = pathlib.Path("data") / "cldf" / "cldf"
IPA_FOLDER = pathlib.Path("data") / "ipa"


def lookup_ipa(language_code, words):
    df = pd.read_csv(IPA_FOLDER / f"{language_code}.txt", delimiter="\t", names=["word", "ipa"])
    return tuple(ATTICUS_RE.search(ipa).group(0) for ipa in tuple(df[df.word.isin(words)].ipa))


def load_swadesh():
    languages = pd.read_csv(CLDF_FOLDER / "languages.csv")
    parameters = pd.read_csv(CLDF_FOLDER / "parameters.csv")
    forms = pd.read_csv(CLDF_FOLDER / "forms.csv")

    german_lang_id = languages[languages.Name == "German"].iloc[0].ID
    german_filter = forms.Language_ID == german_lang_id
    english = SwadeshList("English", lookup_ipa("en_UK", tuple(parameters.Name)))
    german = SwadeshList("German", lookup_ipa("de", tuple(forms[german_filter].Value)))
    return english, german
