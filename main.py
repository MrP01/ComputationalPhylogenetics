import computationalphylogenetics.data
from computationalphylogenetics.model import (
    IPAReplacementOperation,
    similarityScore,
)

if __name__ == "__main__":
    english, german = computationalphylogenetics.data.load_swadesh()

    set_of_operations = []
    # set_of_operations = [IPAReplacementOperation("t", "t͡s")]
    input_list = english.copy()
    for op in set_of_operations:
        input_list.items = tuple(map(op.applyTo, input_list.items))
    score = similarityScore(input_list, german)
    print(score)
