import prodigy
from prodigy.components.loaders import JSONL
from prodigy.core import recipe
#from prodigy.components.stream import get_stream

@prodigy.recipe("ID_of_recipe")
def ehop(dataset: str, source):
    """Annotate the type of texts using different options."""
    stream = JSONL(source) # load in the JSONL file
    stream = add_options(stream)   # add options to each task

    return {
        "dataset": dataset,   # save annotations in this dataset
        "view_id": "choice",  # use the choice interface
        "stream": stream,
        #"config": {
        #    "choice_style": "multiple"
        #}
    }

def add_options(stream):
    # Helper function to add options to every task in a stream
    options = [{"id": "0", "text": "absence"},
               {"id": "1", "text": "presence"},
               {"id": "2", "text": "ancien"}]
    for task in stream:
        task["options"] = options
        yield task