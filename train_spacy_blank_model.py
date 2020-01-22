# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# training data
TRAIN_DATA = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}),
              ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}),
              ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]})]



@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
# def main(model=r'en_core_web_sm', output_dir='en_sk', n_iter=3):
def main(model=None, output_dir='en_sahil_k', n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    
    print (nlp)
    
    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        print (output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)
