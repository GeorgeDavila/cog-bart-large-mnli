# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import pipeline

MODEL_NAME = "facebook/bart-large-mnli"
MODEL_CACHE = "model-cache"
TOKEN_CACHE = "token-cache"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = BartTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=TOKEN_CACHE
        )
        model = BartForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE
        )
        self.model = model.to("cuda")

    def predict(
        self,
        text2classify: str = Input(description="Text you want to classify. ", default="Add salt to boiling water to prevent pasta from sticking together"),
        labels: str = Input(description="Possible class names (comma-separated). This is a zero-shot classifier so you can try any label you'd like. The model will output the top label under key 'mostLikelyClass'.", default="Cooking Instructions, Question about Astronomy"),
    ) -> str:
        """Run a single prediction on the model"""

        classifier = pipeline("zero-shot-classification", model=MODEL_NAME)

        def zeroShotClassification(text_input, candidate_labels):
            labels = [label.strip(' ') for label in candidate_labels.split(',')]
            output = {}
            prediction = classifier(text_input, labels)
            for i in range(len(prediction['labels'])):
                output[prediction['labels'][i]] = prediction['scores'][i]
            return output

        likelihoods = zeroShotClassification(text_input=text2classify, candidate_labels=labels)
        classesSortedMost2Least = sorted(likelihoods.items(), key=lambda x:x[1], reverse=True)
        mostLikelyClass = classesSortedMost2Least[0][0]

        response = {'mostLikelyClass': mostLikelyClass, 'allClasses':likelihoods}

        return response
