# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json

app=FastAPI()

class model_input(BaseModel):
    Text: str

model, word_vectorizer = pickle.load(open('knn_model.pkl', 'rb'))

@app.post('/predict')
def pred(input : model_input):
    
    input_data=input.json()
    input_dict=json.loads(input_data)
    
    text=input_dict['Text']
    word_features = word_vectorizer.transform([text])

    # Predict probabilities for each class
    predicted_probabilities = model.predict_proba(word_features)

    # Get the top 3 predicted categories and their probabilities
    top_3_indices = predicted_probabilities.argsort(axis=1)[:, -3:][:, ::-1]
    top_3_categories = model.classes_[top_3_indices]
    top_3_probabilities = predicted_probabilities[0, top_3_indices[0]]

    # Return the result
    result = []
    for j in range(3):
        result.append({"Category": top_3_categories[0, j], "Probability": top_3_probabilities[j]})

    return {"Top3Predictions": result}
    
