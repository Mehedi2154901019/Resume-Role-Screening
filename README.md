# Resume-Role-Screening

---

##Dataset
The dataset was taken from kaggle, also uploaded [here](assets/UpdatedResumeDataSet.csv).

##Model Training
The dataset was preprocessed through removing emails, mentions, special characters, stopwords and lemmatization.
Afterwards, They were converted to tfidf vectors to have semantic meaning within the texts.
Also, detailed EDA and plots are clearly shown step by step.

The resume role categories with distribution is below:
![pie](assets/pie.png)

Then after labelencoding, the model was trained by few ML algorithms and finally LogisticRegression showed 99.48% accuracy.
For further use, reume_role_mode.pkl, tfidf_vectorizer.pkl and label_encoder.pkl files were saved. [link for pkl files](assets).
[Click here](assets/model.ipynb) to see the model training code.

##FastAPI Endpoints
The fastapi had default home endpoint 'home' and the prediction endpoint '/predict_role/'
The same text preprocessing are done on the texts extracted from pdf through PyPDF2
To run:
```Terminal 
uvicorn main:app --reload
```


