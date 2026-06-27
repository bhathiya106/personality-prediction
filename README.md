# Personality Prediction API

## Project Description

This project predicts whether a person is an Introvert or Extrovert using Machine Learning.

## Dataset

Personality Dataset

## Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest

The best-performing model was selected and deployed.

## Technologies

- Python
- Pandas
- Scikit-learn
- FastAPI
- Uvicorn

## Running the API

```bash
uvicorn app:app --reload
```

Open

```
http://127.0.0.1:8000/docs
```

## API Endpoint

POST

```
/predict
```

Example

```json
{
  "Time_spent_Alone": 7,
  "Stage_fear": "Yes",
  "Social_event_attendance": 2,
  "Going_outside": 1,
  "Drained_after_socializing": "Yes",
  "Friends_circle_size": 3,
  "Post_frequency": 1
}
```

Example Response

```json
{
  "prediction":"Introvert"
}
```