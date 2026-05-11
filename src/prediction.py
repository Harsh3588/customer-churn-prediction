import joblib

def load_model():

    model = joblib.load("models/churn_model.pkl")

    return model


def predict_churn(model, data):

    prediction = model.predict(data)

    return prediction