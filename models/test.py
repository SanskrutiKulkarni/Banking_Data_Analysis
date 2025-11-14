import pickle

with open('models/model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['model']
print("Model loaded successfully!")
print(model)
