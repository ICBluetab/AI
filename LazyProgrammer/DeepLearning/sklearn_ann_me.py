from sklearn.neural_network import MLPClassifier
from process import get_data
from sklearn.model_selection import train_test_split

X, y = get_data()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = MLPClassifier(max_iter=700)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print("Score: ", score)
