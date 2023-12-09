import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Liest die CSV-Datei "emails.csv" und speichert sie in einem DataFrame
df = pd.read_csv("emails.csv")

# Nimmt alle Spalten außer der ersten und der letzten (Email No. & Prediction) aus dem DataFrame und speichert ihre Werte in X
X = df[df.columns[1:-1]].values

# Nimmt die letzte Spalte aus dem DataFrame und speichert ihre Werte in y
y = df[df.columns[-1]].values

# Initialisiert einen StandardScaler
scaler = StandardScaler()

# Passt den Scaler an die Daten in X an und transformiert X
# Dieser Schritt verändert die Werte in X so, dass sie im Durchschnitt 0 sind und eine Streuung von 1 haben.
X = scaler.fit_transform(X)

# Initialisiert ein PCA-Modell, das die Anzahl der Komponenten so wählt, dass 99% der Varianz erhalten bleiben
model = PCA(n_components=0.99)

# Passt das PCA-Modell an die Daten in X an
model.fit(X)

# Transformiert X in den durch das PCA-Modell definierten Raum
X = model.transform(X)

# Teilt die Daten in Trainings- und Testsets auf, wobei 20% der Daten für das Testset reserviert sind
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

# Initialisiert ein Wörterbuch namens 'score_data' mit fünf Schlüsseln: 'algorithms', 'train_scores', 'test_scores', 'cross_val_means' und 'cross_val_stds'.
# Jeder Schlüssel ist mit einer leeren Liste verknüpft.
# 'algorithms' wird verwendet, um die Namen der Algorithmen zu speichern.
# 'train_scores' wird verwendet, um die Trainings-Scores der Algorithmen zu speichern.
# 'test_scores' wird verwendet, um die Test-Scores der Algorithmen zu speichern.
# 'cross_val_means' wird verwendet, um die Durchschnittswerte der Kreuzvalidierungsscores der Algorithmen zu speichern.
# 'cross_val_stds' wird verwendet, um die Standardabweichungen der Kreuzvalidierungsscores der Algorithmen zu speichern.
score_data = {'algorithms': [],
              'train_scores': [],
              'test_scores': [],
              'cross_val_means': [],
              'cross_val_stds': []}

# Definiert eine Funktion namens 'getScores', die einen Algorithmusnamen als Eingabe nimmt


def getScores(name):
    # Gibt den Namen des Algorithmus aus
    print(name)
    # Fügt den Namen des Algorithmus zur Liste 'algorithms' im Wörterbuch 'score_data' hinzu
    score_data['algorithms'].append(name)

    # Trainiert das Modell mit den Trainingsdaten und berechnet den R2-Score
    train_score = r2_score(y_train, model.fit(
        X_train, y_train).predict(X_train))
    # Fügt den R2-Score zur Liste 'train_scores' im Wörterbuch 'score_data' hinzu
    score_data['train_scores'].append(train_score)
    # Gibt den R2-Score für das Training aus
    print('Train R2: {}'.format(train_score))

    # Trainiert das Modell mit den Trainingsdaten und berechnet den R2-Score für die Testdaten
    test_score = r2_score(y_test, model.fit(X_train, y_train).predict(X_test))
    # Fügt den R2-Score zur Liste 'test_scores' im Wörterbuch 'score_data' hinzu
    score_data['test_scores'].append(test_score)
    # Gibt den R2-Score für den Test aus
    print('Test R2: {}'.format(test_score))

    # Führt eine 5-fache Kreuzvalidierung durch und berechnet die R2-Scores
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # Gibt die R2-Scores der Kreuzvalidierung aus
    print('Scores: {}'.format(scores))
    # Gibt den Durchschnitt der R2-Scores der Kreuzvalidierung aus
    print('Mean score: {}'.format(scores.mean()))
    # Gibt die Standardabweichung der R2-Scores der Kreuzvalidierung aus
    print('Std score: {}'.format(scores.std()))

    # Fügt den Durchschnitt der R2-Scores zur Liste 'cross_val_means' im Wörterbuch 'score_data' hinzu
    score_data['cross_val_means'].append(scores.mean())
    # Fügt die Standardabweichung der R2-Scores zur Liste 'cross_val_stds' im Wörterbuch 'score_data' hinzu
    score_data['cross_val_stds'].append(scores.std())

    # Gibt eine leere Zeile aus, um die Ausgabe übersichtlicher zu gestalten
    print()


# Initialisiert das KNeighborsClassifier-Modell mit 2 Nachbarn (bestes Ergebnis)
model = KNeighborsClassifier(n_neighbors=2)
# Ruft die Funktion 'getScores' auf, um das Modell zu trainieren und die R2-Scores zu berechnen
getScores('kNN')

# Initialisiert das LogisticRegression-Modell mit maximal 1000 Iterationen
# Iterationen in der logistischen Regression sind Schritte in einem Optimierungsverfahren (normalerweise Gradientenabstieg),
# das versucht, die bestmöglichen Parameter für das Modell zu finden, indem es den Fehler zwischen den vorhergesagten und den tatsächlichen Werten minimiert.
model = LogisticRegression(max_iter=1000)
getScores('Log Regression')

# Initialisiert das RandomForestClassifier-Modell.
# Dies ist ein Klassifikationsalgorithmus, der eine Gruppe von Entscheidungsbäumen erstellt
# und die am häufigsten vorkommende Klasse (oder den Durchschnitt im Fall einer Regression) als Vorhersage verwendet.
model = RandomForestClassifier()
getScores('Random Forest Classifier')

# Initialisiert das SupportVectorMachine-Modell mit einem linearen Kernel
# Ein linearer Kernel in einer Support Vector Machine (SVM) bedeutet, dass die Trennung zwischen den Klassen durch eine gerade Linie (in 2D) oder eine Hyperebene (in höheren Dimensionen) erfolgt.
# Dies ist die einfachste Art von Kernel und eignet sich gut für Daten, die linear trennbar sind.
model = svm.SVC(kernel='linear')
getScores('SVM')

# Erstellen eines Modells mit dem MLPClassifier (Multi-Layer Perceptron Classifier). Dies ist ein neuronales Netzwerk, das aus mehreren Schichten von Neuronen besteht.
# Die Parameter bedeuten:
# - random_state=42: Dies stellt sicher, dass die Zufälligkeit des Modells bei jedem Lauf gleich ist, so dass die Ergebnisse reproduzierbar sind.
# - max_iter=10000: Dies ist die maximale Anzahl von Iterationen (oder Durchläufen durch das Netzwerk), die das Modell durchführen wird, um die besten Parameter zu finden.
# - hidden_layer_sizes=[100, 100]: Dies bedeutet, dass das Netzwerk zwei versteckte Schichten hat, jede mit 100 Neuronen.
model = MLPClassifier(random_state=42, max_iter=10000,
                      hidden_layer_sizes=[100, 100])
getScores('NEURAL NETWORK')

# Erstellen eines DataFrame mit den Daten aus score_data. Die Daten werden in Spalten organisiert, die 'Algorithmen', 'Train R2', 'Test R2', 'Mean' und 'Std' genannt werden.
data = pd.DataFrame(list(zip(score_data['algorithms'], score_data['train_scores'], score_data['test_scores'], score_data['cross_val_means'],
                    score_data['cross_val_stds'])), columns=['Algorithmen', 'Train R2', 'Test R2', 'Mean', 'Std'])

# Setzen der 'Algorithmen'-Spalte als Index des DataFrame. Dies bedeutet, dass die Algorithmen-Namen jetzt die Zeilenetiketten des DataFrame sind.
data.set_index('Algorithmen', inplace=True)

# Erstellen einer Heatmap mit den Daten im DataFrame. Die Zahlen in der Heatmap werden durch die Option annot=True angezeigt. Die Farbskala der Heatmap wird durch cmap='viridis' festgelegt.
sns.heatmap(data, annot=True, cmap='viridis')
# Hinzufügen eines Titels zur Heatmap.
plt.title('Vergleich der Scores verschiedener Algorithmen')
# Anzeigen der Heatmap und des Titels.
plt.show()
